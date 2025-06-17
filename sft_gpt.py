# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain Llama"""

import torch
import torch.nn.functional as F
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import get_num_microbatches
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.sft_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel
from megatron.model.utils import get_var_len_info
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from collections import OrderedDict
import numpy as np
import os,sys
from megatron.core import mpu
# TODO(hot-switch): Remove or uncomment this.
# from mlflow.ksmlflow_runner import mlflowRunner, mlflowAsyncRunner, AsyncType
import os
from pathlib import Path
import json
import shutil

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.

    keys = ['input_ids', 'label_mask']
    if args.sft_concat:
        keys.append('sample_lengths')
        keys.append('cp_size')
    datatype = torch.int64
    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    packing_info = None
    attention_mask = None
    cp_size = None
    # Unpack.
    if args.sft_concat:
        tokens = data_b['input_ids'][:, :-1].contiguous()
        labels = data_b['input_ids'][:, 1:].contiguous()
        cp_size = data_b['cp_size'].item()
        sample_lengths = data_b['sample_lengths']
        sample_lengths[-1] -= 1
        cu_seq_lens, max_seq_len = get_var_len_info(sample_lengths.cuda())

        loss_mask = data_b['label_mask'][:, 1:]
        ignore_indices = cu_seq_lens[:-1] - 1
        loss_mask.index_fill_(-1, ignore_indices, 0)

        total_seq_len = sample_lengths.sum().item()
        assert total_seq_len == tokens.shape[1]

        packing_info = {
            "sample_lengths": sample_lengths.cpu(),
            "cu_seq_lens": cu_seq_lens.int(),
            "max_seq_len": max_seq_len,
            "num_samples": len(sample_lengths),
            "total_seq_len": total_seq_len
        }

    elif args.sft_padding or args.micro_batch_size > 1 or (args.sequence_parallel and mpu.get_tensor_model_parallel_world_size() > 1):
        tokens = data_b['input_ids'][:, :].contiguous()
        labels = torch.cat([tokens[:, 1:], torch.ones_like(tokens[:, 0].unsqueeze(1))], dim=-1).contiguous()
        attention_mask = tokens.ne(tokenizer.pad_token_id)
        label_mask = data_b['label_mask']
        loss_mask = torch.cat([label_mask[:, 1:], torch.zeros_like(label_mask[:, 0].unsqueeze(1))], dim=-1)
    else:
        tokens = data_b['input_ids'][:, :-1].contiguous()
        labels = data_b['input_ids'][:, 1:].contiguous()
        attention_mask = None
        loss_mask = data_b['label_mask'][:, 1:]

    assert cp_size > 0
    print(f"rank={torch.distributed.get_rank()}, using cp_size: {cp_size}")

    position_ids = torch.arange(tokens.shape[-1], dtype=torch.long,
                                device=tokens.device).unsqueeze(0)

    return tokens, labels, loss_mask, packing_info, attention_mask, position_ids, cp_size


def loss_func(loss_mask, output_tensor, cp_size, packing_info=None):
    losses = output_tensor.float()
    loss_mask = loss_mask.float()

    args = get_args()

    if args.sft_concat:
        sample_indices = packing_info["cu_seq_lens"].long().tolist()
        mbs = len(sample_indices) - 1
        loss_pairs = [((losses[..., start_idx:end_idx] * loss_mask[..., start_idx:end_idx]).sum(-1), loss_mask[..., start_idx:end_idx].int().abs().sum(-1)) for start_idx, end_idx in zip(sample_indices[:-1], sample_indices[1:])]
        loss_group = [loss_sum / valid_tokens if valid_tokens.item() != 0 else loss_sum for loss_sum, valid_tokens in loss_pairs]
        # TODO(hot-switch): Double check if the loss coefficient is correct.
        # Original: loss = torch.cat(loss_group).sum() * (get_num_microbatches() * mpu.get_data_parallel_for_sample_world_size() / args.global_batch_size)
        loss = torch.cat(loss_group).sum()
    else:
        mbs = losses.shape[0]
        losses = torch.sum(losses * loss_mask, dim=-1)
        loss_mask = loss_mask.int().abs().sum(-1)
        # the second arg(i.e. 1) stands for other, not inputs(so weird!)
        loss_mask = loss_mask.where(loss_mask != 0, 1)
        loss_group = losses / loss_mask
        # TODO(hot-switch): Double check if the loss coefficient is correct.
        # Original: loss = loss_group.mean()
        loss = loss_group.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss], mbs, cp_size)

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, packing_info, attention_mask, position_ids, cp_size = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask, cp_size=cp_size,
                          labels=labels, packing_info=packing_info)

    return output_tensor, partial(loss_func, loss_mask, cp_size=cp_size, packing_info=packing_info)


def no_wd_decay_cond(param_name, param):
    """
    Defines whether the parameter requires weight decay.
    if param do not require weight decay, return 'True', otherwise return 'False'
    """
    # do not regularize biases nor Norm parameters
    if param_name.endswith(".bias") or len(param.shape) == 1:
        no_wd = True
    else:
        no_wd = False
    return no_wd


def get_mlflow_data(runner=None, **kwargs): 
    args = get_args()
    # log training info
    save_interval = args.save_interval
    global_step = kwargs['step']
    loss = kwargs['loss']
    grad_norm = kwargs['grad_norm']
    # num_tokens = kwargs['past_num_tokens']
    learning_rate = kwargs['learning_rate']

    log = {
        'loss': loss,
        'grad_norm': grad_norm,
        'lr': learning_rate
    }
    # 'num_tokens': num_tokens

    # mlogger.log_mlflow_async(global_step, log, runner)
    runner.setLogMetric(log, global_step)

def train_valid_test_datasets_provider():
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for Llama ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_impl=args.data_impl,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path)
    print_rank_0("> finished creating Llama datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'NullTokenizerSft'})
