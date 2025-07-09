# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.data.data_samplers import RecomputeMethod, get_recompute_method
from megatron.model import GPTModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group


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

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    v_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    if v_rank is not None and not (pp_rank == 0 and v_rank == 0) and \
            not (pp_rank == args.pipeline_model_parallel_size - 1 and v_rank == args.virtual_pipeline_model_parallel_size - 1) and \
            args.use_flash_attn:
        tokens, labels, loss_mask, attention_mask, position_ids = None, None, None, None, None
        return tokens, labels, loss_mask, attention_mask, position_ids

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, output_tensor, cp_size):
    mbs = get_args().micro_batch_size
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    # TODO(hot-switch): Double check if the loss coefficient is correct.
    # Original: loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum() * mbs

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss], mbs, cp_size)

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    cp_size = mpu.get_context_parallel_world_size()
    # print(f"rank={torch.distributed.get_rank()}, using cp_size: {cp_size}")

    # TODO(hot-switch-recompute): Remove it.
    num_layers = args.num_layers
    recompute_args = [
        {"recompute_norm": False, "recompute_activations": False, "recompute_granularity": None,        "recompute_method": None,    },
        {"recompute_norm": True,  "recompute_activations": False, "recompute_granularity": "selective", "recompute_method": None,    },
        {"recompute_norm": False, "recompute_activations": True,  "recompute_granularity": "selective", "recompute_method": None,    },
        {"recompute_norm": False, "recompute_activations": False, "recompute_granularity": "full",      "recompute_method": "uniform"},
        {"recompute_norm": False, "recompute_activations": False, "recompute_granularity": "full",      "recompute_method": "block"  },
    ]
    recompute_methods = [RecomputeMethod(**recompute_args[i % len(recompute_args)]) for i in range(num_layers)]
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
    recompute_methods_this_mbs = get_recompute_method(recompute_methods, pp_rank, pp_size, vpp_rank, vpp_size, args.num_layers_per_virtual_pipeline_stage, num_layers)

    output_tensor = model(
        tokens,
        position_ids,
        attention_mask,
        cp_size=cp_size,
        recompute_methods=recompute_methods_this_mbs,
        labels=labels
    )

    return output_tensor, partial(loss_func, loss_mask, cp_size=cp_size)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        data_cache_path=args.data_cache_path)
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
