# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Utilities for models."""

import math
import functools
import torch
import torch.nn.functional as F

from megatron import get_args
from megatron.core import mpu
from megatron.core.context_parallel import dattention


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


def get_linear_layer(rows, columns, init_method):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    if get_args().perform_initialization:
        init_method(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
    return layer


@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))


def openai_gelu(x):
    return gelu_impl(x)


#This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype)+torch.ones_like(x).to(dtype=x.dtype))


def slice_lm_inputs_along_cp(input_ids, position_ids, attention_mask, labels, cp_size, packing_info=None):
    cp_group = mpu.get_context_parallel_group(cp_size)
    args = get_args()
    if cp_size >= 2:
        # Check inputs with the same context parallel rank are equal
        if args.curr_iteration < args.iteration + args.kaimm_warmup_iters:
            if input_ids is not None:
                max_input_ids = input_ids.clone()
                torch.distributed.all_reduce(max_input_ids, op=torch.distributed.ReduceOp.MAX,
                                             group=cp_group)
                if (max_input_ids != input_ids).any():
                    raise ValueError("Inputs with the same get_data_parallel_for_sample_rank() should be equal. "
                                     "Please check the dataloader.")
    if input_ids is not None:
        cp_rank = torch.distributed.get_rank(cp_group)
        if args.sft_concat:
            assert packing_info is not None
        else:
            assert packing_info is None
        input_ids = dattention.slice_cp(input_ids, 1, cp_size, cp_rank, packing_info=packing_info)
        position_ids = dattention.slice_cp(position_ids, 1, cp_size, cp_rank, packing_info=packing_info)
        labels = dattention.slice_cp(labels, 1, cp_size, cp_rank, packing_info=packing_info)

    return input_ids, position_ids, attention_mask, labels


def gather_post_lm_output_along_cp(output, cp_size, total_seq_len=None, packing_info=None):
    output = dattention.forward_gather_backward_slice(output, 1, mpu.get_context_parallel_group(cp_size))
    return dattention.recover_packed_seq(output, 1, cp_size, total_seq_len, packing_info)


@functools.lru_cache(maxsize=1)
def get_var_len_info(sample_lengths):
    return F.pad(sample_lengths.cumsum(0), (1, 0), 'constant', 0), sample_lengths.max().item()
