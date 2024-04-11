# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Llama style dataset."""

import hashlib
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from megatron import print_rank_0
from megatron.data.sft_indexed_dataset import make_dataset as make_sft_dataset


def build_train_valid_test_datasets(data_impl,
                                    seq_length,
                                    micro_batch_size,
                                    skip_warmup,
                                    train_data_prefix=None,
                                    valid_data_prefix=None,
                                    test_data_prefix=None):
    """Build train, valid, and test datasets."""

    print_rank_0("Separate data paths provided for train, valid & test. Split string will be ignored.")

    train_dataset, valid_dataset, test_dataset = None, None, None
    # Single dataset.
    if train_data_prefix is not None:
        train_dataset = build_dataset("train", train_data_prefix, data_impl,
                                      seq_length, micro_batch_size, skip_warmup)

    if valid_data_prefix is not None:
        valid_dataset = build_dataset("valid", valid_data_prefix, data_impl,
                                      seq_length, micro_batch_size, False)
    if test_data_prefix is not None:
        test_dataset = build_dataset("test", test_data_prefix, data_impl,
                                     seq_length, micro_batch_size, False)

    return (train_dataset, valid_dataset, test_dataset)


def build_dataset(dataset_name, data_prefix, data_impl,
                  seq_length, micro_batch_size, skip_warmup):
    dataset = None
    if len(data_prefix) == 1:
        batch_size_per_iter = micro_batch_size
        dataset = _build_dataset(dataset_name,
                                 data_prefix[0], data_impl,
                                 seq_length,
                                 batch_size_per_iter,
                                 skip_warmup)
    else:
        raise NotImplementedError

    return dataset


def _build_dataset(dataset_name, data_prefix, data_impl,
                   seq_length, batch_size_per_iter, skip_warmup):
    """
    Build dataset. This method is called when individual
    train, valid, test datasets are provided
    """
    # sft dataset.
    start_time = time.time()

    dataset = make_sft_dataset(seq_length, data_prefix, data_impl)
    print_rank_0(' > finished creating sft dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))

    total_samples = len(dataset)

    print_rank_0('    {}/{}:'.format(dataset_name, data_prefix))
    print_rank_0('    number of sample: {}'.format(total_samples))

    return dataset
