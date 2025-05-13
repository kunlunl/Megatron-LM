# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Dataloaders."""


import random
import torch
import numpy as np
from torch.utils.data import Dataset
from megatron import get_args
from megatron.core import mpu
import itertools
from megatron.global_vars import get_tokenizer, push_cached_num_microbatches
from megatron import print_rank_0
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, List
import math
from collections import deque
import os
import heapq


def build_pretraining_data_loader(dataset, consumed_samples):
    """Buld dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()

    def _init_worker(worker_id):
        np.random.seed(args.seed)

    # Virtual sampler
    if args.iterable_dataset:
        # Notice: here will use default collate_fn, see: torch/utils/data/_utils/collate.py
        if (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0:
            return torch.utils.data.DataLoader(dataset,
                                               batch_size=args.micro_batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               worker_init_fn=_init_worker,
                                               prefetch_factor=args.prefetch_factor)
        else:
            return None

    # Megatron sampler
    if args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size)
    elif args.dataloader_type == 'cyclic':
        if len(args.all_possible_context_parallel_sizes) != 1:
            # When context parallel changes, dp_for_sample also changes, so cannot calcuate the 
            # last_batch_size, therefore cannot calculate epoch and current_epoch_samples.
            raise NotImplementedError('Currently cyclic dataloader does not support hot-switch.')
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_for_sample_rank(),
            data_parallel_size=mpu.get_data_parallel_for_sample_world_size(),
            data_sharding=args.data_sharding)
    else:
        raise Exception('{} dataloader type is not supported.'.format(
                args.dataloader_type))

    # Torch dataloader.
    # Notice: here will use default collate_fn, see: torch/utils/data/_utils/collate.py
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       prefetch_factor=args.prefetch_factor)


def build_sft_data_loader(dataset):
    """Buld dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()
    tokenizer = get_tokenizer()

    sft_concat = args.sft_concat

    if sft_concat:
        print_rank_0("---------------------- using sample concat within batch sampler ----------------------")
        batch_sampler = SftConcatWithinBatchSampler(dataset,
            total_samples=len(dataset),
            consumed_samples=0,
            global_batch_size=args.global_batch_size,
            dataset_sizes=dataset.sizes,
            max_seq_len=args.seq_length)
    elif args.dataloader_type == 'single':
        print_rank_0("---------------------- using distributed sampler ----------------------")
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=0,
            micro_batch_size=args.micro_batch_size,
            sft_concat=sft_concat)
    elif args.dataloader_type == 'cyclic':
        print_rank_0("---------------------- using random distributed sampler ----------------------")
        if len(args.all_possible_context_parallel_sizes) != 1:
            # When context parallel changes, dp_for_sample also changes, so cannot calcuate the 
            # last_batch_size, therefore cannot calculate epoch and current_epoch_samples.
            raise NotImplementedError('Currently cyclic dataloader does not support hot-switch.')
        batch_sampler = MegatronSftRandomSampler(dataset,
                                                 total_samples=len(dataset),
                                                 consumed_samples=0,
                                                 micro_batch_size=args.micro_batch_size,
                                                 data_parallel_rank=mpu.get_data_parallel_for_sample_rank(),
                                                 data_parallel_size=mpu.get_data_parallel_for_sample_world_size(),
                                                 data_sharding=args.data_sharding,
                                                 sft_concat=sft_concat)
    else:
        raise NotImplementedError

    if sft_concat:
        print_rank_0("---------------------- using collate_fn for sample cncatenation ----------------------")
        collate_fn = SampleConcatDataCollatorForSupervisedDataset(tokenizer)
    elif args.sft_padding or args.micro_batch_size > 1:
        print_rank_0("---------------------- using collate_fn for padding ----------------------")
        assert args.seq_length % mpu.get_tensor_model_parallel_world_size() == 0
        collate_fn = DataCollatorForSupervisedDataset(tokenizer, args.seq_length)
    elif args.sequence_parallel and mpu.get_tensor_model_parallel_world_size() > 1:
        print_rank_0("---------------------- using collate_fn for sequence parallel ----------------------")
        collate_fn = DataCollatorForSupervisedDatasetWithOnlySP(tokenizer, mpu.get_tensor_model_parallel_world_size())
    else:
        collate_fn = None

    train_dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_sampler=batch_sampler,
                                                   collate_fn=collate_fn,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)
    train_dataloader.batch_sampler.set_num_workers_times_prefech_factor(train_dataloader.prefetch_factor
                                                                        * train_dataloader.num_workers if train_dataloader.num_workers != 0 else 0)

    print_rank_0(f"epoch size {len(train_dataloader)}")
    return train_dataloader


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    import transformers
    tokenizer: transformers.PreTrainedTokenizer
    max_length: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, label_mask = tuple([instance[key] for instance in instances] for key in ("input_ids", "label_mask"))
        for i in range(len(input_ids)):
            input_ids[i] = np.pad(input_ids[i], (0, self.max_length - input_ids[i].shape[0]),
                                  constant_values=self.tokenizer.pad_token_id)
            label_mask[i] = np.pad(label_mask[i], (0, self.max_length - label_mask[i].shape[0]), constant_values=0)
        return dict(
            input_ids=torch.as_tensor(np.stack(input_ids, axis=0)),
            label_mask=torch.as_tensor(np.stack(label_mask, axis=0))
        )


@dataclass
class SampleConcatDataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    import transformers
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        '''
        for cp&sp:
        divisor = math.lcm(2, tp_world_size) * cp_world_size
        2 stands for guassian summation

        for flip_cp_ in gather_post_lm_output_along_cp
        divisor = 4 * 2 * cp_world_size
        because its input is loss(instead of logits) with shape (seq_len, bs), in flip_cp_, we need to make
        seq_len / (2 * cp_world_size) * element_size to be divisible by element_size of uint64_t
        in this case, we use bf16 as dtype, and the element_size of uint64_t is 8, thus we need to make
        seq_len / (2 * cp_world_size) divisible by 4

        for more details, please refer to fused_kernels/fast_flip_cuda.cu:flip
        '''
        divisor = np.lcm(4, mpu.get_tensor_model_parallel_world_size()) * 2 * mpu.get_context_parallel_world_size()

        all_input_ids = []
        all_label_mask = []

        for i, instance in enumerate(instances):
            input_ids = instance["input_ids"]
            label_mask = instance["label_mask"]
            # sequence length -1 for the last instance for label shift
            seq_len = input_ids.shape[0] - int(i == len(instances) - 1)
            padding = math.ceil(seq_len / divisor) * divisor - seq_len
            if padding != 0:
                input_ids = np.pad(input_ids, (0, padding), constant_values=self.tokenizer.pad_token_id)
                label_mask = np.pad(label_mask, (0, padding), constant_values=0)
            all_input_ids.append(input_ids)
            all_label_mask.append(label_mask)

        input_ids = np.concatenate(all_input_ids, axis=0)
        label_mask = np.concatenate(all_label_mask, axis=0)
        local_lengths = [i.shape[0] for i in all_input_ids]

        return dict(
            input_ids=torch.as_tensor(np.expand_dims(input_ids, axis=0)),
            label_mask=torch.as_tensor(np.expand_dims(label_mask, axis=0)),
            sample_lengths=torch.as_tensor(local_lengths)
        )


@dataclass
class DataCollatorForSupervisedDatasetWithOnlySP(object):
    """Collate examples for supervised fine-tuning."""

    import transformers
    tokenizer: transformers.PreTrainedTokenizer
    tp_world_size: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, label_mask = tuple([instance[key] for instance in instances] for key in ("input_ids", "label_mask"))
        cur_length = input_ids[0].shape[0]
        padded_length = math.ceil(cur_length / self.tp_world_size) * self.tp_world_size
        padding_length = padded_length - cur_length
        for i in range(len(input_ids)):
            input_ids[i] = np.pad(input_ids[i], (0, padding_length), constant_values=self.tokenizer.pad_token_id)
            label_mask[i] = np.pad(label_mask[i], (0, padding_length), constant_values=0)
        return dict(
            input_ids=torch.as_tensor(np.stack(input_ids, axis=0)),
            label_mask=torch.as_tensor(np.stack(label_mask, axis=0))
        )


class VirtualPretrainingSampler:
    """
    A virtual sampler to enable self.auto_collation in torch/utils/data/_utils/collate.py
    """

    def __init__(self, micro_batch_size):
        self.micro_batch_size = micro_batch_size

        assert self.micro_batch_size > 0

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in itertools.count():
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                yield batch
                batch = []


class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size, drop_last=True, sft_concat=False):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.drop_last = drop_last
        self.sft_concat = sft_concat

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        if self.sft_concat:
            start_idx = 0
            end_idx = self.micro_batch_size * mpu.get_data_parallel_for_sample_world_size()
        else:
            start_idx = mpu.get_data_parallel_for_sample_rank() * self.micro_batch_size
            end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_size * mpu.get_data_parallel_for_sample_world_size():
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class SftConcatWithinBatchSampler:

    def __init__(self, dataset, total_samples, consumed_samples, global_batch_size, dataset_sizes,
                 max_seq_len: int = 4096, drop_last=True):
        self.dataset = dataset
        self.total_samples = total_samples  # 一个 epoch 的样本数
        self.consumed_samples = consumed_samples  # 全局已消耗的样本数
        self.global_batch_size = global_batch_size
        self.drop_last = drop_last
        self.name = "SftConcatWithinBatchSampler"
        self.dataset_sizes = dataset_sizes
        self.num_workers_times_prefech_factor = 0
        self.max_seq_len = max_seq_len
        self.num_micro_batch = None
        self.micro_batches_queue = None
        self.num_samples_global_micro_batch_queue = None
        self.consumed_samples_backoff_queue = None
        self.total_trained_tokens = 0
        self.total_trained_tokens_power2 = 0

        # 初始化 epoch 和在当前 epoch 中已消耗的样本数
        self.epoch = self.consumed_samples // self.total_samples
        self.consumed_samples_in_epoch = self.consumed_samples % self.total_samples

        assert self.total_samples > 0, 'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples_in_epoch < self.total_samples, \
            'no samples left to consume in current epoch: {}, {}'.format(self.consumed_samples_in_epoch,
                                                                         self.total_samples)
        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)
        self.predetermine_within_minibatch()
        self.total_trained_tokens = 0
        self.total_trained_tokens_power2 = 0
        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.global_batch_size > 0
    
    def set_num_workers_times_prefech_factor(self, num_workers_times_prefech_factor):
        self.consumed_samples_backoff_queue = deque(maxlen=num_workers_times_prefech_factor)

    def __len__(self):
        return self.total_samples

    def num_micro_batch(self) -> int:
        return self.num_micro_batch

    def solver(self, lengths: List[int], num_buckets: int):
        assert len(lengths) >= num_buckets, f"global batch size must be at least {num_buckets}"
        # 0 stands for total seq_len in bucket, [] stands for idx in bucket
        buckets = [[0, []] for _ in range(num_buckets)]
        heapq.heapify(buckets)
        sorted_indices = np.argsort(lengths)[::-1]
        for idx in sorted_indices:
            entry = heapq.heappop(buckets)
            entry[0] += lengths[idx]
            entry[1].append(idx)
            heapq.heappush(buckets, entry)
        return [entry[1] for entry in list(buckets)], max([bucket[0] for bucket in buckets])

    def search_for_buckets(self, batch: List[int]):
        # TODO(kunlunl): Remove this.
        # Need to replace with real scheduler logic.
        current_cp_size = mpu.get_context_parallel_world_size()
        possible_cp_sizes = mpu.get_context_parallel_all_possible_world_sizes()
        for i, cp_size in enumerate(possible_cp_sizes):
            if cp_size == current_cp_size:
                break
        next_cp_size = possible_cp_sizes[(i + 1) % len(possible_cp_sizes)]
        print("next_cp_size:", next_cp_size)

        dp_for_sample_rank = mpu.get_data_parallel_rank() // next_cp_size
        dp_for_sample_size = mpu.get_data_parallel_world_size() // next_cp_size

        lengths = self.dataset_sizes[batch]
        one_sample_per_bucket = get_args().sft_concat_mbs1
        i = 0
        assert max(lengths) <= self.max_seq_len, f"dp sample rank: {dp_for_sample_rank}: Maximum sequence length in this batch of samples exceeds max_seq_len requirement."
        while(True):
            i += 1
            if one_sample_per_bucket:
                num_buckets = get_args().global_batch_size
                indices_buckets, max_bucket_sum = self.solver(lengths, num_buckets)
                break
            else:
                num_buckets = i * dp_for_sample_size * mpu.get_pipeline_model_parallel_world_size()
                indices_buckets, max_bucket_sum = self.solver(lengths, num_buckets)
                if max_bucket_sum <= self.max_seq_len:
                    break
        num_local_buckets = len(indices_buckets) // dp_for_sample_size
        start_idx = dp_for_sample_rank * num_local_buckets
        end_idx = start_idx + num_local_buckets
        local_micro_batches = [[batch[idx] for idx in indices] for indices in indices_buckets[start_idx:end_idx]]
        num_samples_global_micro_batch = [sum(len(local_micro_batches) for local_micro_batches in indices_buckets[i::num_local_buckets]) for i in range(0, num_local_buckets)]
        assert len(local_micro_batches) == len(num_samples_global_micro_batch)
        return local_micro_batches, num_samples_global_micro_batch, next_cp_size

    # pre-determine sample arrangement within minibatch and number of micro batch 
    def predetermine_within_minibatch(self):
        self.micro_batches_queue = []
        self.num_micro_batch = 0
        self.num_samples_global_micro_batch_queue = []

        # 计算当前批次的开始和结束索引
        batch_start = self.consumed_samples_in_epoch
        batch_end = batch_start + self.global_batch_size

        if batch_end <= self.total_samples:
            batch = list(range(batch_start, batch_end))
        else:
            # 如果超出当前 epoch 的样本数，需要根据是否丢弃最后一个批次来处理
            batch = list(range(batch_start, self.total_samples))
            remaining = self.global_batch_size - len(batch)
            if not self.drop_last and remaining > 0:
                batch += list(range(0, remaining))
            elif self.drop_last:
                batch = []  # 丢弃不完整的批次

        if len(batch) == self.global_batch_size or (len(batch) > 0 and not self.drop_last):
            # 根据样本长度搜索桶（buckets）
            micro_batches, num_samples_global_micro_batch, next_cp_size = self.search_for_buckets(batch)
            self.micro_batches_queue = micro_batches
            self.num_micro_batch = len(micro_batches)
            self.num_samples_global_micro_batch_queue = num_samples_global_micro_batch
            assert self.num_micro_batch == len(self.num_samples_global_micro_batch_queue)
            if mpu.is_pipeline_last_stage():
                push_cached_num_microbatches(self.num_micro_batch, next_cp_size)
        else:
            # 当前 epoch 没有更多批次
            self.micro_batches_queue = []
            self.num_micro_batch = 0
            self.num_samples_global_micro_batch_queue = []

    def __iter__(self):
        assert self.micro_batches_queue is not None
        assert self.num_samples_global_micro_batch_queue is not None
        assert self.consumed_samples_backoff_queue is not None, "num_workers_times_prefech_factor must be set correctly!"
        while True:
            if len(self.micro_batches_queue) == 0:
                # 当前 epoch 结束，进入下一个 epoch
                self.epoch += 1
                self.consumed_samples_in_epoch = 0
                if isinstance(self.dataset, RandomSeedDataset):
                    self.dataset.set_epoch(self.epoch)
                self.predetermine_within_minibatch()
                if len(self.micro_batches_queue) == 0:
                    break  # 没有更多数据，结束迭代
            else:
                assert len(self.micro_batches_queue) == len(self.num_samples_global_micro_batch_queue)
                for micro_batch, num_samples_global_micro_batch in zip(self.micro_batches_queue, self.num_samples_global_micro_batch_queue):
                    token_lengths = [len(self.dataset[idx % self.total_samples]['input_ids']) for idx in micro_batch]
                    self.total_trained_tokens += sum(token_lengths)
                    self.total_trained_tokens_power2 += sum(length ** 2 for length in token_lengths)
                    self.consumed_samples += num_samples_global_micro_batch
                    self.consumed_samples_in_epoch += num_samples_global_micro_batch
                    self.consumed_samples_backoff_queue.append(num_samples_global_micro_batch)
                    yield micro_batch
                # 准备下一个批次
                self.predetermine_within_minibatch()

    def load_state_dict(self, state):
        sampler_state = state.get(self.name, {})
        self.epoch = sampler_state.get("epoch", 0)
        self.consumed_samples = sampler_state.get("consumed_samples", 0)
        self.consumed_samples_in_epoch = self.consumed_samples % self.total_samples
        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)
        # 预先确定下一个小批次
        self.predetermine_within_minibatch()

    def state_dict(self):
        state_dict = {self.name: {}}
        state_dict[self.name]["epoch"] = self.epoch
        consumed_samples_backoff = sum(self.consumed_samples_backoff_queue) if self.consumed_samples_backoff_queue else 0
        state_dict[self.name]["consumed_samples"] = self.consumed_samples - consumed_samples_backoff
        return state_dict


class RandomSeedDataset(Dataset):

    def __init__(self, dataset):
        args = get_args()
        self.base_seed = args.seed
        self.curr_seed = args.seed
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.curr_seed = self.base_seed + epoch

    def __getitem__(self, idx):
        seed = idx + self.curr_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return self.dataset[idx]


class MegatronPretrainingRandomSampler:

    def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, data_sharding):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                           * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size
            
            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) \
                                * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = \
                torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.data_parallel_rank::self.data_parallel_size]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []


class MegatronSftRandomSampler:

    def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, data_sharding, num_workers_times_prefech_factor=1, sft_concat=False):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.name = "MegatronSftRandomSampler"
        self.offset_inner_epoch = 0
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size
        self.sft_concat = sft_concat
        self.total_trained_tokens = 0
        self.total_trained_tokens_power2 = 0
        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)
        self.num_workers_times_prefech_factor = 0

    def __len__(self):
        return self.total_samples

    def set_num_workers_times_prefech_factor(self, num_workers_times_prefech_factor):
        self.num_workers_times_prefech_factor = num_workers_times_prefech_factor

    def __iter__(self):
        self.active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // self.active_total_samples
        current_epoch_samples = self.consumed_samples % self.active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling

        if self.sft_concat:
            sample_size = self.micro_batch_times_data_parallel_size
            start_idx = 0
            sample_step = 1
        else:
            sample_size = self.micro_batch_size
            start_idx = self.data_parallel_rank
            sample_step = self.data_parallel_size

        if self.data_sharding:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(self.active_total_samples, generator=g).tolist()
            random_idx_micro_batch_bucket = [random_idx[sample_size * i: sample_size * (i + 1)]
                                             for i in range(start_idx,
                                                            self.active_total_samples // self.micro_batch_size,
                                                            sample_step)
                                             ]
            idx_range = [idx for micro_batch_bucket in random_idx_micro_batch_bucket for idx in micro_batch_bucket]

        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) \
                                * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = \
                torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[start_idx::sample_step]

        for i in range(0, self.num_workers_times_prefech_factor):  # return prefech samples
            if (self.offset_inner_epoch > 0):
                self.offset_inner_epoch -= 1
                self.consumed_samples -= self.micro_batch_times_data_parallel_size

        batch = []
        # Last batch if not complete will be dropped.
        # for idx in idx_range:
        offset = self.offset_inner_epoch
        for ii in range(offset * sample_size, len(idx_range)):
            idx = idx_range[ii]
            batch.append(idx)
            length = len(self.dataset[idx]['input_ids'])
            self.total_trained_tokens += length
            self.total_trained_tokens_power2 += (length*length)
            if len(batch) == sample_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                self.offset_inner_epoch += 1
                if (self.offset_inner_epoch * sample_size >= len(idx_range)):
                    self.offset_inner_epoch = 0
                yield batch
                batch = []
        self.offset_inner_epoch = 0

    def load_state_dict(self, state):
        self.epoch = state[self.name]["epoch"]
        self.offset_inner_epoch = state[self.name]["offset_inner_epoch"]
        self.consumed_samples = state[self.name]["consumed_samples"]

    def state_dict(self):
        state_dict = {self.name: {}}
        state_dict[self.name]["epoch"] = self.epoch
        state_dict[self.name]["offset_inner_epoch"] = self.offset_inner_epoch
        state_dict[self.name]["consumed_samples"] = self.consumed_samples
        return state_dict
