# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# copied from fairseq/fairseq/data/indexed_dataset.py
# Removed IndexedRawTextDataset since it relied on Fairseq dictionary
# other slight modifications to remove fairseq dependencies
# Added document index to index file and made it accessible.
#    An empty sentence no longer separates documents.

from functools import lru_cache
import os
import shutil
import struct
from itertools import accumulate

import numpy as np
import torch
from megatron import print_rank_0
import time


def get_available_dataset_impl():
    return ['lazy', 'cached', 'mmap']


def make_dataset(max_length, path, impl, skip_warmup=False):
    if not SFTMMapIndexedDataset.exists(path):
        print(f"Dataset does not exist: {path}")
        print("Path should be a basename that both .meta and .bin can be appended to get full filenames.")
        return None
    if impl == 'mmap' and SFTMMapIndexedDataset.exists(path):
        return SFTMMapIndexedDataset(max_length, path, skip_warmup)
    print(f"Unknown dataset implementation: {impl}")
    return None


def dataset_exists(path, impl):
    if impl == 'mmap':
        return SFTMMapIndexedDataset.exists(path)


def inputs_id_file_path(prefix_path):
    return prefix_path + '_input_ids.bin'


def sample_index_file_path(prefix_path):
    return prefix_path + '_sample_index.bin'


def sample_size_file_path(prefix_path):
    return prefix_path + '_sample_len.bin'


def labels_mask_file_path(prefix_path):
    return prefix_path + '_labels_mask.bin'


def create_doc_idx(sizes):
    doc_idx = [0]
    for i, s in enumerate(sizes):
        if s == 0:
            doc_idx.append(i + 1)
    return doc_idx


def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class SFTMMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00'

        def __init__(self, idx_path, size_path, skip_warmup=False):
            if not skip_warmup:
                print_rank_0("    warming up index mmap file...")
                _warmup_mmap_file(idx_path)
                _warmup_mmap_file(size_path)

            self._idx_bin_buffer_mmap = np.memmap(idx_path, mode='r', order='C')
            self._size_bin_buffer_mmap = np.memmap(size_path, mode='r', order='C')
            self._idx_bin_buffer = memoryview(self._idx_bin_buffer_mmap)
            self._size_bin_buffer = memoryview(self._size_bin_buffer_mmap)
            print_rank_0("    reading sizes...")
            self._sizes = np.frombuffer(
                self._size_bin_buffer_mmap,
                dtype=np.int32,
                offset=0)
            print_rank_0("    reading pointers...")
            self._pointers = np.frombuffer(self._idx_bin_buffer,
                                           dtype=np.int64,
                                           offset=0)
            print_rank_0("    reading document index...")
            self._doc_idx = np.arange(self._pointers.shape[0], dtype=np.int64)

        def __del__(self):
            self._idx_bin_buffer_mmap._mmap.close()
            del self._idx_bin_buffer_mmap
            self._size_bin_buffer_mmap._mmap.close()
            del self._size_bin_buffer_mmap

        @property
        def dtype(self):
            raise NotImplementedError

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._pointers.shape[0]

    def __init__(self, max_length, path, skip_warmup=False):
        super().__init__()

        self._path = None
        self._index = None
        self._input_idx_bin_buffer = None
        self._label_mask_bin_buffer = None
        self.max_length = max_length

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state, skip_warmup=True)

    def _do_init(self, path, skip_warmup):
        self._path = path
        self._index = self.Index(idx_path=sample_index_file_path(self._path),
                                 size_path=sample_size_file_path(self._path),
                                 skip_warmup=skip_warmup)

        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(inputs_id_file_path(self._path))
        print_rank_0("    creating numpy buffer of mmap...")
        self._input_idx_bin_buffer_mmap = np.memmap(inputs_id_file_path(self._path), mode='r', order='C')
        self._label_mask_bin_buffer_mmap = np.memmap(labels_mask_file_path(self._path), mode='r', order='C')
        print_rank_0("    creating memory view of numpy buffer...")
        self._input_idx_bin_buffer = memoryview(self._input_idx_bin_buffer_mmap)
        self._label_mask_bin_buffer = memoryview(self._label_mask_bin_buffer_mmap)

    def __del__(self):
        self._input_idx_bin_buffer_mmap._mmap.close()
        self._label_mask_bin_buffer_mmap._mmap.close()
        del self._input_idx_bin_buffer_mmap
        del self._label_mask_bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        """ Good Lucky !!! Never run error !!!
        """
        if isinstance(idx, slice):
            raise NotImplementedError(f"Using batch_sampler instead of smapler !!!")
        ptr, size = self._index[idx]
        ptr *= np.int32().itemsize
        input_ids = np.frombuffer(self._input_idx_bin_buffer, dtype=np.int32,
                                  count=size, offset=ptr)
        label_mask = np.frombuffer(self._label_mask_bin_buffer, dtype=np.int32,
                                   count=size, offset=ptr)

        return dict(input_ids=np.array(input_ids[:self.max_length], dtype=np.int64),
                    label_mask=np.array(label_mask[:self.max_length], dtype=np.int64))

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return (
                os.path.exists(inputs_id_file_path(path)) and os.path.exists(labels_mask_file_path(path))
                and os.path.exists(sample_index_file_path(path)) and os.path.exists(sample_size_file_path(path))
        )
