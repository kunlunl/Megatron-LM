#!/bin/bash

# Copyright (c) 2024, Kuaishou Technology. All rights reserved.

set -euo pipefail

source ./llama-7b

export SEQ_LENGTH=100000
export GLOBAL_BATCH_SIZE=16

export HOSTFILE=
export MASTER_ADDR=127.0.0.1
export NUM_GPUS=8

export TP=1
export CP=2
export ALL_CP="2"
export PP=2
export CKPT=full
export OFFLOAD_ALPHA=0.0
export NUM_LAYERS=32

./sft_llama_no_vpp.sh
