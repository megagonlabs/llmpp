#!/bin/bash

set -eu

sft_method=llmpp.sft_$1
model=$2
dataset=$3
templates=${@:4:($#-3)}
batch_size=""

for template in ${templates}
do
  if [[ "${template}" == "--b" ]]; then
    batch_size="--b"
    continue
  elif [[ "${batch_size}" == "--b" ]]; then
    batch_size="--b ${template}"
    continue
  fi
  python -m ${sft_method} ${batch_size} \
    --c config/${model}.yaml \
    --t data/${dataset}/${template}.train.jsonl

  ./infer_and_eval.sh \
    models/${model}_${dataset}_${template}.train \
    data/${dataset}/${template}.test.jsonl
done
