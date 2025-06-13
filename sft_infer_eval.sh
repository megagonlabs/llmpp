#!/bin/bash

model=$1
template=$2
datasets=${@:3:($#-2)}
sft_method=llmpp.sft_unsloth

for dataset in ${datasets}
do
  python -m ${sft_method} \
    --c config/${model}.yaml \
    --t data/${dataset}/${template}.train.jsonl

  ./infer_and_eval.sh \
    models/${model}_${dataset}_${template}.train \
    data/${dataset}/${template}.test.jsonl
done
