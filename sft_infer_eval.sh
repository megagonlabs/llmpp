#!/bin/bash

sft_method=llmpp.sft_$1
model=$2
dataset=$3
templates=${@:4:($#-3)}

for template in ${templates}
do
  python -m ${sft_method} \
    --c config/${model}.yaml \
    --t data/${dataset}/${template}.train.jsonl

  ./infer_and_eval.sh \
    models/${model}_${dataset}_${template}.train \
    data/${dataset}/${template}.test.jsonl
done
