#!/bin/bash

sft_method=llmpp.sft_$1
model=$2
template=$3
datasets=${@:4:($#-3)}

for dataset in ${datasets}
do
  python -m ${sft_method} \
    --c config/${model}.yaml \
    --t data/${dataset}/${template}.train.jsonl

  if [[ "${template}" == *linearized* ]]; then
    option="--r"
  else
    option=""
  fi

  ./infer_and_eval.sh \
    models/${model}_${dataset}_${template}.train \
    data/${dataset}/${template}.test.jsonl ${option}
done
