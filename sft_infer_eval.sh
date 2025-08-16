#!/bin/bash

set -eu

sft_method=llmpp.sft_$1
config=$2
model=$3
dataset=$4
templates=${@:5:($#-4)}
batch_size=""

model=${model%/}

echo sft_method: ${sft_method}
echo config: ${config}
echo model: ${model}
echo dataset: data/${dataset}/
echo templates: ${templates}

for template in ${templates}
do
  if [[ "${template}" == "--m" ]]; then
    model="--m"
    continue
  elif [[ "${model}" == "--m" ]]; then
    model=${template%/}
    echo model changed: ${model}
    continue
  fi
  if [[ "${template}" == "--d" ]]; then
    dataset="--d"
    continue
  elif [[ "${dataset}" == "--d" ]]; then
    dataset=${template}
    echo dataset changed: data/${dataset}
    continue
  fi
  if [[ "${template}" == "--b" ]]; then
    batch_size="--b"
    continue
  elif [[ "${batch_size}" == "--b" ]]; then
    batch_size="--b ${template}"
    echo batch size changed: ${batch_size}
    continue
  fi

  peft_dir=${model}_${dataset}_${template}.train
  train_jsonl=data/${dataset}/${template}.train.jsonl
  test_jsonl=data/${dataset}/${template}.test.jsonl
  merge_lora_weights=`grep '"merge_lora_weights"' ${config}`
  if [ ${merge_lora_weights} ]; then
    result_dir=${peft_dir}.merged
  else
    result_dir=${peft_dir}
  fi

  if [ -f ${result_dir}/${test_jsonl}/completion.eval.json ]; then
    echo skip training due to existence of ${peft_dir}/${test_jsonl}/completion.eval.json
    continue
  fi

  if [ -f ${peft_dir}/adapter_config.json ]; then
    echo use existing ${result_dir}/
  else
    python -m ${sft_method} ${batch_size} --c ${config} --m ${model} --t ${train_jsonl}
  fi

  if [ ${merge_lora_weights} ] ; then
    python -m llmpp.merge_peft_model ${peft_dir}
  fi

  ./infer_and_eval.sh ${result_dir} ${test_jsonl}

  if [ ${merge_lora_weights} ]; then
    rm -f ${result_dir}/model*.safetensors ${result_dir}/model.safetensors.index.json
  fi
done
