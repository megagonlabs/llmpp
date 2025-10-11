#!/bin/bash

set -eu

sft_method=llmpp.sft_$1
targets=${@:2:($#-1)}
config=""
model=""
dataset=""
template=""
batch_size=""
epoch=""
epoch_suffix=""
lr=""
lr_suffix=""

echo sft_method: ${sft_method}
echo targets: ${targets}

setup=1
attr=""
prev_attr=""
for target in ${targets}
do
  if [[ ${target} == "-" ]]; then
    setup=$((1-setup))
    continue
  fi
  if [[ ${target} == "--"* ]]; then
    prev_attr=${attr}
    attr=${target}
    continue
  fi
  if [[ "${attr}" == "--c" ]]; then
    config=${target}
    echo config=${config}
  elif [[ "${attr}" == "--m" ]]; then
    model=${target%/}
    echo model=${model}
  elif [[ "${attr}" == "--d" ]]; then
    dataset=${target}
    echo dataset=data/${dataset}
  elif [[ "${attr}" == "--t" ]]; then
    template=${target}
    echo template=${template}
  elif [[ "${attr}" == "--b" ]]; then
    batch_size="--b ${target}"
    echo batch_size=${target}
    attr=${prev_attr}
    continue
  elif [[ "${attr}" == "--lr" ]]; then
    lr="--lr ${target}"
    echo lr=${target}
    lr_suffix=-lr${target}
    attr=${prev_attr}
    continue
  elif [[ "${attr}" == "--e" ]]; then
    epoch="--e ${target}"
    echo epoch=${target}
    epoch_suffix=-epoch${target}
    attr=${prev_attr}
    continue
  else
    echo "invalid attribute ${attr}"
    exit 1
  fi
  if [ ${setup} -eq 1 ]; then
    continue
  fi

  peft_dir=${model}${lr_suffix}${epoch_suffix}_${dataset}_${template}.train
  train_jsonl=data/${dataset}/${template}.train.jsonl
  test_jsonl=data/${dataset}/${template}.test.jsonl
  set +e
  merge_lora_weights=`grep '"merge_lora_weights"' ${config}`
  set -e
  if [ ${merge_lora_weights} ]; then
    result_dir=${peft_dir}.merged
  else
    result_dir=${peft_dir}
  fi

  if [ -f ${result_dir}/${test_jsonl}/completion.jsonl ]; then
    echo skip training due to existence of ${result_dir}/${test_jsonl}/completion.jsonl
    continue
  fi

  if [ -f ${peft_dir}/adapter_config.json ]; then
    echo use existing ${peft_dir}/
  else
    python -m ${sft_method} ${batch_size} ${lr} ${epoch} --c ${config} --m ${model} --t ${train_jsonl}
  fi

  if [ ${merge_lora_weights} ] ; then
    python -m llmpp.merge_peft_model ${peft_dir} local/${peft_dir}.merged
    ./infer_and_eval.sh local/${peft_dir}.merged ${test_jsonl}
    rm -f local/${peft_dir}.merged/model*.safetensors local/${peft_dir}.merged/model.safetensors.index.json
    mv local/${peft_dir}.merged ${peft_dir}.merged
  else
    ./infer_and_eval.sh ${result_dir} ${test_jsonl}
  fi
done
