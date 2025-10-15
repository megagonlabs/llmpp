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
force=0

echo sft_method: ${sft_method}
echo targets: ${targets}

setup=1
attr=""
prev_setup=1
prev_attr=""
for target in ${targets}
do
  if [[ "${target}" == "--force" ]]; then
    echo set force
    force=1
    continue
  elif [[ ${target} == "--"* ]]; then
    prev_attr=${attr}
    prev_setup=${setup}
    attr=${target#--}
    setup=1
    continue
  elif [[ ${target} == "-" ]]; then
    setup=0
  elif [[ ${target} == "-"* ]]; then
    prev_attr=${attr}
    prev_setup=${setup}
    attr=${target#-}
    setup=0
    continue
  elif [[ "${attr}" == "c" ]]; then
    config=${target}
    echo config=${config}
  elif [[ "${attr}" == "m" ]]; then
    model=${target%/}
    echo model=${model}
  elif [[ "${attr}" == "d" ]]; then
    dataset=${target#data/}
    dataset=${dataset%%/}
    echo dataset=data/${dataset}
  elif [[ "${attr}" == "t" ]]; then
    template=${target}
    echo template=${template}
  elif [[ "${attr}" == "b" ]]; then
    batch_size="--b ${target}"
    echo batch_size=${target}
    attr=${prev_attr}
    if [[ ${setup} -eq 1 ]]; then
      setup=${prev_setup}
      continue
    else
      setup=${prev_setup}
    fi
  elif [[ "${attr}" == "lr" ]]; then
    lr="--lr ${target}"
    echo lr=${target}
    lr_suffix=-lr${target}
    attr=${prev_attr}
    if [[ ${setup} -eq 1 ]]; then
      setup=${prev_setup}
      continue
    else
      setup=${prev_setup}
    fi
  elif [[ "${attr}" == "e" ]]; then
    epoch="--e ${target}"
    echo epoch=${target}
    epoch_suffix=-epoch${target}
    attr=${prev_attr}
    if [[ ${setup} -eq 1 ]]; then
      setup=${prev_setup}
      continue
    else
      setup=${prev_setup}
    fi
  else
    echo "invalid attribute ${attr}"
    exit 1
  fi
  if [[ ${setup} -eq 1 ]]; then
    continue
  fi

  peft_dir=${model}${lr_suffix}${epoch_suffix}_${dataset}_${template}.train
  train_jsonl=data/${dataset}/${template}.train.jsonl
  test_jsonl=data/${dataset}/${template}.test.jsonl
  set +e
  merge_lora_weights=`grep '"merge_lora_weights"' ${config}`
  set -e
  if [[ ${merge_lora_weights} ]]; then
    result_dir=${peft_dir}.merged
  else
    result_dir=${peft_dir}
  fi

  if [[ ${force} -eq 0 ]] && [[ -f ${result_dir}/${test_jsonl}/completion.jsonl ]]; then
    echo skip training due to existence of ${result_dir}/${test_jsonl}/completion.jsonl
    continue
  fi

  if [[ ${force} -eq 0 ]] && [[ -f ${peft_dir}/adapter_config.json ]]; then
    echo use existing ${peft_dir}/
  elif [[ ${batch_size} == "" ]]; then
    python -m ${sft_method} ${lr} ${epoch} --c ${config} --m ${model} --t ${train_jsonl}
  else  # recovering from CUDA OOM
    set +e
    bs_origin=${batch_size#--b }
    for ((bs = ${bs_origin}; bs >= 1; bs--)); do
      if [[ ${bs} -lt ${bs_origin} ]]; then
        echo retrying with setting batch_size=${bs} ...
      fi
      if [[ ${bs} -eq 1 ]]; then
        set -e
      fi
      python -m ${sft_method} --b ${bs} ${lr} ${epoch} --c ${config} --m ${model} --t ${train_jsonl}
      if [[ $? -eq 0 ]]; then
        set -e
        break
      fi
    done
  fi

  if [[ ${merge_lora_weights} ]] ; then
    python -m llmpp.merge_peft_model ${peft_dir} local_models/${peft_dir}.merged
    ./infer_and_eval.sh local_models/${peft_dir}.merged ${test_jsonl}
    rm -f local_models/${peft_dir}.merged/model*.safetensors local_models/${peft_dir}.merged/model.safetensors.index.json
    mv local_models/${peft_dir}.merged ${peft_dir}.merged
  else
    ./infer_and_eval.sh ${result_dir} ${test_jsonl}
  fi
done
