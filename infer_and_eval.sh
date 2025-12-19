#!/usr/bin/bash

set -e

model_dir=$1
target_jsonl=$2
eval_type=$3
if [ -z ${eval_type} ]; then
  eval_type=eval
fi
vllm_options=${@:4:($#-3)}

source venv/bin/activate

eval_jsonl=${model_dir}/${target_jsonl}/completion.jsonl

if [[ $model_dir == *"Llama-3"* ]]; then
    python -m llmpp.completion_vllm --m ${model_dir} --i ${target_jsonl} \
        --ct '^\{\{- bos_token \}\}\n' ${vllm_options}
elif [[ $model_dir == *"gemma-2"* ]]; then
    python -m llmpp.completion_vllm --m ${model_dir} --i ${target_jsonl} \
        --rsr ${vllm_options}
elif [[ $model_dir == *"Llama-2"* ]] || [[ $model_dir == *"llm-jp-"* ]] || [[ $model_dir == *"OLMo-"* ]]; then
    python -m llmpp.completion_vllm --m ${model_dir} --i ${target_jsonl} \
        --rsr --max_tokens 4096 ${vllm_options}
else
    python -m llmpp.completion_vllm --m ${model_dir} --i ${target_jsonl} \
        ${vllm_options}
fi

if [[ "${target_jsonl}" == *-linearized-* ]] || [[ "${target_jsonl}" == *-bracketing-* ]]; then
  if [[ "${target_jsonl}" == *no-terminal* ]]; then
    opetion="--r --nt"
  else
    option="--r"
  fi
else
  option=""
fi
python -m llmpp.${eval_type} ${option} ${eval_jsonl}
