#!/usr/bin/bash

set -e

model_dir=$1
target_jsonl=$2
vllm_options=${@:3:($#-2)}

source venv/bin/activate

eval_jsonl=${model_dir}/${target_jsonl}/completion.jsonl

if [[ $model_dir == *"Llama-3"* ]]; then
    python -m llmpp.completion_vllm --m ${model_dir} --i ${target_jsonl} \
        --ct '^\{\{- bos_token \}\}\n' ${vllm_options}
elif [[ $model_dir == *"gemma-2"* ]] || [[ $model_dir == *"Llama-2"* ]] || [[ $model_dir == *"llm-jp-"* ]]; then
    python -m llmpp.completion_vllm --m ${model_dir} --i ${target_jsonl} \
        --rsr ${vllm_options}
else
    python -m llmpp.completion_vllm --m ${model_dir} --i ${target_jsonl} \
        ${vllm_options}
fi

if [[ "${target_jsonl}" == *-linearized-* ]]; then
  option="--r"
else
  option=""
fi
python -m llmpp.eval ${option} ${eval_jsonl}
