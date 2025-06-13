# llmpp

`LLM Prompt-based Parsing (llmpp)` - A library for fine-tuning and evaluating the LLMs on prompt-based dependency parsing tasks.

This repository contains implementations for training, inference, and evaluation of thep roposed method in the paper ["Step-by-step Instructions and a Simple Tabular Output Format Improve the Dependency Parsing Accuracy of LLMs."](https://arxiv.org/abs/2506.09983), accepted to SyntaxFest 2025.

## Environment

- Ubuntu 22.04
- CUDA 12.1
- Python 3.11 or later
  - [requirements.txt](./requirements.txt) for base environment.

- Create venv and install requirements
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Download Models

```bash
huggingface-cli download google/gemma-2-9b --local-dir models/gemma-2-9b --token {your_hf_token}
```

## Dataset

### Download and JSONL Generation

See [download_ud.sh](./download_ud.sh) and [llmpp/conllu_to_prompt.py](./llmpp/conllu_to_prompt.py).

You need to accept the lLicense for each UD dataset.

#### Process Each Language

```bash
./download_ud.sh UD_English-EWT en_ewt r2.15
for s in train dev test
do
  python -m llmpp.conllu_to_prompt templates/en-turn1-step3.toml data/en_ewt-r2.15/en_ewt-ud-$s.conllu LANGUAGE English
done
```

## LoRA SFT

### TRL

See [llmpp/sft_trl.py](./llmpp/sft_trl.py).

#### Training without DeepSpeed

Using default settings:

```bash
CUDA_VISIBLE_DEVICES=0 python -m llmpp.sft_trl --c config/gemma-2-9b.yaml
```

Specifying model and train data path:

```bash
CUDA_VISIBLE_DEVICES=0 python -m llmpp.sft_trl \
  --c config/gemma-2-9b.yaml \
  --m models/gemma-2-2b \
  --t data/ja_gsd-r2.15/en-turn1-step3.train.jsonl
```

#### Training with DeepSpeed

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --main_process_port 29500 \
  --config_file config/accelerate_deepspeed_zero3_4gpu.yaml \
  -m llmpp.sft_trl \
  --c config/gemma-2-9b.yaml
```

### Unsloth

Unsolth can improve training throughput by up to 25%-40%.

See [llmpp/sft_unsloth.py](./llmpp/sft_unsloth.py).

- Create venv_unsloth and install unsloth:
```bash
python -m venv venv_unsloth
source venv_unsloth/bin/activate
pip install -r requirements_unsloth.txt
```

- Set the batch size to twice the size for TRL:
```bash
source venv_unsloth/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m llmpp.sft_unsloth --c config/Llama-3.1-8B.yaml --b 4
CUDA_VISIBLE_DEVICES=1 python -m llmpp.sft_unsloth --c config/Qwen2.5-7B.yaml --b 4
CUDA_VISIBLE_DEVICES=2 python -m llmpp.sft_unsloth --c config/gemma-2-9b.yaml --b 2

### OpenAI API Fine-tuning Settings

The standard hyper-parameters are:
- Epochs: 2
- Batch size: 9 (default)
- LR multiplier: 1.8 (default)

## Inference

### vLLM

See [llmpp/completion_vllm.py](./llmpp/completion_vllm.py).

```bash
source venv_vllm/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m llmpp.completion_vllm \
  --m models/gemma-2-9b_en_ewt-r2.15_en_turn1_step3.train/ \
  --i data/en_ewt-r2.15/en-turn1.step3.test.jsonl \
  --rsr
```

For above example, the output directory is `models/gemma-2-9b_en_ewt-r2.15_en_turn1_step3.train/data/en_ewt-r2.15/en-turn1-step3.test.jsonl/` and the output files are:
- Inference results
  - `completion.jsonl`
- Log
  - `completion.jsonl.log`

The option `--rtr` is required only for `gemma-2` models.

If you want to avoid bos_token duplication, add `-ct '^\{\{-? bos_token \}\}\n'` to the CLI options.

### OpenAI API

See [llmpp/completion_openai.py](./llmpp/completion_openai.py).

```bash
python -m llmpp.completion_openai -m ft:gpt-4o-mini-2024-07-18:your-account:someprefix:somedigest -i data/en_ewt-r2.15/en-turn1-step3.test.jsonl
```

## Evaluation

See [llmpp/eval.py](./llmpp/eval.py).

```bash
python -m llmpp.eval models/gemma-2-9b_en_ewt-r2.15_en-turn1-step3.train/data/en_ewt-r2.15/en-turn1-step3.test.jsonl/completion.jsonl
```

For above example, the output files will be placed in the same directory of the input file.
- Evaluation result
  - `completion.eval.json`
- Evaluation report
  - `completion.eval.report`

### Aggregate eval JSONs to TSV

```bash
./aggregate_eval_json_to_tsv.sh models/*_en_ewt-r2.15_en-turn1-step3.train/data/en_ewt-r2.15/en-turn1-step3.test.jsonl/completion.eval.json
```

# Citation

```bibtex
@misc{matsuda2025llmparsing,
      title={Step-by-step Instructions and a Simple Tabular Output Format Improve the Dependency Parsing Accuracy of LLMs}, 
      author={Hiroshi Matsuda and Chunpeng Ma and Masayuki Asahara},
      year={2025},
      eprint={2506.09983},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.09983}, 
}
```
