import json
import os
import re
from argparse import ArgumentParser
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.entrypoints.chat_utils import (apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         parse_chat_messages,
                                         resolve_chat_template_content_format)
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizer import MistralTokenizer
from vllm.v1.engine.llm_engine import LLMEngine as LLMEngineV1

from .completion_base import execute_completions, parse_args
from .utils import create_system_role_replaced_tempfiles


def main():
    parser = ArgumentParser()
    parser.add_argument("--gpu_memory_utilization", "--gmu", default=0.9, type=float)
    parser.add_argument("--tensor_parallel_size", "--tp", default=1, type=int)
    parser.add_argument("--num_scheduler_steps", "--ss", default=8, type=int)
    parser.add_argument("--enable_prefix_caching", "--pc", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--quantization", "--q")
    parser.add_argument("--max_lora_rank", "--mlr", default=8, type=int)
    parser.add_argument("--load_format", "-lf", default="auto")
    args = parse_args(parser)
    if "-bnb-" in args.model_name and not args.quantization and args.load_format == "auto":
        args.quantization = "bitsandbytes"
        args.load_format = "bitsandbytes"
    if "-4bit" in args.model_name or  "-8bit" in args.model_name:
        args.dtype = "auto"

    input_jsonl = create_system_role_replaced_tempfiles(args.input_jsonl) if args.replace_system_role else args.input_jsonl
    output_jsonl_path = args.output_jsonl

    adapter_name: str | None = None
    if os.path.exists(args.model_name):  # Local file
        adapter_config_file: str = os.path.join(
            args.model_name, "adapter_config.json"
        )
        if os.path.exists(adapter_config_file):  # LoRA adapter
            adapter_name = args.model_name
            with open(adapter_config_file) as f:
                model_name_or_path = json.load(f)["base_model_name_or_path"]
        else:  # Full parameter
            model_name_or_path = args.model_name
    else:  # HuggingFace
        model_name_or_path = args.model_name

    llm = LLM(
        model=model_name_or_path,
        enable_lora=adapter_name is not None,
        tokenizer=args.model_name,
        dtype=args.dtype,
        max_model_len=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        quantization=args.quantization,
        max_lora_rank=args.max_lora_rank,
        load_format=args.load_format,
    )
    tokenizer = llm.get_tokenizer()
    if args.chat_template_truncate_pattern:
        tokenizer.chat_template = re.sub(args.chat_template_truncate_pattern, "", tokenizer.chat_template)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    _log_first_prompt = [True]
    def run_completion(batch, chat_index, logger):
        lora_request = LoRARequest("adapter", 1, adapter_name) if adapter_name else None

        if _log_first_prompt and not isinstance(llm.llm_engine, LLMEngineV1):
            input_ids = encode_prompt(llm, lora_request, batch[0][2][:chat_index])
            lines = ""
            for index, id in enumerate(input_ids):
                lines += f"{index:7} {json.dumps(tokenizer.decode(id), ensure_ascii=False)[1:-1]:32}{id:8}\n"
            logger.debug(f"""
======= probing prompt tokens begin =======
{lines}
======= probing prompt tokens end =======""")
            _log_first_prompt.clear()

        results = llm.chat(
            messages=[full_messages[:chat_index] for line_index, chat_index, full_messages in batch],
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=lora_request,
        )
        assert len(batch) == len(results), f"{len(batch)=} not equals to {len(results)}"
        is_first = True
        for (line_index, chat_index, full_messages), result in zip(batch, results):
            m = full_messages[chat_index]
            m["gold"] = m["content"]
            content_text = result.outputs[0].text
            m["content"] = content_text
            if is_first:
                is_first = False
                logger.debug(f"\n====== line #{line_index} - {chat_index} ======")
                logger.debug("\n" + full_messages[chat_index - 1]["content"])
                logger.debug("\n" + content_text)

    execute_completions(input_jsonl, output_jsonl_path, run_completion, batch_size=128)


def encode_prompt(llm, lora_request, messages):
    tokenizer = llm.get_tokenizer()
    model_config = llm.llm_engine.get_model_config()
    chat_template = None
    chat_template_content_format = "auto"
    add_generation_prompt = True
    continue_final_message = False
    tools = None

    try:
        resolved_content_format = resolve_chat_template_content_format(
            chat_template=chat_template,
            given_format=chat_template_content_format,
            tokenizer=tokenizer,
        )
    except:
        resolved_content_format = resolve_chat_template_content_format(
            chat_template=chat_template,
            tools=None,
            given_format=chat_template_content_format,
            tokenizer=tokenizer,
        )
    conversation, mm_data = parse_chat_messages(
        messages,
        model_config,
        tokenizer,
        content_format=resolved_content_format,
    )
    if isinstance(tokenizer, MistralTokenizer):
        prompt_data = apply_mistral_chat_template(
            tokenizer,
            messages=messages,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
        )
    else:
        prompt_data = apply_hf_chat_template(
            tokenizer,
            conversation=conversation,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
        )
    preprocessed_inputs = llm.llm_engine.input_preprocessor.preprocess(
        prompt_data,
        request_id="dump",
        lora_request=lora_request,
        prompt_adapter_request=None,
    )
    processed_inputs = llm.llm_engine.input_processor(preprocessed_inputs)
    return processed_inputs["prompt_token_ids"]

if __name__ == "__main__":
    main()
