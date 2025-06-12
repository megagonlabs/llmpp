import os
import time
from pathlib import Path

from openai import OpenAI, BadRequestError

from .completion_base import execute_completions, parse_args

RETRY_MAX = 120
RETRY_INTERVAL = 30
RUN_INTERVAL = 0.25


def main():
    def default_output_dir_func(model_name, input_jsonl):
        if model_name.startswith("ft:"):
            model_name = model_name[3:]
        elif model_name.startswith("gpt-"):
            model_name = model_name[4:]
        b = model_name.split(":")
        return f"models/{b[0]}-{b[2]}/{input_jsonl}"
    args = parse_args(default_output_dir_func=default_output_dir_func)
    assert not args.replace_system_role, "--replace_system_role not supported"
    assert not args.chat_template_truncate_pattern, "--chat_template_truncate_pattern not supported"

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    output_jsonl_path = args.output_jsonl

    _log_first_prompt = [True]
    def run_completion(batch, chat_index, logger):
        if _log_first_prompt:
            logger.info(f"Ignoring below settings:\n{args.dtype=}")
            _log_first_prompt.clear()
        for line_index, chat_index, full_messages in batch:
            logger.debug(f"\n====== line #{line_index} - {chat_index} ======")
            messages = [{"role": _["role"], "content": _["content"]} for _ in full_messages[:chat_index]]
            instruction = messages[-1]["content"]
            max_completion_tokens = args.max_tokens
            retry = 0
            while True:
                try:
                    logger.debug("\n" + instruction)
                    completion = client.chat.completions.create(
                        model=args.model_name,
                        messages=messages,
                        temperature=args.temperature,
                        max_completion_tokens=max_completion_tokens,
                    )
                    content_text = completion.choices[0].message.content
                    logger.debug("\n" + content_text)
                    m = full_messages[chat_index]
                    m["gold"] = m["content"]
                    m["content"] = content_text
                    break
                except BadRequestError as e:
                    raise e
                except Exception as e:
                    retry += 1
                    logger.info(f"Retry #{retry}: {e}")
                    time.sleep(RETRY_INTERVAL)
            time.sleep(RUN_INTERVAL)

    execute_completions(args.input_jsonl, output_jsonl_path, run_completion, batch_size=1)


if __name__ == "__main__":
    main()
