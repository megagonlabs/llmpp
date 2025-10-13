import json
import os
import shutil
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from .utils import create_logger


def parse_args(parser: ArgumentParser = None, default_output_dir_func = lambda model_name, input_jsonl: f"{model_name}/{input_jsonl}"):
    if not parser:
        parser = ArgumentParser()
    parser.add_argument("--model_name", "--m", type=str)
    parser.add_argument("--input_jsonl", "--i", type=str)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--output_dir", "--d", type=str, default=None)
    group.add_argument("--output_jsonl", "--o", type=str, default=None)
    parser.add_argument("--temperature", "--t", type=float, default=0.)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_tokens", default=8192, type=int)
    parser.add_argument("--replace_system_role", "--rsr", action="store_true")
    parser.add_argument("--chat_template_truncate_pattern", "--ct")
    args = parser.parse_args()
    if not args.output_jsonl:
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = default_output_dir_func(args.model_name, args.input_jsonl)
        if os.path.exists(output_dir):
            shutil.move(output_dir, output_dir.rstrip("/") + "_" + datetime.fromtimestamp(os.path.getmtime(output_dir)).strftime("%Y%m%d-%H%M%S"))
        os.makedirs(output_dir)
        args.output_jsonl = f"{output_dir}/completion.jsonl"
    return args


def execute_completions(
        input_jsonl_path,
        output_jsonl_path,
        completion_func,
        batch_size=1,
        logger=None,
):
    if not logger:
        log_file_path = f"{output_jsonl_path}.log"
        logger = create_logger(log_file_path=log_file_path)
    logger.debug(f" input file: {input_jsonl_path}")
    with open(input_jsonl_path, "r", encoding="utf8") as fin:
        records = [json.loads(_) for _ in fin]
    logger.debug(f"output file: {output_jsonl_path}")
    with open(output_jsonl_path, "w", encoding="utf8"):
        pass  # clear the output file for appending results

    start = time.perf_counter()
    batch = []
    try:
        while True:
            completed_all = True
            prev = 0
            for line_index, record in enumerate(records, 1):
                messages = record["messages"]
                for chat_index, m in enumerate(messages):
                    if m["role"] == "assistant" and "gold" not in m and "error" not in m:
                        batch.append([line_index, chat_index, messages])
                        completed_all = False
                        if prev + 1 < line_index:
                            logger.debug(f"skipping lines #{prev + 1} to #{line_index - 1}")
                        prev = line_index
                        break
                if len(batch) == batch_size or (batch and line_index == len(records)):
                    completion_func(batch, chat_index, logger=logger)
                    batch.clear()
            assert not batch
            if completed_all:
                break
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        logger.debug(f"inference_runtime: {time.perf_counter() - start:.03f}")
        logger.debug(f"saving: {output_jsonl_path}")
        with open(output_jsonl_path, "w", encoding="utf8") as fout:
            for record in records:
                json.dump(record, fout, ensure_ascii=False)
                print(file=fout)
