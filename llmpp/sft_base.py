import json
import os
import re
import shutil
import yaml
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from transformers import TrainerCallback
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from .utils import create_logger, create_system_role_replaced_tempfiles


def get_config(parser: ArgumentParser = None):
    if not parser:
        parser = ArgumentParser()
    parser.add_argument("--config", "--c", required=True)
    parser.add_argument("--pretrained_model_name_or_path", "--m")
    parser.add_argument("--tokenizer")
    parser.add_argument("--train_jsonl_path", "--t")
    parser.add_argument("--dev_jsonl_path", "--d")
    parser.add_argument("--output_dir", "--o")
    parser.add_argument("--per_device_train_batch_size", "--b", type=int)
    parser.add_argument("--gradient_accumulation_steps", "--ga", type=int)
    parser.add_argument("--max_seq_length", "--l", type=int)
    parser.add_argument("--num_train_epochs", "--e", type=int)
    parser.add_argument("--learning_rate", "--lr")
    parser.add_argument("--load_in_4bit", "--li4", action="store_true")
    parser.add_argument("--load_in_8bit", "--li8", action="store_true")
    parser.add_argument("--lora_r", "--r", type=int)
    parser.add_argument("--lora_alpha", "--a", type=int)
    parser.add_argument("--lora_all_linear_with_lm_head", "--lmh", action="store_true")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf8") as fin:
        config = yaml.safe_load(fin)

    if args.pretrained_model_name_or_path:
        config["model_args"]["pretrained_model_name_or_path"] = args.pretrained_model_name_or_path
    if args.tokenizer:
        config["tokenizer_args"]["pretrained_model_name_or_path"] = args.tokenizer
    if args.train_jsonl_path:
        config["dataset_args"]["train_jsonl_path"] = args.train_jsonl_path
    if args.dev_jsonl_path:
        config["dataset_args"]["dev_jsonl_path"] = args.dev_jsonl_path
    if args.output_dir:
        config["sft_config_args"]["output_dir"] = args.output_dir
    if args.per_device_train_batch_size:
        config["sft_config_args"]["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.gradient_accumulation_steps:
        config["sft_config_args"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.max_seq_length:
        config["sft_config_args"]["max_seq_length"] = args.max_seq_length
    if args.num_train_epochs:
        config["sft_config_args"]["num_train_epochs"] = args.num_train_epochs
    if args.learning_rate:
        config["sft_config_args"]["learning_rate"] = float(args.learning_rate)
    if args.load_in_4bit:
        config["model_args"]["load_in_4bit"] = True
    if args.load_in_8bit:
        config["model_args"]["load_in_8bit"] = True
    if args.lora_r:
        config["lora_args"]["r"] = args.lora_r
        if "lora_alpha" not in config["lora_args"]:
            config["lora_args"]["lora_alpha"] = args.lora_r
    if args.lora_all_linear_with_lm_head:
        config["lora_args"]["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]

    if not config["sft_config_args"]["output_dir"]:
        model_name = config["model_args"]["pretrained_model_name_or_path"].rstrip("/").split("/")[-1]
        if args.lora_r:
            model_name += f"-r{args.lora_r}"
        if args.lora_alpha:
            model_name += f"-a{args.lora_alpha}"
        if args.lora_all_linear_with_lm_head:
            model_name += "-lmh"
        if args.learning_rate:
            model_name += f"-lr{args.learning_rate}"
        if args.num_train_epochs:
            model_name += f"-epoch{args.num_train_epochs}"
        train_jsonl_path = Path(config["dataset_args"]["train_jsonl_path"])
        dataset_name = f"{train_jsonl_path.parent.name}_{train_jsonl_path.stem}"
        config["sft_config_args"]["output_dir"] = f"models/{model_name}_{dataset_name}"

    return config


def run_sft(config, prepare_model_func):
    is_rank0 = os.getenv("LOCAL_RANK", "0") == "0"
    output_dir = config["sft_config_args"]["output_dir"]
    if is_rank0:
        if os.path.exists(output_dir) and (
            os.path.exists(f"{output_dir}/config.json") or
            os.path.exists(f"{output_dir}/adapter_config.json")
        ):
            shutil.move(output_dir, output_dir.rstrip("/") + "_" + datetime.fromtimestamp(os.path.getmtime(output_dir)).strftime("%Y%m%d-%H%M%S"))
        os.makedirs(output_dir, exist_ok=True)
    log_file_path = f"{output_dir}/sft.log"
    logger = create_logger(log_file_path=log_file_path, is_dummy=not is_rank0)
    logger.debug(f"\n{json.dumps(config, ensure_ascii=True, indent=1)}")

    model, tokenizer, formatting_func, sft_trainer_args = prepare_model_func(**config)

    if config.get("chat_template_replace_pattern"):
        chat_template = re.sub(
            config["chat_template_replace_pattern"]["pattern"],
            config["chat_template_replace_pattern"]["repl"],
            tokenizer.chat_template,
        )
        if tokenizer.chat_template == chat_template:
            logger.warning(f"""chat_template_replace_pattern has no effect on this tokenizer.
{tokenizer.chat_template=}
{config["chat_template_replace_pattern"]=}
""")
        else:
            tokenizer.chat_template = chat_template

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        **config["collator_args"],
    )

    dataset_args = config["dataset_args"]
    train_jsonl_path = dataset_args["train_jsonl_path"]
    dev_jsonl_path = dataset_args.get("dev_jsonl_path")
    if dataset_args["replace_system_role"]:
        train_jsonl_path = create_system_role_replaced_tempfiles(train_jsonl_path)
        dev_jsonl_path = create_system_role_replaced_tempfiles(dev_jsonl_path) if dev_jsonl_path else None
    try:
        train_dataset = load_dataset("json", data_files=train_jsonl_path)["train"].map(formatting_func, batched=True)
        dev_dataset = load_dataset("json", data_files=dev_jsonl_path)["train"].map(formatting_func, batched=True) if dev_jsonl_path else None

        class _TrainerCallback(TrainerCallback):
            def on_evaluate(self, args, state, control, **kwargs):
                logger.info(state.log_history[-1])

        trainer = SFTTrainer(
            model,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=collator,
            callbacks=[_TrainerCallback()],
            **sft_trainer_args,
        )

        lines = "\n".join(name for name, _ in model.named_parameters())
        logger.debug(f"""
======= probing model layers begin =======
{lines}
======= probing model layers end =======""")
        if hasattr(trainer.model, "print_trainable_parameters"):
            trainer.model.print_trainable_parameters()

        probe = next(iter(trainer.get_train_dataloader()))
        input_ids = probe["input_ids"][0].cpu()
        attention_mask = probe["attention_mask"][0].cpu()
        labels = probe["labels"][0].cpu()
        lines = ""
        prev_mask = 1
        for index, (id, mask, label) in enumerate(zip(input_ids, attention_mask, labels)):
            if mask == 0 and prev_mask == 0 and index + 1 != len(input_ids):
                continue
            lines += f"{index:7} {json.dumps(tokenizer.decode(id), ensure_ascii=False)[1:-1]:32}{id:8}{mask:8}{label:8}\n"
            prev_mask = mask
        logger.debug(f"""
======= probing train data begin =======
{lines}
======= probing train data end =======""")

        trainer.train()
        if is_rank0:
            trainer.save_model()
            tokenizer.save_pretrained(config["sft_config_args"]["output_dir"])
    finally:
        if dataset_args["replace_system_role"]:
            os.unlink(train_jsonl_path)
            if dev_jsonl_path:
                os.unlink(dev_jsonl_path)
