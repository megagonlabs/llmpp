import json
import os
import shutil
from datetime import datetime

import torch
from unsloth import FastLanguageModel, FastModel
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from datasets import load_dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from .sft_base import get_config
from .utils import create_logger, create_system_role_replaced_tempfiles


def run_sft(
    logger,
    model_args: dict,
    tokenizer_args: dict,
    dataset_args: dict,
    sft_config_args: dict,
    lora_args: dict,
    sft_trainer_args: dict = {},
    unsloth_args: dict = {},
    **kwargs,
):
    assert not tokenizer_args.get("pretrained_model_name_or_path"), "unsloth does not support external tokenizer"

    is_rank0 = os.getenv("LOCAL_RANK", "0") == "0"

    model_args["dtype"] = getattr(torch, model_args["dtype"])
    model, tokenizer = FastModel.from_pretrained(
        model_name = model_args["pretrained_model_name_or_path"],
        max_seq_length = sft_config_args["max_seq_length"],
        load_in_4bit = model_args.get("load_in_4bit", False),
        load_in_8bit = model_args.get("load_in_8bit", False),
        full_finetuning = not lora_args.get("use_lora", False),
    )
    if lora_args.get("use_lora"):
        target_modules = lora_args.get("target_modules")
        if target_modules == "all-linear":
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        model = FastLanguageModel.get_peft_model(
            model,
            r = lora_args.get("r", 8),
            target_modules=target_modules,
            lora_alpha = lora_args.get("alpha", 8), # 16
            lora_dropout = lora_args.get("lora_dropout", 0), # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            use_gradient_checkpointing = True, # "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            max_seq_length = sft_config_args["max_seq_length"],
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
            **unsloth_args.get("peft_model_args", {}),
        )
    for k, v in tokenizer_args.items():
        if k not in ["pretrained_model_name_or_path", "padding_side"]:
            setattr(tokenizer, k, v)

    if unsloth_args.get("chat_template"):
        tokenizer = get_chat_template(tokenizer, unsloth_args["chat_template"])

    train_jsonl_path = dataset_args["train_jsonl_path"]
    dev_jsonl_path = dataset_args.get("dev_jsonl_path")
    if dataset_args.get("replace_system_role"):
        train_jsonl_path = create_system_role_replaced_tempfiles(train_jsonl_path)
        dev_jsonl_path = create_system_role_replaced_tempfiles(dev_jsonl_path) if dev_jsonl_path else None
    try:
        def formatting_prompts_func(examples):
            messages = examples["messages"]
            texts = [tokenizer.apply_chat_template(m, tokenize = False, add_generation_prompt = False) for m in messages]
            if unsloth_args.get("remove_prefix"):
                remove_prefix = unsloth_args["remove_prefix"]
                texts = [text.removeprefix(remove_prefix) for text in texts]
            return {"text" : texts}

        dataset = load_dataset("json", data_files=train_jsonl_path)
        train_dataset = standardize_data_formats(
            load_dataset("json", data_files=train_jsonl_path)["train"],
            tokenizer,
        ).map(formatting_prompts_func, batched=True)
        dev_dataset = standardize_data_formats(
            load_dataset("json", data_files=dev_jsonl_path)["train"],
            tokenizer,
        ).map(formatting_prompts_func, batched=True) if dev_jsonl_path else None

        sft_config = SFTConfig(
            dataset_text_field = "text",
            label_names = ["labels"],
            **sft_config_args,
        )

        class _TrainerCallback(TrainerCallback):
            def on_evaluate(self, args, state, control, **kwargs):
                logger.info(state.log_history[-1])

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            callbacks=[_TrainerCallback()],
            args=sft_config,
            **sft_trainer_args,
        )
        if unsloth_args.get("train_on_responses_only"):
            trainer = train_on_responses_only(
                trainer,
                instruction_part = unsloth_args["train_on_responses_only"]["instruction_part"],
                response_part = unsloth_args["train_on_responses_only"]["response_part"],
            )

        lines = "\n".join(name for name, _ in model.named_parameters())
        logger.debug(f"""
======= probing model layers begin =======
{lines}
======= probing model layers end =======""")
        if hasattr(trainer.model, "print_trainable_parameters"):
            trainer.model.print_trainable_parameters()

        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(sft_config_args["output_dir"])

    finally:
        if dataset_args["replace_system_role"]:
            os.unlink(train_jsonl_path)
            if dev_jsonl_path:
                os.unlink(dev_jsonl_path)


def main():
    config = get_config()
    is_rank0 = os.getenv("LOCAL_RANK", "0") == "0"
    output_dir = config["sft_config_args"]["output_dir"]
    if is_rank0:
        if os.path.exists(output_dir):
            shutil.move(output_dir, output_dir.rstrip("/") + "_" + datetime.fromtimestamp(os.path.getmtime(output_dir)).strftime("%Y%m%d-%H%M%S"))
        os.makedirs(output_dir)
    log_file_path = f"{output_dir}/sft.log"
    logger = create_logger(log_file_path=log_file_path, is_dummy=not is_rank0)
    logger.debug(f"\n{json.dumps(config, ensure_ascii=True, indent=1)}")
    run_sft(logger=logger, **config)


if __name__ == "__main__":
    main()
