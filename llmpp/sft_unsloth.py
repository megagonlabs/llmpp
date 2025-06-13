import torch
from unsloth import FastLanguageModel, FastModel
from trl import SFTConfig

from .sft_base import get_config, run_sft


def prepare_model(
    model_args: dict,
    tokenizer_args: dict,
    sft_config_args: dict,
    lora_args: dict,
    unsloth_args: dict = {},
    **kwargs,
):
    assert not tokenizer_args.get("pretrained_model_name_or_path"), "unsloth does not support external tokenizer"

    model_args["torch_dtype"] = getattr(torch, model_args["torch_dtype"])
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
            r = lora_args.get("r", 16),
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

    def formatting_func(examples):
        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in examples["messages"]]
        return {"text" : texts}

    # if "optim" not in sft_config_args:
    #     sft_config_args["optim"] = "adamw_8bit"
    sft_config = SFTConfig(**sft_config_args)

    return (
        model,
        tokenizer,
        formatting_func,
        {
            "tokenizer": tokenizer,
            "args": sft_config,
        }
    )


def main():
    run_sft(get_config(), prepare_model)


if __name__ == "__main__":
    main()
