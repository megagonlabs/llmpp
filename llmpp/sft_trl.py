import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig
from peft import LoraConfig

from .sft_base import get_config, run_sft


def prepare_model(
    model_args: dict,
    tokenizer_args: dict,
    sft_config_args: dict,
    lora_args: dict,
    save_merged: bool,
    **kwargs,
):
    assert not save_merged, "save_merged not supported"
    if not tokenizer_args.get("pretrained_model_name_or_path"):
        tokenizer_args["pretrained_model_name_or_path"] = model_args["pretrained_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)

    lora_config: LoraConfig | None = None
    if lora_args.pop("use_lora"):
        lora_config = LoraConfig(**lora_args)

    model_args["torch_dtype"] = getattr(torch, model_args["torch_dtype"])
    if "gemma-3-" in model_args["pretrained_model_name_or_path"]:
        model = transformers.Gemma3ForConditionalGeneration.from_pretrained(**model_args)
    else:
        if "use_cache" not in model_args and sft_config_args.get("gradient_checkpointing"):
            model_args["use_cache"] = False
        model = AutoModelForCausalLM.from_pretrained(**model_args)

    formatting_func = lambda _: _
    sft_config = SFTConfig(**sft_config_args)

    return (
        model,
        tokenizer,
        formatting_func,
        {
            "processing_class": tokenizer,
            "peft_config": lora_config,
            "args": sft_config,
        }
    )


def main():
    run_sft(get_config(), prepare_model)


if __name__ == "__main__":
    main()
