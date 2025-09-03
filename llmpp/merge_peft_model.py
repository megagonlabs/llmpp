import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel


def main():
    peft_model = sys.argv[1]
    peft_config = PeftConfig.from_pretrained(peft_model)
    base_model = peft_config.base_model_name_or_path
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = peft_model.rstrip("/") + ".merged"
    print(f"{base_model=}\n{peft_model=}\nmerge started", file=sys.stderr)
    save_merged_weights(
        base_model_path=base_model,
        peft_model_path=peft_model,
        output_dir=output_dir,
    )
    print(f"merge completed and saved in {output_dir}", file=sys.stderr)


def save_merged_weights(
    base_model_path,
    peft_model_path,
    output_dir,
):
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, peft_model_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir, max_shard_size="2GB", safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
