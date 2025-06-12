import json
import sys

from transformers import AutoTokenizer

from .utils import create_system_role_replaced_tempfiles


def main():
    model_path = sys.argv[1]
    jsonl_path_list = sys.argv[2:]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    try:
        tokenizer.apply_chat_template([{"system": "test"}, {"user": "test"}, {"assitant": "test"}])
        replace_system_role = False
    except Exception:
        replace_system_role = True
    total_max = -1
    total_max_path = None
    for jsonl_path in jsonl_path_list:
        try:
            local_max = -1
            if replace_system_role:
                input_path = create_system_role_replaced_tempfiles(jsonl_path)
            else:
                input_path = jsonl_path
            with open(input_path, "r", encoding="utf8") as fin:
                for _ in fin:
                    prompt = json.loads(_)
                    messages = prompt["messages"]
                    length = len(tokenizer.apply_chat_template(messages, tokenize=True))
                    if local_max < length:
                        local_max = length
                    if total_max < length:
                        total_max = length
                        total_max_path = jsonl_path
            print(local_max, jsonl_path, sep="\t")
        except Exception as e:
            print("#ERR#", jsonl_path, e, sep="\t")
    print(total_max, f"total: {total_max_path}", sep="\t")


if __name__ == "__main__":
    main()
