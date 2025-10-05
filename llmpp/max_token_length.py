import json
import sys

from transformers import AutoTokenizer

from .utils import create_system_role_replaced_tempfiles

LOCAL_TOP_N = 10


def main():
    argv = sys.argv[1:]
    if argv[0] == "--max":
        max_length = int(argv[1])
        argv = argv[2:]
    else:
        max_length = 0
    model_path = argv[0]
    jsonl_path_list = argv[1:]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    try:
        tokenizer.apply_chat_template([{"system": "test"}, {"user": "test"}, {"assitant": "test"}])
        replace_system_role = False
    except Exception:
        replace_system_role = True
    total_max = -1
    total_max_path = None
    total_char = 0
    total_token = 0
    for jsonl_path in jsonl_path_list:
        try:
            local_max = []
            local_total_char = 0
            local_total_token = 0
            if jsonl_path.endswith(".text"):
                input_path = jsonl_path
            elif replace_system_role:
                input_path = create_system_role_replaced_tempfiles(jsonl_path)
            else:
                input_path = jsonl_path
            with open(input_path, "r", encoding="utf8") as fin:
                for line_index, line in enumerate(fin, 1):
                    if input_path.endswith(".text"):
                        text = line.rstrip("\n")
                        char_length = len(text)
                        token_length = len(tokenizer.encode(text))
                    else:
                        prompt = json.loads(line)
                        messages = prompt["messages"]
                        char_length = len(tokenizer.apply_chat_template(messages, tokenize=False))
                        token_length = len(tokenizer.apply_chat_template(messages, tokenize=True))
                    local_total_char += char_length
                    local_total_token += token_length
                    if token_length <= max_length:
                        print(line, end="", file=sys.stderr)
                    elif max_length > 0:
                        print("removed line", line_index, token_length)
                    local_max = sorted(local_max + [token_length], reverse=True)[:LOCAL_TOP_N]
                    if total_max < token_length:
                        total_max = token_length
                        total_max_path = jsonl_path
            print(*local_max, "local char token:", local_total_char, local_total_token, jsonl_path, sep="\t")
            total_char += local_total_char
            total_token += local_total_token
        except Exception as e:
            print("#ERR#", jsonl_path, e, sep="\t")
    print("total:", total_max, total_max_path, sep="\t")
    print(f"total char token", total_char, total_token, sep="\t")


if __name__ == "__main__":
    main()
