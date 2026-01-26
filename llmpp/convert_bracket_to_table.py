import json
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

from .utils import constituent_tree, recover_bracketing, recover_word, select_last_bracketing_line, yield_constituent_units


def add_args(parser: ArgumentParser = None) -> ArgumentParser:
    parser = parser or ArgumentParser()
    parser.add_argument("completion_results_jsonl_files", nargs="+")
    parser.add_argument("--no-terminal", "--nt", action="store_true")
    parser.add_argument("--apply_recovery", "--r", action="store_true")
    parser.add_argument("--stop_on_error", "--e", action="store_true")
    return parser


def blacket_terminal_mapping(blacketed_text: str) -> str:
    result = ""
    prev_end = 0
    for m in re.finditer(r"\([^ ()]+ ([()]) ?\)", blacketed_text):
        result += blacketed_text[prev_end:m.start(1)] + {"(": "-LRB-", ")": "-RRB-"}[m.group(1)]
        prev_end = m.end(1)
    result += blacketed_text[prev_end:]
    return result    


def flatten_tree(tree: list[str|list]) -> list:
    def _traverse_tree(tree: list[str|list], records: list[list]) -> list:
        head = None
        children = []
        label = tree[0]
        for subtree in tree[1:]:
            if len(subtree) == 1:
                assert isinstance(subtree[0], str), f"bad subtree: {subtree}"
                pos = subtree[0]
                head = [None, "", pos, None, label, children]
                records.append(head)
            elif len(subtree) == 2 and isinstance(subtree[1], str):
                pos, word = subtree
                head = [None, word, pos, None, label, children]
                records.append(head)
            else:
                child = _traverse_tree(subtree, records)
                children.append(child)
        assert head, f"no head: {tree}"
        return head

    records = []
    root = _traverse_tree(tree, records)
    root[3] = "0"
    for index, r in enumerate(records, 1):
        r[0] = str(index)
        for child in r[-1]:
            child[3] = str(index)
        del r[-1]
    assert all(_ is not None for r in records for _ in r), f"incompleted: {records}"
    return records


def bracket_to_table(
        bracket_jsonl_path: str,
        table_jsonl_path: str = None,
        no_terminal: bool = False,
        apply_recovery: bool = False,
        stop_on_error: bool = False,
):
    if not table_jsonl_path:
        p = Path(bracket_jsonl_path)
        table_jsonl_path = f"{p.parent / p.stem}.table.jsonl"
    with open(bracket_jsonl_path, "r", encoding="utf8") as fin, open(table_jsonl_path, "w", encoding="utf8") as fout:
        for line in fin.readlines():
            prompt = json.loads(line)
            result = prompt["messages"][-1]
            pred_text = blacket_terminal_mapping(
                select_last_bracketing_line(
                    result["content"].strip()
                )
            ).replace(" deleteById", "")
            gold_text = blacket_terminal_mapping(
                select_last_bracketing_line(
                    result["gold"].strip()
                )
            ) if "gold" in result else None
            if apply_recovery:
                pred_text = re.sub(r"\) +", ")", pred_text)
                pred_text = recover_bracketing(pred_text)
                if gold_text:
                    gold_text = re.sub(r"\) +", ")", gold_text)
                    pred_text = recover_word(gold_text.strip(), pred_text)
            try:
                pred_table = flatten_tree(constituent_tree(list(yield_constituent_units(pred_text)), no_terminal)[1])
            except Exception as e:
                print(result, file=sys.stderr)
                if stop_on_error:
                    raise e
                pred_table = []
            try:
                gold_table = flatten_tree(constituent_tree(list(yield_constituent_units(gold_text)), no_terminal)[1]) if gold_text else None
            except Exception as e:
                if stop_on_error:
                    print(result, file=sys.stderr)
                    print(gold_text, file=sys.stderr)
                raise e
            prompt["messages"].append(
                {
                    "role": "assistant",
                    "content": "\n".join("\t".join(r) for r in pred_table) + "\n",
                    "gold": "\n".join("\t".join(r) for r in gold_table) + "\n",
                }
            )
            json.dump(prompt, fout, ensure_ascii=False)
            print(file=fout)
    return table_jsonl_path


def main():
    args = add_args().parse_args()
    for prompt_jsonl_path in args.completion_results_jsonl_files:
        table_jsonl_path = bracket_to_table(
            prompt_jsonl_path,
            no_terminal=args.no_terminal,
            apply_recovery=args.apply_recovery,
            stop_on_error=args.stop_on_error,
        )
        print(f"bracketed: {prompt_jsonl_path}\n => table: {table_jsonl_path}", file=sys.stderr)


if __name__ == '__main__':
    main()