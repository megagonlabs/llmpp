import json
import sys
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
from typing import IO

from .utils import select_tsv_part


REPORTING_FIELDS = {
    "sentence": ["gold", "content", "aligned", "correct_form"],
    "token": ["gold", "content", "aligned", "correct_form", "correct_pos1", "correct_pos2"],
}


def parse_args() -> Namespace:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("completion_results_jsonl_files", nargs="+")
    parser.add_argument("--stop_on_error", "--e", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    for completion_results_jsonl in args.completion_results_jsonl_files:
        try:
            base_path = Path(completion_results_jsonl)
            output_eval_json_path = f"{base_path.parent}/{base_path.stem}.tokenize.json"
            output_report_path = f"{base_path.parent}/{base_path.stem}.tokenize.report"
            with open(completion_results_jsonl, "r", encoding="utf8") as fin:
                completion_results = [json.loads(_)["messages"] for _ in fin]
            with open(output_report_path, "w", encoding="utf8") as f_report:
                stats = eval(completion_results, f_report, stop_on_error=args.stop_on_error)
            with open(output_eval_json_path, "w", encoding="utf8") as f_eval:
                json.dump(stats, f_eval, ensure_ascii=False, indent=1)
                print(file=f_eval)
            if stats["sentence"]["gold"] > 0:
                print(completion_results_jsonl, stats["digest"], file=sys.stderr)
                print(completion_results_jsonl, *[stats[u][k] for u in REPORTING_FIELDS for k in REPORTING_FIELDS[u]], sep="\t")
            else:
                print("no content", completion_results_jsonl, file=sys.stderr)
        except Exception as e:
            print("skipping", completion_results_jsonl, file=sys.stderr)
            if args.stop_on_error:
                raise e
            print(e, file=sys.stderr)
        print(file=sys.stderr)


def eval(
    completion_results: list[dict],
    f_report: IO,
    stop_on_error: bool = False,
) -> dict:
    gold_sentence = 0
    gold_token = 0
    content_token = 0
    aligned_sentence = 0
    aligned_token = 0
    correct_pos1_sentence = 0
    correct_pos1_token = 0
    correct_pos2_sentence = 0
    correct_pos2_token = 0
    recovered_index_sentence = 0
    recovered_index_token = 0
    confusion_pos1 = defaultdict(lambda: defaultdict(int))
    confusion_pos2 = defaultdict(lambda: defaultdict(int))

    for line_index, messages in enumerate(completion_results, 1):
        if "tokenization" not in messages[-2]["content"]:
            continue
        result = messages[-1]
        assert "gold" in result, f"Inference result not saved in line #{line_index}"
        gold_list = parse_records(result["gold"], stop_on_error)
        content_list = parse_records(result["content"], stop_on_error)
        for gold, content in zip(gold_list, content_list):
            gold_sentence += 1
            gold_forms = " ".join(_["form"] for _ in gold)
            content_forms = " ".join(_["form"] for _ in content)
            if gold_forms == content_forms:
                if f_report:
                    print("=", gold_forms, file=f_report)
            else:
                print("<", gold_forms, file=f_report)
                print(">", content_forms, file=f_report)
                content = _recover_content(gold, content)
                recovered_forms = " ".join(_["form"] for _ in content)
                if recovered_forms != content_forms:
                    print(":", recovered_forms, file=f_report)
            gold_token += len(gold)
            content_token += len(content)

            index_recoveries = []
            for index, c in enumerate(content, 1):
                if c["index"] != index:
                    index_recoveries.append(f'index recovery: {c["index"]} -> {index}')
                    c["index"] = index
                    recovered_index_token += 1
            if index_recoveries:
                recovered_index_sentence += 1
                if f_report:
                    print(*index_recoveries, sep="\n", file=f_report)

            align_errors = []
            offset_g = offset_c = 0
            index_g = index_c = 0
            correct_pos1 = correct_pos2 = 0
            while index_g < len(gold) and index_c < len(content):
                g = gold[index_g]
                c = content[index_c]
                end_g = offset_g + len(g["form"])
                end_c = offset_c + len(c["form"])
                if offset_g == offset_c and end_g == end_c:
                    aligned_token += 1
                    if g["pos1"]:
                        if g["pos1"] == c["pos1"]:
                            correct_pos1 += 1
                        else:
                            confusion_pos1[g["pos1"]][c["pos1"]] += 1
                    if g["pos2"]:
                        if g["pos2"] == c["pos2"]:
                            correct_pos2 += 1
                        else:
                            confusion_pos2[g["pos2"]][c["pos2"]] += 1
                else:
                    align_errors.append(
                        f'align error: {c["index"]}, {json.dumps(c["form"], ensure_ascii=False)} -> {json.dumps(g["form"], ensure_ascii=False)}')
                    if g["pos1"]:
                        confusion_pos1[g["pos1"]][None] += 1
                    if g["pos2"]:
                        confusion_pos2[g["pos2"]][None] += 1
                if end_g == end_c:
                    index_g += 1
                    index_c += 1
                    offset_g = end_g
                    offset_c = end_c
                elif end_g < end_c:
                    index_g += 1
                    offset_g = end_g
                else:
                    index_c += 1
                    offset_c = end_c
            if not align_errors:
                aligned_sentence += 1
            correct_pos1_token += correct_pos1
            if correct_pos1 == len(gold):
                correct_pos1_sentence += 1
            correct_pos2_token += correct_pos2
            if correct_pos2 == len(gold):
                correct_pos2_sentence += 1

    def f1(m: int, g: int, c: int) -> float:
        return 2. / (g / m + c / m) if m > 0 else 0.
    aligned_s = form_s = f1(aligned_sentence, gold_sentence, gold_sentence)
    aligned_t = form_t = f1(aligned_token, gold_token, content_token)
    pos1 = f1(correct_pos1_token, gold_token, content_token)
    pos2 = f1(correct_pos2_token, gold_token, content_token)
    confusion1 = {g: {c: v for c, v in sorted(d.items(), key=lambda _: (-_[1], _[0] or ""))} for g, d in confusion_pos1.items()}
    confusion2 = {g: {c: v for c, v in sorted(d.items(), key=lambda _: (-_[1], _[0] or ""))} for g, d in confusion_pos2.items()}
    return {
        "digest": f"{aligned_s=:.4f}, {aligned_t=:.4f}, {form_s=:.4f}, {form_t=:.4f}, {pos1=:.4f}, {pos2=:.4f}",
        "sentence": {
            "gold": gold_sentence,
            "content": gold_sentence,
            "aligned": aligned_sentence,
            "correct_form": aligned_sentence,
            "correct_pos1": correct_pos1_sentence,
            "correct_pos2": correct_pos2_sentence,
            "recoverd_index": recovered_index_sentence,
        },
        "token": {
            "gold": gold_token,
            "content": content_token,
            "aligned": aligned_token,
            "correct_form": aligned_token,
            "correct_pos1": correct_pos1_token,
            "correct_pos2": correct_pos2_token,
            "recoverd_index": recovered_index_token,
        },
        "confusion_pos1": confusion1,
        "confusion_pos2": confusion2,
    }


def _recover_content(gold, content):
    gold_text = "".join(_["form"] for _ in gold)
    for forward_c, c in enumerate(content):
        if gold_text.startswith(c["form"]):
            gold_text = gold_text[len(c["form"]):]
            if gold_text:
                continue
            else:
                return content[:forward_c + 1]
        backward_c = len(content) - 1
        for backward_c in range(len(content) - 1, forward_c, -1):
            c = content[backward_c]
            if gold_text.endswith(c["form"]):
                gold_text = gold_text[:-len(c["form"])]
                if gold_text:
                    continue
                else:
                    return content[:forward_c] + content[backward_c:]
            else:
                break
        else:
            backward_c = forward_c
        error_span_token = {
            "index": forward_c,
            "form": gold_text,
            "pos1": content[forward_c]["pos1"],
            "pos2": content[forward_c]["pos2"],
        }
        return content[:forward_c] + [error_span_token] + content[backward_c + 1:]
    return content


def parse_records(content: str, stop_on_error: bool = False, dummy_form: str = ".") -> list[list[dict]]:
    rows_list = [_ for _ in select_tsv_part(content, tsv_index=None, min_columns=2, end_regex=r"(?m)^(dependency parsing:)$") if _ and _[0][1]]
    if not rows_list:
        return []
    field_num = len(rows_list[0][0])
    records_list = []
    for rows in rows_list:
        records = []
        for r in rows:
            try:
                if len(r) < field_num:
                    r = r + ["_"] * (field_num - len(r))
                if field_num == 2:
                    records.append({"index": int(r[0]), "form": r[1] or dummy_form, "pos1": "", "pos2": ""})
                elif field_num == 3:
                    records.append({"index": int(r[0]), "form": r[1] or dummy_form, "pos1": r[2], "pos2": ""})
                elif field_num == 4:
                    records.append({"index": int(r[0]), "form": r[1] or dummy_form, "pos1": r[2], "pos2": r[3]})
            except Exception as e:
                if stop_on_error:
                    raise e
                break
        if records:  # eliminate tail whitespaces of last token
            records[-1]["form"] = records[-1]["form"].rstrip(" ")
            records_list.append(records)
    return records_list


def reporting_fields(r: dict) -> list:
    fields = [r["index"], r["form"]]
    if "upos" in r:
        fields.append(r["upos"])
    if r["head"]:
        fields += [r["head"]["index"], r["head"]["form"], r["deprel"]]
    else:
        fields += [-1, "ROOT", "root"]
    return fields


if __name__ == "__main__":
    main()
