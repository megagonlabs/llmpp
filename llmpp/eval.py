import json
import re
import sys
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
from typing import IO

from .convert_bracket_to_table import bracket_to_table
from .utils import select_last_bracketing_line, select_tsv_part


def parse_args() -> Namespace:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("completion_results_jsonl_files", nargs="+")
    parser.add_argument("--format", "--f", type=str, choices=["auto", "tsv", "bracketing"], default="auto")
    parser.add_argument("--no_terminal_bracketing", "--nt", action="store_true")
    parser.add_argument("--apply_bracketing_recovery", "--r", action="store_true")
    parser.add_argument("--index_field", "--i", default=0, type=int)
    parser.add_argument("--use_deprel_subtypes", "--s", action="store_true")
    parser.add_argument("--stop_on_error", "--e", action="store_true")
    args = parser.parse_args()
    return args

REPORTING_FIELDS = {
    "sentence": ["gold", "content", "aligned", "correct_form"],
    "token": ["gold", "content", "aligned", "correct_form", "correct_upos", "correct_head", "correct_head_deprel"],
}

USER_PROMPT_SENTENCE_PATTERN = re.compile(r"\ninput sentence:\n(.+)\n")
USER_PROMPT_WORDS_PATTERN = re.compile(r"(?s)\n(?:indexes and )?words:\n(.+)")


def main():
    args = parse_args()
    for completion_results_jsonl in args.completion_results_jsonl_files:
        try:
            base_path = Path(completion_results_jsonl)
            output_eval_json_path = f"{base_path.parent}/{base_path.stem}.eval.json"
            output_eval_ignore_punct_json_path = f"{base_path.parent}/{base_path.stem}.eval.ignore-punct.json"
            output_report_path = f"{base_path.parent}/{base_path.stem}.eval.report"
            output_conllu_path = f"{base_path.parent}/{base_path.stem}.eval.conllu"
            config = {
                "src": completion_results_jsonl,
                "use_deprel_subtypes": args.use_deprel_subtypes,
                "index_field": args.index_field,
            }
            with open(completion_results_jsonl, "r", encoding="utf8") as fin:
                completion_results = [json.loads(_)["messages"] for _ in fin]
            if args.format == "bracketing" or args.format == "auto" and select_last_bracketing_line(completion_results[0][-1].get("gold") or completion_results[0][-1]["content"]):
                table_jsonl_path = bracket_to_table(
                    completion_results_jsonl,
                    no_terminal=args.no_terminal_bracketing,
                    apply_recovery=args.apply_bracketing_recovery,
                    stop_on_error=args.stop_on_error,
                )
                with open(table_jsonl_path, "r", encoding="utf8") as fin:
                    completion_results = [json.loads(_)["messages"] for _ in fin]

            with open(output_report_path, "w", encoding="utf8") as f_report, open(output_conllu_path, "w", encoding="utf8") as f_conllu:
                stats = eval(completion_results, args.index_field, args.use_deprel_subtypes, f_report, f_conllu, ignore_punct=False, stop_on_error=args.stop_on_error)
            stats["config"] = config
            stats["config"]["ignore_punct"] = False
            with open(output_eval_json_path, "w", encoding="utf8") as f_eval:
                json.dump(stats, f_eval, ensure_ascii=False, indent=1)
                print(file=f_eval)
            if stats["sentence"]["gold"] > 0:
                print(completion_results_jsonl, stats["digest"], file=sys.stderr)
                print(completion_results_jsonl, *[stats[u][k] for u in REPORTING_FIELDS for k in REPORTING_FIELDS[u]], sep="\t")
            else:
                print("no content", completion_results_jsonl, file=sys.stderr)

            stats = eval(completion_results, args.index_field, args.use_deprel_subtypes, None, None, ignore_punct=True)
            stats["config"] = config
            stats["config"]["ignore_punct"] = True
            with open(output_eval_ignore_punct_json_path, "w", encoding="utf8") as f_eval:
                json.dump(stats, f_eval, ensure_ascii=False, indent=1)
                print(file=f_eval)
        except Exception as e:
            print("skipping", completion_results_jsonl, file=sys.stderr)
            if args.stop_on_error:
                raise e
            print(e, file=sys.stderr)
        print(file=sys.stderr)


def is_punctuation(pos):
    punct_set = '.' '``' "''" ':' ','  # for PTB
    pos_set = ['pu', 'punct']  # for CTB & UD
    return (pos in punct_set) or (pos.lower() in pos_set)


def eval(
    completion_results: list[dict],
    index_field: int,
    use_deprel_subtypes: bool,
    f_report: IO,
    f_conllu: IO,
    ignore_punct: bool = False,
    stop_on_error: bool = False
) -> dict:
    gold_sentence = 0
    gold_token = 0
    content_sentence = 0
    content_token = 0
    aligned_sentence = 0
    aligned_token = 0
    correct_form_sentence = 0
    correct_form_token = 0
    correct_upos_sentence = 0
    correct_upos_token = 0
    correct_head_sentence = 0
    correct_head_token = 0
    correct_head_deprel_sentence = 0
    correct_head_deprel_token = 0
    consistent_sentence = 0
    consistent_token = 0
    single_root_sentence = 0
    single_root_token = 0
    no_loop_sentence = 0
    no_loop_token = 0
    recovered_index_sentence = 0
    recovered_index_token = 0
    recovered_form_sentence = 0
    recovered_form_token = 0
    confusion_upos = defaultdict(lambda: defaultdict(int))
    confusion_deprel = defaultdict(lambda: defaultdict(int))

    for line_index, messages in enumerate(completion_results, 1):
        user_prompt = messages[-2]["content"]
        if "dependency parsing" not in user_prompt:
            continue
        input_text, input_tokens = parse_user_prompt(user_prompt)
        result = messages[-1]
        gold_sentence += 1
        assert "gold" in result, f"Inference result not saved in line #{line_index}"
        gold = parse_records(result["gold"], index_field, stop_on_error)
        gold_text = input_text or "".join(_.get("form", "") for _ in gold)

        content = parse_records(result["content"], index_field, stop_on_error)
        if len(gold) == len(content):
            content_sentence += 1
        else:
            content = []
        gold_token += sum(1 for _ in gold if not ignore_punct or not is_punctuation(_["upos"]))
        content_token += sum(1 for _ in content if not ignore_punct or not is_punctuation(_["upos"]))

        index_recoveries = []
        for index, c in enumerate(content, 1):
            if c["index"] != index:
                index_recoveries.append(f'index recovery: {c["index"]} -> {index}')
                c["index"] = index
                recovered_index_token += 1
        if index_recoveries:
            recovered_index_sentence += 1

        form_recoveries = []
        for g, c in zip(gold, content):
            if c.get("form") != g.get("form"):
                form_recoveries.append(f'form recovery: {c["index"]}, {json.dumps(c["form"], ensure_ascii=False)} -> {json.dumps(g["form"], ensure_ascii=False)}')
                c["form"] = g["form"]
                recovered_form_token += 1
        if form_recoveries:
            recovered_form_sentence += 1

        if len(index_recoveries) == 0 and len(form_recoveries) == 0 and len(gold) == len(content):
            aligned_sentence += 1
            aligned_token += sum(1 for _ in gold if not ignore_punct or not is_punctuation(_["upos"]))

        content_text = "".join(_.get("form", "") for _ in content)
        if f_report:
            if gold_text == content_text:
                print("=", gold_text, file=f_report)
            else:
                print("<", " ".join(_.get("form", "") for _ in gold), file=f_report)
                print(">", " ".join(_.get("form", "") for _ in content), file=f_report)
            print(*index_recoveries, *form_recoveries, sep="\n", file=f_report)

        # evaluate dependency by checking offset + form 
        gold_offsets = {}
        content_offsets = {}
        for records, offsets in zip([gold, content], [gold_offsets, content_offsets]):
            offset = 0
            for r in records:
                r["index"] = offset
                offsets[offset] = r
                offset += len(r.get("form", ""))
            for r in records:
                if r["head"] == 0:
                    r["head"] = None
                else:
                    if int(r["head"]) - 1 < len(records):
                        r["head"] = records[int(r["head"]) - 1]
                    else:
                        r["head"] = None
        correct_offsets = set()
        correct_form = True
        correct_upos = True
        correct_head = True
        correct_head_deprel = True
        for offset, g in gold_offsets.items():
            if ignore_punct and is_punctuation(g["upos"]):
                continue
            if offset in content_offsets:
                c = content_offsets[offset]
                if g.get("form") == c.get("form"):
                    confusion_upos[g.get("upos", "")][c.get("upos", "")] += 1
                    confusion_deprel[g["deprel"]][c.get("deprel", "")] += 1
                    correct_form_token += 1
                    if "upos" in gold[0] and g["upos"] == c.get("upos", ""):
                        correct_upos_token += 1
                    else:
                        correct_upos = False
                    if g["head"] and c["head"] and g["head"]["index"] == c["head"]["index"] and g["head"].get("form") == c["head"].get("form") or g["head"] is None and c["head"] is None:
                        correct_head_token += 1
                        if use_deprel_subtypes:
                            g_deprel = g["deprel"]
                            c_deprel = c["deprel"]
                        else:
                            g_deprel = g["deprel"].split(":")[0]
                            c_deprel = c["deprel"].split(":")[0]
                        if g_deprel == c_deprel:
                            correct_head_deprel_token += 1
                            correct_offsets.add(offset)
                        else:
                            correct_head_deprel = False
                    else:
                        correct_head = False
                        correct_head_deprel = False
                else:
                    if "upos" in g:
                        confusion_upos[g["upos"]][None] += 1
                    confusion_deprel[g["deprel"]][None] += 1
                    correct_form = False
            else:
                if "upos" in g:
                    confusion_upos[g["upos"]][None] += 1
                confusion_deprel[g["deprel"]][None] += 1
                correct_form = False
        if correct_form:
            correct_form_sentence += 1
        if correct_upos:
            correct_upos_sentence += 1
        if correct_head:
            correct_head_sentence += 1
        if correct_head_deprel:
            correct_head_deprel_sentence += 1
        
        if f_conllu:
            print(f"# text = {input_text or gold_text}", file=f_conllu)
            index_map = {r["index"]:i for i, r in enumerate(content, 1)}
            for _, (r, i) in enumerate(zip(content, input_tokens), 1):
                index = index_map[r["index"]]
                head = index_map[r["head"]["index"]] if r["deprel"] != "root" else 0
                form = r.get("form") or i[0]
                upos = r["upos"] if "upos" in r else (i[1] if len(i) > 1 else "_")
                deprel = r["deprel"]
                misc = "SpaceAfter=No" if form == form.rstrip() and _ < len(content) and _ < len(input_tokens) else "_"
                print(index, form, "_", upos, "_", "_", head, deprel, "_", misc, sep="\t", file=f_conllu)
            print(file=f_conllu)

        single_root = is_single_root(content, f_report)
        no_loop = has_no_loop(content, f_report)
        if single_root and no_loop:
            consistent_sentence += 1
            consistent_token += len(content)
        if single_root:
            single_root_sentence += 1
            single_root_token += len(content)
        if no_loop:
            no_loop_sentence += 1
            no_loop_token += len(content)

        if f_report:
            if len(correct_offsets) < len(gold):
                for offset in range(max([len(gold_text), len(content_text)])):
                    if offset in gold_offsets:
                        print("=" if offset in correct_offsets else "<", *reporting_fields(gold_offsets[offset]), sep="\t", file=f_report)
                    if offset in content_offsets and offset not in correct_offsets:
                        print(">", *reporting_fields(content_offsets[offset]), sep="\t", file=f_report)
            print(file=f_report)

    def f1(m: int, g: int, c: int) -> float:
        return 2. / (g / m + c / m) if m > 0 else 0.
    aligned_s = f1(aligned_sentence, gold_sentence, content_sentence)
    aligned_t = f1(aligned_token, gold_token, content_token)
    form_s = f1(correct_form_sentence, gold_sentence, content_sentence)
    form_t = f1(correct_form_token, gold_token, content_token)
    upos = f1(correct_upos_token, gold_token, content_token)
    uas = f1(correct_head_token, gold_token, content_token)
    las = f1(correct_head_deprel_token, gold_token, content_token)
    confusion_upos = {g: {c: v for c, v in sorted(d.items(), key=lambda _: (-_[1], _[0] or ""))} for g, d in confusion_upos.items()}
    confusion_deprel = {g: {c: v for c, v in sorted(d.items(), key=lambda _: (-_[1], _[0] or ""))} for g, d in confusion_deprel.items()}
    return {
        "digest": f"{aligned_s=:.4f}, {aligned_t=:.4f}, {form_s=:.4f}, {form_t=:.4f}, {upos=:.4f}, {uas=:.4f}, {las=:.4f}",
        "sentence": {
            "gold": gold_sentence,
            "content": content_sentence,
            "aligned": aligned_sentence,
            "correct_form": correct_form_sentence,
            "correct_upos": correct_upos_sentence,
            "correct_head": correct_head_sentence,
            "correct_head_deprel": correct_head_deprel_sentence,
            "consistent_dependency": consistent_sentence,
            "single_root": single_root_sentence,
            "no_loop": no_loop_sentence,
        },
        "token": {
            "gold": gold_token,
            "content": content_token,
            "aligned": aligned_token,
            "correct_form": correct_form_token,
            "correct_upos": correct_upos_token,
            "correct_head": correct_head_token,
            "correct_head_deprel": correct_head_deprel_token,
            "consistent_dependency": consistent_token,
            "single_root": single_root_token,
            "no_loop": no_loop_token,
            "recoverd_index": recovered_index_token,
            "recovered_form": recovered_form_token,
        },
        "confusion_upos": confusion_upos,
        "confusion_deprel": confusion_deprel,
    }


def parse_user_prompt(content: str) -> tuple:
    m = USER_PROMPT_SENTENCE_PATTERN.search(content)
    if m:
        input_text = m.group(1)
    else:
        input_text = None
    m = USER_PROMPT_WORDS_PATTERN.search(content)
    if m:
        input_tokens = [_.split("\t") for _ in m.group(1).rstrip("\n").split("\n")]
        if len(input_tokens[0]) > 1:
            input_tokens = [r[1:] for r in input_tokens]
        if not input_text:
            input_text = "".join(r[0] for r in input_tokens)
    else:
        input_tokens = None
    return input_text, input_tokens


def parse_records(content: str, index_field: int, stop_on_error: bool = False) -> list[dict]:
    rows = select_tsv_part(content, min_columns=3)
    if not rows:
        return []
    field_num = len(rows[0])
    form_field = 1 - index_field
    f2_isdigit = rows[0][2].isdigit()
    f4_isdigit = rows[0][4].isdigit() if len(rows[0]) > 4 else False
    records = []
    for r in rows:
        try:
            assert field_num == len(r)
            if field_num == 3:
                records.append({"index": int(r[0]), "head": int(r[1]), "deprel": r[2]})
            elif field_num == 4:
                records.append({"index": int(r[0]), "upos": r[1], "head": int(r[2]), "deprel": r[3]})
                # records.append({"index": int(r[index_field]), "form": r[form_field].replace("　", " "), "head": int(r[2]), "deprel": r[3]})
            elif field_num == 5:
                if f2_isdigit:
                    records.append({"index": int(r[index_field]), "form": r[form_field].replace("　", " "), "head": int(r[2]), "upos": r[3], "deprel": r[4]})
                else:
                    records.append({"index": int(r[index_field]), "form": r[form_field].replace("　", " "), "upos": r[2], "head": int(r[3]), "deprel": r[4]})
            elif field_num == 6:
                if f2_isdigit:
                    records.append({"index": int(r[index_field]), "form": r[form_field].replace("　", " "), "head": int(r[2]), "deprel": r[4], "upos": r[5]})
                elif f4_isdigit:
                    records.append({"index": int(r[index_field]), "form": r[form_field].replace("　", " "), "upos": r[2], "head": int(r[4]), "deprel": r[5]})
                else:
                    records.append({"index": int(r[index_field]), "form": r[form_field].replace("　", " "), "upos": r[2], "head": int(r[3]), "deprel": r[5]})
        except Exception as e:
            if stop_on_error:
                raise e
            break
    if records and "form" in records[-1]:  # eliminate tail whitespaces of last token
        records[-1]["form"] = records[-1]["form"].rstrip(" ")
    for _, r in enumerate(records):
        if r["head"] < 0:
            r["head"] = 1
        elif r["head"] > len(records):
            r["head"] = len(records)
        elif r["head"] == 0 and r["deprel"] != "root":
            r["deprel"] = "root"
    return records


def reporting_fields(r: dict) -> list:
    fields = [r["index"]]
    if "form" in r:
        fields.append(r["form"])
    if "upos" in r:
        fields.append(r["upos"])
    if r["head"]:
        fields += [r["head"]["index"], r["head"]["form"], r["deprel"]] if "form" in r["head"] else [r["head"]["index"], r["deprel"]]
    else:
        fields += [-1, "ROOT", "root"]
    return fields


def is_single_root(records: list[dict], f_report: IO) -> bool:
    if not records:
        return False
    roots = []
    for r in records:
        if r["head"] is None:
            roots.append(r)
    if not roots:
        if f_report:
            print("no-root:", file=f_report)
        return False
    elif len(roots) > 1:
        if f_report:
            print("multi-root:", " ".join(str(_) for _ in roots), file=f_report)
        return False
    else:
        return True


def has_no_loop(records: list[dict], f_report: IO) -> bool:
    if not records:
        return False
    safe_index_set = set()
    for r in records:
        if r["head"] is None:
            safe_index_set.add(r["index"])
            continue
        visited = [r["index"]]
        h = r
        while h["head"] is not None and h["head"]["index"] not in safe_index_set:
            if h["head"]["index"] in visited:
                if f_report:
                    print("loop:", *visited, h["head"]["index"], file=f_report)
                return False
            visited.append(h["head"]["index"])
            h = h["head"]
        safe_index_set |= set(visited)
    assert len(safe_index_set) == len(records), f"{sorted(safe_index_set)}" + "\n" + "\n".join(str(_) for _ in records)
    return True


if __name__ == "__main__":
    main()
