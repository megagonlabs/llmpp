import json
import random
import re
import tomllib
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path


def add_args(parser: ArgumentParser = None) -> ArgumentParser:
    parser = parser or ArgumentParser()
    parser.add_argument("--template_toml_list", "--t", nargs="+")
    parser.add_argument("--input_conllu_list", "--i", nargs="+")
    parser.add_argument("--language", "--l")
    parser.add_argument("--name_suffix", "--s")
    parser.add_argument("--pos", choices=["UPOS", "XPOS"], default="UPOS")
    parser.add_argument("--add_whitespace", "--a", choices=["after", "before", False, None], default=None)
    parser.add_argument("--space_between_rrb", "--r")
    parser.add_argument("--mask_rate", "--m", type=float, default=None)
    parser.add_argument("--prefill_rate", "--p", type=float, default=None)
    parser.add_argument("replacements", nargs="*")
    return parser


def main():
    args = add_args().parse_args()
    random.seed(42)
    replacements = [args.replacements[_:_ + 2] for _ in range(0, len(args.replacements), 2)]
    if args.language:
        replacements.append(["LANGUAGE", args.language])
    name_suffix = f"-{args.name_suffix}" if args.name_suffix else ""
    pos = args.pos

    for template_toml in args.template_toml_list:
        template_path = Path(template_toml)
        with open(template_path, "rb") as fin:
            template = tomllib.load(fin)
        template_name = template_path.stem
        message_template = template["message_template"]
        add_whitespace = template.get("add_whitespace", "after") if args.add_whitespace is None else args.add_whitespace
        space_between_rrb = args.space_between_rrb or template.get("space_between_rrb", "")
        mask_rate = template.get("mask_rate", 0.) if args.mask_rate is None else args.mask_rate
        prefill_rate = template.get("prefill_rate", 0.) if args.prefill_rate is None else args.prefill_rate
        prefill_fields = template.get("prefill_fields", [])

        for input_conllu in args.input_conllu_list:
            input_path = Path(input_conllu)
            output_jsonl_path = f"{input_path.parent}/{template_name}{name_suffix}.{input_path.stem.split('-')[-1]}.jsonl"

            with open(input_conllu, "r", encoding="utf8") as fin:
                conllu_lines = fin.readlines()

            outputs = []
            for s in convert_lines(conllu_lines, add_whitespace):
                tokens = [
                    {
                        "INDEX": f["id"] + 1,
                        "ORTH": f["orth"],
                        "ORTHWS": f["orth_with_whitespace"],
                        "UPOS": f["upos"],
                        "XPOS": f["xpos"],
                        "POS": f[pos],
                        "HEAD": 0 if f["label"] == "root" else f["head"] + 1,
                        "HEADORTH": "ROOT" if f["label"] == "root" else s["tokens"][f["head"]]["orth"],
                        "LABEL": f["label"],
                        "CHILDREN": [],
                    } for f in s["tokens"]
                ]
                root = None
                for t in tokens:
                    if t["HEAD"] > 0:
                        t["HEAD_TOKEN"] = tokens[t["HEAD"] - 1]
                        t["HEAD_TOKEN"]["CHILDREN"].append(t)
                    else:
                        root = t
                def traverse(token, f) -> str:
                    l = ""
                    r = ""
                    for t in token["CHILDREN"]:
                        if t["INDEX"] < token["INDEX"]:
                            l += traverse(t, f) + " "
                        else:
                            r += " " + traverse(t, f)
                    return f(token, l, r)
                linearized_deprel = traverse(root, lambda t, l, r: f"({t['LABEL']} {l}({t[pos]} {t['ORTH']}){r}{space_between_rrb})")
                linearized_pos = " ".join(f'({t[pos]} {t["ORTH"]})' for t in tokens)

                masked_token_indexes = set()
                while len(masked_token_indexes) < len(tokens) * mask_rate:
                    masked_token_indexes.add(random.randrange(len(tokens)))
                if mask_rate > 0. and not masked_token_indexes:
                    masked_token_indexes.add(random.randrange(len(tokens)))
                masked_token_indexes = sorted(masked_token_indexes)
                masked_tokens = deepcopy(tokens)
                for _ in masked_token_indexes:
                    masked_tokens[_] = {k: masked_tokens[_][k] for k in ["INDEX", "ORTHWS"]}

                prefilled_token_indexes = []
                for i in range(len(tokens)):
                    if random.random() < prefill_rate:
                        prefilled_token_indexes.append(i)

                messages = deepcopy(message_template)
                def make_tsv(records, fields, prefilled_token_indexes, prefill_fields):
                    return "\n".join(
                        "\t".join(
                            str(r[f]) for f in prefill_fields if f in r
                        ) if  _ in prefilled_token_indexes else "\t".join(
                            str(r[f]) for f in fields if f in r
                        ) for _, r in enumerate(records)
                    )
                for m in messages:
                    for key in m.keys():
                        origin = m[key]
                        result = ""
                        prev = 0
                        for match in re.finditer(r"<<<([^>:]+)>>>|<<<([^>:]+):([^>]*)>>>", m[key]):
                            for meta_name, target in [
                                ["SENTENCE", s["sentence"]],
                                ["TOKEN_NUM", len(s["tokens"])],
                                ["TOKEN_TSV", tokens],
                                ["MASKED_TOKEN_INDEXES", ", ".join(str(_ + 1) for _ in masked_token_indexes)],
                                ["MASKED_TOKEN_TSV", masked_tokens],
                                ["PREFILLED_TOKEN_INDEXES", ", ".join(str(_ + 1) for _ in prefilled_token_indexes)],
                                ["LINEARIZED_POS", linearized_pos],
                                ["LINEARIZED_DEPREL", linearized_deprel],
                            ] + replacements:
                                if meta_name == match.group(1):
                                    result += origin[prev:match.start()]
                                    result += str(target)
                                    prev = match.end()
                                    break
                                if meta_name == match.group(2):
                                    fields = match.group(3).split("_")
                                    result += origin[prev:match.start()]
                                    result += make_tsv(target, fields, prefilled_token_indexes, prefill_fields)
                                    prev = match.end()
                                    break
                            else:
                                assert False, f"meta field not replaced: {match.group(0)}"
                        m[key] = result + m[key][prev:]
                outputs.append(messages)

            with open(output_jsonl_path, "w", encoding="utf8") as fout:
                for messages in outputs:
                    json.dump({"messages": messages}, fout, ensure_ascii=False)
                    print(file=fout)


CONLLU_TEXT_PATTERN = re.compile(
    r"^# text = ?(.+)$"
)
CONLLU_TOKEN_PATTERN = re.compile(
    r"^([1-9][0-9]*)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([0-9]*)\t([^\t]+)\t([^\t]+)\t([^\t]*)$"
)
CONLLU_TOKEN_SKIP_PATTERN = re.compile(
    r"^(([1-9][0-9]*[\-.][1-9][0-9]*)\t|# (sent_id =|text_en =|translit =|source =|generator =|udpipe_model =|note =|auto =|ToDoOrigText =|ToDoOrigtext =|orig_file_sentence|duplicate:) ).+$|^#$|^0(.+)$|^# Tectogrammatical annotation available(.+)$"
)
CONLLU_BUNSETU_PATTERN = re.compile(r"BunsetuBILabel=(.)")


def convert_lines(lines, add_whitespace):
    sentences = []
    tokens = []
    bunsetu = []
    bunsetu_id = 0
    bunsetu_list = []
    state = "text"
    prev_whitespace = False

    for line in lines:
        line = line.rstrip()

        if state == "text":
            m = CONLLU_TEXT_PATTERN.match(line)
            if m is None:
                continue
            sentence = m.group(1)
            state = "token"
            prev_whitespace = False

        elif state == "token" and line != "":
            m = CONLLU_TOKEN_PATTERN.match(line)
            if m is None:
                m = CONLLU_TOKEN_SKIP_PATTERN.match(line)
                assert m is not None, f"{sentence=}, {line=}"
                continue

            token_id = int(m.group(1)) - 1
            orth = m.group(2)
            upos = m.group(4)
            xpos = m.group(5)
            head_id = int(m.group(7)) - 1
            if head_id < 0:
                head_id = token_id
            label = m.group(8).lower()
            options = m.group(10)
            whitespace = options.find("SpaceAfter=No") < 0

            m = CONLLU_BUNSETU_PATTERN.search(options)
            if m and m.group(1) == "B":
                if bunsetu:
                    bunsetu_list.append(
                        (
                            bunsetu,
                            "".join(t["orth"] for t in bunsetu),
                        )
                    )
                    bunsetu_id += 1
                    bunsetu = []

            assert add_whitespace in ["after", "before", False]
            if add_whitespace == "after" and whitespace:
                orth_with_whitespace = f"{orth} "
            elif add_whitespace == "before" and prev_whitespace:
                orth_with_whitespace = f" {orth}"
            else:
                orth_with_whitespace = orth
            token = {
                "id": token_id,
                "orth": orth,
                "orth_with_whitespace": orth_with_whitespace,
                "upos": upos,
                "xpos": xpos,
                "label": label,
                "head": head_id,
                "bunsetu_id": bunsetu_id,
                "whitespace": whitespace,
            }
            tokens.append(token)
            bunsetu.append(token)
            prev_whitespace = whitespace

        elif state == "token" and line == "":
            if tokens[-1]["whitespace"]:
                tokens[-1]["whitespace"] = False
                tokens[-1]["orth_with_whitespace"] = tokens[-1]["orth_with_whitespace"][:-1]
            if bunsetu:
                bunsetu_list.append(
                    (
                        bunsetu,
                        "".join(t["orth_with_whitespace"] for t in bunsetu),
                    )
                )
            assert bunsetu_list, f"{sentence=}, {line=}"
            bunsetu_deps = []
            for bunsetu_id, (bunsetu, bunsetu_orth) in enumerate(bunsetu_list):
                bunsetu_begin = bunsetu[0]["id"]
                bunsetu_end = bunsetu[-1]["id"] + 1
                bunsetu_dep = None
                for token in bunsetu:
                    if token["label"] == "root":
                        bunsetu_head_id = -1
                    elif token["head"] < bunsetu_begin or bunsetu_end <= token["head"]:
                        bunsetu_head_id = tokens[token["head"]]["bunsetu_id"]
                    else:
                        continue
                    bunsetu_dep = {
                        "id": bunsetu_id + 1,
                        "orth": bunsetu_orth,
                        "head": bunsetu_head_id + 1,
                        "label": token["label"],
                    }
                assert bunsetu_dep
                bunsetu_deps.append(bunsetu_dep)

            sentences.append(
                {
                    "sentence": sentence,
                    "bunsetu_deps": bunsetu_deps,
                    "tokens": tokens,
                }
            )

            sentence = ""
            tokens = []
            bunsetu = []
            bunsetu_id = 0
            bunsetu_list = []
            state = "text"

        else:
            assert False, f"{sentence=}, {line=}"

    assert state == "text"

    return sentences


if __name__ == "__main__":
    main()
