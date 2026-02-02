import json
from pathlib import Path
import random
import sys
import tomllib

from .conllu_to_prompt import *


def main():
    text_length_limit = 1024 * 6
    template_path = sys.argv[1]
    conllu_path_list = sys.argv[2:]
    with open(template_path, "rb") as fin:
        template = tomllib.load(fin)
    for conllu_path in conllu_path_list:
        random.seed(42)
        m = UD_PATH_PATTERN.match(conllu_path) or DATASET_PATH_PATTERN.match(conllu_path)
        lang = m.group(1)
        subset = m.group(5)
        output_path = f"{Path(conllu_path).parent / Path(template_path).stem}.{subset}.jsonl"
        with open(conllu_path, "r", encoding="utf8") as fin:
            conllu_lines = fin.readlines()
        sp_ratio = space_after_ratio(conllu_lines)
        omit_newdoc = doc_per_sentence(conllu_lines) >= 0.5
        skip_until_text = True
        docs = []
        force_newdoc_count = int(random.lognormvariate(0, 1) * 10) + 1
        for _ in conllu_lines:
            m = CONLLU_TEXT_PATTERN.match(_)
            if m:
                if skip_until_text:
                    skip_until_text = False
                    if not docs:
                        docs.append({"sentences": [], "text": ""})
                else:
                    docs[-1]["text"] += sentence_delimiter(sp_ratio)
                docs[-1]["sentences"].append(m.group(1))
                docs[-1]["text"] += m.group(1)
                force_newdoc_count -= 1
            if force_newdoc_count == 0 or (docs and len(docs[-1]["text"]) >= text_length_limit) or (not omit_newdoc and CONLLU_NEWDOC_PATTERN.match(_)):
                if not skip_until_text:
                    skip_until_text = True
                    docs[-1]["text"] += document_delimiter()
                if not docs or docs[-1]["sentences"]:
                    docs.append({"sentences": [], "text": ""})
                force_newdoc_count = int(random.lognormvariate(0, 1) * 10) + 1
            elif not omit_newdoc and CONLLU_NEWPAR_PATTERN.match(_):
                if not skip_until_text:
                    skip_until_text = True
                    docs[-1]["text"] += paragraph_delimiter()
        if not skip_until_text:
            docs[-1]["text"] += document_delimiter()
        with open(output_path, "w", encoding="utf8") as fout:
            for doc in docs:
                json.dump(make_prompt(template, lang, doc["text"], doc["sentences"]), fout, ensure_ascii=False)
                print(file=fout)


def make_prompt(template: list[dict], lang: str, text: str, sentences: list[str]) -> list[dict]:
    sentence_list = "\n".join(sentences)
    prompt = {"messages": deepcopy(template["message_template"])}
    for m in prompt["messages"]:
        for k, v in m.items():
            m[k] = v.replace("<<<LANGUAGE>>>", lang).replace("<<<TEXT>>>", text).replace("<<<SENTENCE_LINES>>>", sentence_list)
    return prompt


def sentence_delimiter(space_after_ratio: float) -> str:
    d = ""
    for _ in range(round((random.random() ** 2) * 2.1 + 0.9)):
        p = random.random()
        if p < 0.02:
            d += " " * (int(random.random() * 3) + 2)
        elif p < 0.03:
            d += "\t"
        elif p < 0.13:
            d += "\n"
        elif p < space_after_ratio:
            d += " "
    return d


def paragraph_delimiter() -> str:
    d = ""
    for _ in range(int((random.random() ** 2) * 2.55 + 0.95)):
        p = random.random()
        if p < 0.02:
            d += " " * (int(random.random() * 3) + 1)
        elif p < 0.03:
            d += "\t"
        else:
            d += "\n"
    return d


def document_delimiter() -> str:
    return "\n" * int((random.random() ** 2) * 2.1 + 0.9)


def space_after_ratio(conllu_lines: list[str]) -> float:
    sentence_count = 0
    token_count = 0
    space_after_count = 0
    for _ in conllu_lines:
        if CONLLU_TEXT_PATTERN.match(_):
            sentence_count += 1
        if CONLLU_TOKEN_PATTERN.match(_):
            token_count += 1
            if "SpaceAfter=No" not in _:
                space_after_count += 1
    return space_after_count / (token_count - sentence_count + 1)


def doc_per_sentence(conllu_lines: list[str]) -> float:
    sentence_count = 0
    newdoc_count = 0
    for _ in conllu_lines:
        if CONLLU_TEXT_PATTERN.match(_):
            sentence_count += 1
        if CONLLU_NEWDOC_PATTERN.match(_):
            newdoc_count += 1
    return newdoc_count / sentence_count


if __name__ == "__main__":
    main()
