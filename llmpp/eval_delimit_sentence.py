import json
import re
import sys
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path


REPORTING_FIELDS = {
    "sentence": ["gold", "content", "correct"],
    "lang": ["gold", "content", "correct"],
}


def parse_args() -> Namespace:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("completion_results_jsonl_files", nargs="+")
    parser.add_argument("--stop_on_error", "--e", action="store_true")
    parser.add_argument("--task_regexp", "--t", default=r"sentence delimitation")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    task_regexp = re.compile(args.task_regexp)
    for completion_results_jsonl in args.completion_results_jsonl_files:
        try:
            if completion_results_jsonl.endswith(".jsonl"):
                base_path = Path(completion_results_jsonl)
                output_eval_json_path = f"{base_path.parent}/{base_path.stem}.sentence.json"
            else:
                output_eval_json_path = f"{base_path}.sentence.json"
            with open(completion_results_jsonl, "r", encoding="utf8") as fin:
                completion_results = [m for m in [json.loads(_)["messages"] for _ in fin] if task_regexp.search(m[-2]["content"])]
            stats = eval(completion_results)
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
) -> dict:
    gold_lang_count = 0
    content_lang_count = 0
    correct_lang_count = 0
    gold_sentence_count = 0
    content_sentence_count = 0
    correct_sentence_count = 0
    confusion_lang = defaultdict(lambda: defaultdict(int))

    for line_index, messages in enumerate(completion_results, 1):
        if "sentence delimitation" not in messages[-2]["content"]:
            continue
        result = messages[-1]
        assert "gold" in result, f"Inference result not saved in line #{line_index}"
        gold_lang, gold_sentences = parse_records(result["gold"])
        content_lang, content_sentences = parse_records(result["content"])
        gold_lang_count += 1
        if content_lang:
            content_lang_count += 1
        confusion_lang[gold_lang][content_lang] += 1
        if gold_lang == content_lang:
            correct_lang_count += 1
        gold_sentence_count += len(gold_sentences)
        content_sentence_count += len(content_sentences)
        for i, (g, c) in enumerate(zip(gold_sentences, content_sentences)):
            if g == c:
                correct_sentence_count += 1
            else:
                for i in range(-1, -len(gold_sentences) + i - 1, -1):
                    if len(content_sentences) + i < 0:
                        break
                    if gold_sentences[i] == content_sentences[i]:
                        correct_sentence_count += 1
                break
    def f1(m: int, g: int, c: int) -> float:
        return 2. / (g / m + c / m) if m > 0 else 0.
    correct_sent = f1(correct_sentence_count, gold_sentence_count, content_sentence_count)
    correct_lang = f1(correct_lang_count, gold_lang_count, content_lang_count)
    return {
        "digest": f"{correct_sent=:.4f}, {correct_lang=:.4f}",
        "sentence": {
            "gold": gold_sentence_count,
            "content": content_sentence_count,
            "correct": correct_sentence_count,
        },
        "lang": {
            "gold": gold_lang_count,
            "content": content_lang_count,
            "correct": correct_lang_count,
        },
        "confusion_lang": confusion_lang,
    }


def parse_records(content: str) -> tuple[str, list[str]]:
    lines = content.split("\n")
    l0 = lines[0]
    if len(lines) < 3 or not l0 or "\t" in l0 or l0.startswith("- Task") or lines[1] or not lines[2]:
        return None, []  # no delimiting task there
    lang = l0
    mode = "token" if "\t" in lines[2] else "sentence"
    sentences = []
    sentence = ""
    for _ in lines[2:]:
        if mode == "sentence":
            if _:
                sentences.append(_)
            else:
                break
        elif mode == "token":
            if "\t" in _:
                sentence += _.split("\t")[1]
            else:
                if sentence:
                    sentences.append(sentence)
                    sentence = ""
                else:
                    break
        else:
            assert False, f"invalid mode: {mode}"
    if sentence:
        sentences.append(sentence)
    return lang, sentences


if __name__ == "__main__":
    main()
