import json
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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    for completion_results_jsonl in args.completion_results_jsonl_files:
        print(completion_results_jsonl)
        try:
            base_path = Path(completion_results_jsonl)
            output_eval_json_path = f"{base_path.parent}/{base_path.stem}.sentence.json"
            with open(completion_results_jsonl, "r", encoding="utf8") as fin:
                completion_results = [json.loads(_)["messages"] for _ in fin]
            stats = eval(completion_results, stop_on_error=args.stop_on_error)
            with open(output_eval_json_path, "w", encoding="utf8") as f_eval:
                json.dump(stats, f_eval, ensure_ascii=False, indent=1)
                print(file=f_eval)
            print(stats["digest"])
            print(completion_results_jsonl, *[stats[u][k] for u in REPORTING_FIELDS for k in REPORTING_FIELDS[u]], sep="\t")
        except Exception as e:
            if args.stop_on_error:
                raise e
            print(e, file=sys.stderr)
            print("skipping", completion_results_jsonl)
        print()


def eval(
    completion_results: list[dict],
    stop_on_error: bool = False,
) -> dict:
    gold_lang_count = 0
    content_lang_count = 0
    correct_lang_count = 0
    gold_sentence_count = 0
    content_sentence_count = 0
    correct_sentence_count = 0
    confusion_lang = defaultdict(lambda: defaultdict(int))

    for line_index, messages in enumerate(completion_results, 1):
        result = messages[-1]
        assert "gold" in result, f"Inference result not saved in line #{line_index}"
        gold_lang, gold_sentences = parse_records(result["gold"], stop_on_error)
        content_lang, content_sentences = parse_records(result["content"], stop_on_error)
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
    correct_s = f1(correct_sentence_count, gold_sentence_count, content_sentence_count)
    correct_l = f1(correct_lang_count, gold_lang_count, content_lang_count)
    return {
        "digest": f"{correct_s=:.4f}, {correct_l=:.4f}",
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


def parse_records(content: str, stop_on_error: bool = False) -> tuple[str, list[str]]:
    lang = ""
    sentences = []
    try:
        lines = [_.strip() for _ in content.split("\n") if _.strip()]
        lang = lines[0]
        sentences = lines[1:]
    except Exception as e:
        if stop_on_error:
            raise e
    return lang, sentences


if __name__ == "__main__":
    main()
