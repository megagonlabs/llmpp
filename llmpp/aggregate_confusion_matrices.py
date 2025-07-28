import json
import re
import sys

from collections import defaultdict


model_path_pattern = re.compile(r"/(.+?)[-_]([0-9]+)[Bb]-stage([0-9]+)(?:-ingredient)?([0-9]*)-step([0-9]+)-tokens([0-9]+)B_([^_/]+_[^_/]+)_([^./]+)")


def calc_stats(stats, key_pattern=None):
    gold_total = 0
    content_total = 0
    correct_total = 0
    gold_stats = defaultdict(int)
    content_stats = defaultdict(int)
    correct_stats = defaultdict(int)
    if key_pattern:
        gold_keys = sorted({re.search(key_pattern, k).group(1) for k in stats})
    else:
        gold_keys = sorted(stats.keys())
    for gold, stat in stats.items():
        if key_pattern:
            gold = re.search(key_pattern, gold).group(1)
        for content, c in stat.items():
            if key_pattern:
                content = re.search(key_pattern, content).group(1)
            gold_total += c
            gold_stats[gold] += c
            if content in gold_keys:
                content_total += c
                content_stats[content] += c
            if gold == content:
                correct_total += c
                correct_stats[gold] += c
    recalls = {"*": correct_total / gold_total if gold_total else 0.}
    for key in gold_keys:
        recalls[key] = correct_stats.get(key, 0) / gold_stats[key] if gold_stats.get(key) else 0.
    precisions = {"*": correct_total / content_total if content_total else 0.}
    for key in gold_keys:
        precisions[key] = correct_stats.get(key, 0) / content_stats[key] if content_stats.get(key) else 0.
    return {
        "recall": recalls,
        "precision": precisions,
    }



def main():
    upos_stats = {}
    deprel_stats = {}
    for target_path in sys.argv[1:]:
        m = model_path_pattern.search(target_path)
        model_name = m.group(1)
        model_size = m.group(2)
        stage = int(m.group(3))
        ingredient = int(m.group(4) or 0)
        step = int(m.group(5))
        num_tokens = int(m.group(6))
        key = (model_name, model_size, stage, ingredient, step, num_tokens)
        with open(target_path, "r", encoding="utf8") as fin:
            result = json.load(fin)
        upos_stats[key] = calc_stats(result["confusion_upos"])
        deprel_stats[key] = calc_stats(result["confusion_deprel"], r"^([^:]+)")
    for title, stats, metrics in [
        ["UPOS Recall", upos_stats, "recall"],
        ["UPOS Precision", upos_stats, "precision"],
        ["DEPREL Recall", deprel_stats, "recall"],
        ["DEPREL Precision", deprel_stats, "precision"],
    ]:
        keys = sorted(stats.keys())
        labels = sorted(stats[keys[0]][metrics].keys())
        print(title)
        print("Model", "Size", "Stage", "Ingredient", "Step", "Tokens", *labels, sep="\t")
        prev_ingredient = -1
        prev_step = 0
        prev_num_tokens = 0
        step_offset = 0
        num_tokens_offset = 0
        for key in keys:
            stat = stats[key]
            model_name, model_size, stage, ingredient, step, num_tokens = key
            if prev_ingredient != ingredient:
                prev_ingredient = ingredient
                step_offset = prev_step
                num_tokens_offset = prev_num_tokens
            step += step_offset
            num_tokens += num_tokens_offset
            prev_step = step
            prev_num_tokens = num_tokens
            print(model_name, model_size, stage, ingredient, step, num_tokens, *(stat[metrics].get(label, 0.) for label in labels), sep="\t")


if __name__ == "__main__":
    main()
