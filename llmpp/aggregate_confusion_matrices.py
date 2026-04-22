import json
import re
import sys

from collections import defaultdict


model_path_pattern = re.compile(r"/(.+?)[-_]([0-9]+)[Bb]-stage([0-9]+)(?:-ingredient)?([0-9]*)-step([0-9]+)-tokens([0-9]+)B([^_]*)_([^_/]+_[^_/]+)_([^./]+)")

upos_list = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

deprel_list = ["acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp", "clf", "compound", "conj", "cop", "csubj", "det", "dep", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark", "nmod", "nsubj", "nummod", "obj", "obl", "orphan", "parataxis", "punct", "reparandum", "root", "vocative", "xcomp"]


def calc_stats(stats, keys=None, key_pattern=None):
    gold_total = 0
    content_total = 0
    correct_total = 0
    gold_stats = defaultdict(int)
    content_stats = defaultdict(int)
    correct_stats = defaultdict(int)
    if keys:
        gold_keys = sorted(keys)
    elif key_pattern:
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
            if content != "null":
                content_total += c
                content_stats[gold] += c
                if gold == content:
                    correct_total += c
                    correct_stats[gold] += c
    recalls = {"*": correct_total / gold_total if gold_total else 0.}
    for key in gold_keys:
        recalls[key] = correct_stats.get(key, 0) / gold_stats[key] if gold_stats.get(key) else 0.
    precisions = {"*": correct_total / content_total if content_total else 0.}
    f1s = {}
    for key in gold_keys:
        precisions[key] = correct_stats.get(key, 0) / content_stats[key] if content_stats.get(key) else 0.
        f1s[key] = f1(recalls[key], precisions[key])
    return {
        "recall": recalls,
        "precision": precisions,
        "f1": f1s,
    }


def calc_head_stats(stats, keys=None, key_pattern=None):
    gold_total = 0
    content_total = 0
    correct_total = 0
    gold_stats = defaultdict(int)
    content_stats = defaultdict(int)
    correct_stats = defaultdict(int)
    if keys:
        gold_keys = sorted(keys)
    elif key_pattern:
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
            if content != "null":
                content_total += c
                content_stats[gold] += c
                if content != "-":
                    correct_total += c
                    correct_stats[gold] += c
    recalls = {"*": correct_total / gold_total if gold_total else 0.}
    for key in gold_keys:
        recalls[key] = correct_stats.get(key, 0) / gold_stats[key] if gold_stats.get(key) else 0.
    precisions = {"*": correct_total / content_total if content_total else 0.}
    f1s = {}
    for key in gold_keys:
        precisions[key] = correct_stats.get(key, 0) / content_stats[key] if content_stats.get(key) else 0.
        f1s[key] = f1(recalls[key], precisions[key])
    return {
        "recall": recalls,
        "precision": precisions,
        "f1": f1s,
    }


def f1(r, p):
    return 2. * r * p / (r + p) if r * p else 0.

def main():
    upos_stats = {}
    deprel_stats = {}
    head_stats = {}
    total_stats = {}
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
        upos_stats[key] = calc_stats(result["confusion_upos"], keys=upos_list)
        r = upos_stats[key]["recall"]["*"] = result["token"]["correct_upos"] / (result["token"]["gold"] or 1)
        p = upos_stats[key]["precision"]["*"] = result["token"]["correct_upos"] / (result["token"]["content"] or 1)
        upos_stats[key]["f1"]["*"] = f1(r, p)
        deprel_stats[key] = calc_stats(result["confusion_deprel"], keys=deprel_list, key_pattern=r"^([^:]+)")
        r = deprel_stats[key]["recall"]["*"] = result["token"]["correct_head_deprel"] / (result["token"]["gold"] or 1)
        p = deprel_stats[key]["precision"]["*"] = result["token"]["correct_head_deprel"] / (result["token"]["content"] or 1)
        deprel_stats[key]["f1"]["*"] = f1(r, p)
        head_stats[key] = calc_head_stats(result["confusion_deprel"], keys=deprel_list, key_pattern=r"^([^:]+)")
        r = head_stats[key]["recall"]["*"] = result["token"]["correct_head"] / (result["token"]["gold"] or 1)
        p = head_stats[key]["precision"]["*"] = result["token"]["correct_head"] / (result["token"]["content"] or 1)
        head_stats[key]["f1"]["*"] = f1(r, p)
        total_stats[key] = {}
        total_stats[key]["recall"] = {
            "Aligned": result["token"]["aligned"] / (result["token"]["gold"] or 1),
            "UAS": result["token"]["correct_head"] / (result["token"]["gold"] or 1),
        }
        total_stats[key]["precision"] = {
            "Aligned": result["token"]["aligned"] / (result["token"]["content"] or 1),
            "UAS": result["token"]["correct_head"] / (result["token"]["content"] or 1),
        }
        total_stats[key]["f1"] ={
            "Aligned": f1(total_stats[key]["recall"]["Aligned"], total_stats[key]["precision"]["Aligned"]),
            "UAS": f1(total_stats[key]["recall"]["UAS"], total_stats[key]["precision"]["UAS"]),
        }
    for title, stats, metrics in [
        ["UPOS Recall", upos_stats, "recall"],
        ["UPOS Precision", upos_stats, "precision"],
        ["UPOS F1", upos_stats, "f1"],
        ["DEPREL Recall", deprel_stats, "recall"],
        ["DEPREL Precision", deprel_stats, "precision"],
        ["DEPREL F1", deprel_stats, "f1"],
        ["HEAD Recall", head_stats, "recall"],
        ["HEAD Precision", head_stats, "precision"],
        ["HEAD F1", head_stats, "f1"],
        ["TOTAL Recall", total_stats, "recall"],
        ["TOTAL Precision", total_stats, "precision"],
        ["TOTAL F1", total_stats, "f1"],
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
