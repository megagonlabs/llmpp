import json
import re
import sys

from collections import defaultdict

each_metrics_lines = 100

model_path_pattern = re.compile(r"/(.+?)[-_]([0-9]+[Bb])-((?:stage|phase|mid)[0-9]+)-(?:(ingredient[0-9]+)-)?(?:step|iter)([0-9]+)(?:-tokens[0-9]+B)?(?:[^_]*)_([^_/]+_[^_/]+)_([^./]+)")

offset_multipliers = {
    "OLMo-2": {
        "7B": {
            "stage1": (0, 4096 * 1024),
            "stage2": {
                "ingredient1": (4096 * 1024 * 928646, 4096 * 1024),
                "ingredient2": (4096 * 1024 * (928646 + 11931), 4096 * 1024),
                "ingredient3": (4096 * 1024 * (928646 + 11931 * 2), 4096 * 1024),
            },
        }
    },
    "llm-jp-4": {
        "8b": {
            "phase1": (0, 4096 * 512),
            "phase2": (4096 * 512 * 3000000, 4096 * 2048),
            "mid1": (4096 * (512 * 3000000 + 2048 * 500000), 4096 * 2048),
            "mid2": (4096 * (512 * 3000000 + 2048 * (500000 + 100000)), 16384 * 512),
            "mid3": (4096 * (512 * 3000000 + 2048 * (500000 + 100000)) + 16384 * 512 * 25000, 65536 * 128),
        },
    },
}

upos_list = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

deprel_list = ["acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp", "clf", "compound", "conj", "cop", "csubj", "det", "dep", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark", "nmod", "nsubj", "nummod", "obj", "obl", "orphan", "parataxis", "punct", "reparandum", "root", "vocative", "xcomp"]


def calc_stats(stats, content_stats, keys=None, key_pattern=None):
    gold_total = 0
    content_total = 0
    correct_total = 0
    gold_stats = defaultdict(int)
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
            if gold == content:
                correct_total += c
                correct_stats[gold] += c
    content_total = sum(content_stats.values())
    recalls = {"*": correct_total / gold_total if gold_total else 0.}
    for key in gold_keys:
        recalls[key] = correct_stats.get(key, 0) / gold_stats[key] if gold_stats.get(key) else 0.
    precisions = {"*": correct_total / content_total if content_total else 0.}
    f1s = {}
    for key in gold_keys:
        assert correct_stats.get(key, 0) <= content_stats.get(key, 0), f"{key=}, {correct_stats.get(key, 0)=} > {content_stats.get(key, 0)=}"
        precisions[key] = correct_stats.get(key, 0) / content_stats[key] if content_stats.get(key) else 0.
        f1s[key] = f1(recalls[key], precisions[key])
    return {
        "recall": recalls,
        "precision": precisions,
        "f1": f1s,
    }


def calc_head_stats(stats, content_stats, keys=None, key_pattern=None):
    gold_total = 0
    content_total = 0
    correct_total = 0
    gold_stats = defaultdict(int)
    correct_gold = defaultdict(int)
    correct_content = defaultdict(int)
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
            if (gold == "root" and gold == content) or (gold != "root" and content not in ["-", "null"]):
                correct_total += c
                correct_gold[gold] += c
                correct_content[content] += c
    content_total = sum(content_stats.values())
    recalls = {"*": correct_total / gold_total if gold_total else 0.}
    for key in gold_keys:
        recalls[key] = correct_gold.get(key, 0) / gold_stats[key] if gold_stats.get(key) else 0.
    precisions = {"*": correct_total / content_total if content_total else 0.}
    f1s = {}
    for key in gold_keys:
        assert correct_content.get(key, 0) <= content_stats.get(key, 0), f"{key=}, {correct_content.get(key, 0)=} > {content_stats.get(key, 0)=}"
        precisions[key] = correct_content.get(key, 0) / content_stats[key] if content_stats.get(key) else 0.
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
    treebank = None
    template = None
    for target_path in sys.argv[1:]:
        print(f"target_path: {target_path}", file=sys.stderr)
        m = model_path_pattern.search(target_path)
        model_name = m.group(1)
        model_size = m.group(2)
        stage = m.group(3)
        ingredient = m.group(4)
        step = int(m.group(5))
        if treebank is None:
            treebank = m.group(6)
        if template is None:
            template = m.group(7)
        key = (model_name, model_size, stage, ingredient, step)
        with open(target_path, "r", encoding="utf8") as fin:
            result = json.load(fin)
        upos_stats[key] = calc_stats(result["confusion_upos"], result["content_upos"], keys=upos_list)
        r = upos_stats[key]["recall"]["*"] = result["token"]["correct_upos"] / (result["token"]["gold"] or 1)
        p = upos_stats[key]["precision"]["*"] = result["token"]["correct_upos"] / (result["token"]["content"] or 1)
        upos_stats[key]["f1"]["*"] = f1(r, p)
        deprel_stats[key] = calc_stats(result["confusion_deprel"], result["content_deprel"], keys=deprel_list, key_pattern=r"^([^:]+)")
        r = deprel_stats[key]["recall"]["*"] = result["token"]["correct_head_deprel"] / (result["token"]["gold"] or 1)
        p = deprel_stats[key]["precision"]["*"] = result["token"]["correct_head_deprel"] / (result["token"]["content"] or 1)
        deprel_stats[key]["f1"]["*"] = f1(r, p)
        head_stats[key] = calc_head_stats(result["confusion_deprel"], result["content_deprel"], keys=deprel_list, key_pattern=r"^([^:]+)")
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
        def _sort_steps(key):
            v0 = offset_multipliers.get(key[0])
            if not isinstance(v0, dict):
                return key
            v1 = v0.get(key[1])
            if not isinstance(v1, dict):
                return key
            v2 = v1.get(key[2])
            if v2 is not None:
                key2 = list(v1.keys()).index(key[2])
                if isinstance(v2, dict) and key[3] in v2:
                    key3 = list(v2.keys()).index(key[3])
                else:
                    key3 = key[3]
                return (key[0], key[1], key2, key3, key[4])
            return key
        keys = sorted(stats.keys(), key=_sort_steps)
        labels = sorted(stats[keys[0]][metrics].keys())
        print(title, "", f"{treebank}_{template}", sep="\t")
        print("model", "size", "stage", "ingredient", "step", "tokens", *labels, sep="\t")
        for key in keys:
            stat = stats[key]
            model_name, model_size, stage, ingredient, step = key
            offset_multiplier = offset_multipliers
            for k in [model_name, model_size, stage, ingredient]:
                offset_multiplier = offset_multiplier[k]
                if not isinstance(offset_multiplier, dict):
                    break
            offset, multiplier = offset_multiplier
            tokens = offset + step * multiplier
            print(model_name, model_size, stage, ingredient, step, tokens, *(stat[metrics].get(label, 0.) for label in labels), sep="\t")
        print("\n" * (each_metrics_lines - len(keys) - 2), end="")


if __name__ == "__main__":
    main()
