import json
import sys
from pathlib import Path


def main():
    print_title_line = True
    fields = []
    files = []
    target = "fields"
    for _ in sys.argv[1:]:
        if _ == "-":
            target = "files"
            continue
        if target == "fields":
            if _.startswith("."):
                fields.append(_[1:])
            else:
                fields.append(_.split("."))
        else:
            files.append(_)
    src_list = []
    for f in files:
        p = Path(f)
        with open(p, "r", encoding="utf8") as fin:
            src_list.append([p, json.load(fin)])
    if not files:
        src_list.append(["-", json.load(sys.stdin)])
    for p, src in src_list:
        record = aggregate(src, fields)
        if print_title_line:
            print("file_name", *[_[0] for _ in record], sep="\t")
            print_title_line = False
        print(p, *[_[1] for _ in record], sep="\t")


def aggregate(src: dict, fields: list[str] | str):
    record = []
    for field in fields:
        if isinstance(field, str):
            record.append(["", field])
            continue
        node = src
        for f in field:
            node = node[f]
        record.append([f, "" if node is None else str(node)])
    return record


if __name__ == "__main__":
    main()
