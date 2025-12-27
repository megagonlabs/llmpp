import re
import sys

from collections import defaultdict


def main():
    language_pattern = re.compile(sys.argv[1])
    input_files = sys.argv[2:]
    language_lines = defaultdict(list)
    for input_file in input_files:
        print("start", input_file, file=sys.stderr)
        with open(input_file, "r", encoding="utf8") as fin:
            for idx, line in enumerate(fin.readlines(), 1):
                m = language_pattern.search(line)
                for g in m.groups() if m else []:
                    if g:
                        language_lines[g].append(line)
                        break
                else:
                    print(f"line #{idx} - no match,", line, end="", file=sys.stderr)
        for language, lines in language_lines.items():
            with open(f"{input_file}__{language}", "w", encoding="utf8") as fout:
                print(*lines, sep="", end="", file=fout)


if __name__ == "__main__":
    main()
