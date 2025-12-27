import re
import sys

from collections import defaultdict


def main():
    language_pattern = re.compile(sys.argv[1])
    output_path_base = sys.argv[2]
    input_files = sys.argv[3:]
    language_lines = defaultdict(list)
    for input_file in input_files:
        with open(input_file, "r", encoding="utf8") as fin:
            for line in fin.readlines():
                m = language_pattern.search(line)
                for g in m.groups() if m else []:
                    if g:
                        language_lines[g].append(line)
                        break
                else:
                    print("no match:", line, end="", file=sys.stderr)
    for language, lines in language_lines.items():
        with open(f"{output_path_base}.{language}", "w", encoding="utf8") as fout:
            print(*lines, sep="", end="", file=fout)


if __name__ == "__main__":
    main()
