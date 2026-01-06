import sys


def main():
    suffix_num_lines_csv = sys.argv[1]
    input_files = sys.argv[2:]
    suffix_num_lines = {k: int(v) for k, v in [_.split(":") for _ in suffix_num_lines_csv.split(",")]}
    total_num_lines = sum(suffix_num_lines.values())
    print(f"{total_num_lines=}, {suffix_num_lines=}", file=sys.stderr)
    for input_file in input_files:
        print("input path:", input_file, file=sys.stderr)
        with open(input_file, "r", encoding="utf8") as fin:
            lines = fin.readlines()
        assert len(lines) == total_num_lines, f"{len(lines)=} != {total_num_lines}"
        offset = 0
        for suffix, num_lines in suffix_num_lines.items():
            output_file = f"{input_file}-{suffix}"
            with open(output_file, "w", encoding="utf8") as fout:
                print(*lines[offset:offset + num_lines], sep="", end="", file=fout)
                offset += num_lines
            print("    create:", output_file, file=sys.stderr)


if __name__ == "__main__":
    main()
