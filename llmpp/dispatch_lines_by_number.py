import sys

from pathlib import Path


def main():
    suffix_num_lines_csv = sys.argv[1]
    input_files = sys.argv[2:]
    suffix_num_lines = {k: int(v) for k, v in [_.split(":") for _ in suffix_num_lines_csv.split(",")]}
    total_num_lines = sum(suffix_num_lines.values())
    print(f"{total_num_lines=}, {suffix_num_lines=}")
    for input_file in input_files:
        print("input path:", input_file)
        with open(input_file, "r", encoding="utf8") as fin:
            lines = fin.readlines()
        if len(lines) != total_num_lines:
            print(f"skipping due to inconsistency ({len(lines)=} != {total_num_lines}): {input_file}", file=sys.stderr)
            continue
        offset = 0
        base_path = Path(input_file)
        for suffix, num_lines in suffix_num_lines.items():
            output_path = f"{base_path.parent}/{base_path.stem}-{suffix}.{base_path.suffix}"
            with open(output_path, "w", encoding="utf8") as fout:
                print(*lines[offset:offset + num_lines], sep="", end="", file=fout)
                offset += num_lines
            print("   created:", output_path)


if __name__ == "__main__":
    main()
