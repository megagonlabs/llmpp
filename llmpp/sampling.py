import random
import sys


def main():
    prob = float(sys.argv[1])
    random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 42)
    for _ in sys.stdin:
        if random.random() <= prob:
            print(_, end="")


if __name__ == "__main__":
    main()
