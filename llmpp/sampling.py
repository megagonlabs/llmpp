import random
import sys


def main():
    prob = float(sys.argv[1])
    for _ in sys.stdin:
        if random.random() <= prob:
            print(_, end="")


if __name__ == "__main__":
    main()
