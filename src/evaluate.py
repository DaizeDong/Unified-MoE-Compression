import os
import sys

sys.path = [os.getcwd()] + sys.path

from llmtuner import Evaluator


def main():
    evaluator = Evaluator()
    evaluator.eval()


if __name__ == "__main__":
    main()
