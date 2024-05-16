import os
import sys

sys.path = [os.getcwd()] + sys.path

from llmtuner import Evaluator_Sparse


def main():
    evaluator = Evaluator_Sparse()
    evaluator.eval()


if __name__ == "__main__":
    main()
