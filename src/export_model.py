import os
import sys

sys.path = [os.getcwd()] + sys.path

from llmtuner import export_model


def main():
    export_model()


if __name__ == "__main__":
    main()
