import os
import sys

sys.path = [os.getcwd()] + sys.path

from torch.utils.flop_counter import FlopCounterMode
from global_utils.io import create_dir
from llmtuner.model.deepseek.configuration_deepseek import DeepseekConfig
from llmtuner.model.deepseek.modeling_deepseek import DeepseekModel, DeepseekForCausalLM

import torch
import argparse
import transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer


AutoConfig.register("deepseek", DeepseekConfig)
AutoModel.register(DeepseekConfig, DeepseekModel)
AutoModelForCausalLM.register(DeepseekConfig, DeepseekForCausalLM)


@torch.no_grad()
def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    inputs = tokenizer.encode("hi, hi, hi, hi, hi, hi, ok, ok, ok, ok, a, a, a, a", return_tensors="pt")
    inputs = [inputs for _ in range(1 + args.seq_len // inputs.shape[1])]
    inputs = torch.cat(inputs, dim=1)[:, :args.seq_len]
    inputs = inputs.expand(args.batch_size, args.seq_len)
    inputs = inputs.to(args.device)
    print(inputs, inputs.shape)

    model.to(args.device)
    model.eval()

    flop_counter = FlopCounterMode(model, depth=5, display=True)
    with flop_counter:
        model(input_ids=inputs)
    print(flop_counter.get_total_flops())

    if args.save_file is not None:
        create_dir(os.path.dirname(args.save_file))
        with open(args.save_file, "w") as f:
            f.write(str(flop_counter.get_total_flops()) + "\n")
            f.write(str(flop_counter.get_table()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/mnt/petrelfs/share_data/quxiaoye/models/Mixtral-8x7B-v0.1")
    parser.add_argument("--save_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
