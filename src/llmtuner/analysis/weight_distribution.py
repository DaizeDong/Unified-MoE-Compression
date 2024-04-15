import os

import torch
from matplotlib import pyplot as plt

from transformers import MixtralForCausalLM
from global_utils.io import compress_png_image, create_dir


def plot_graph(weight, fig_name, save_path):
    num_bins = 1000
    create_dir(save_path)

    fig = plt.figure(fig_name)
    ax = fig.add_subplot(1, 1, 1)

    weight = weight.abs()
    min_sim = weight.min().item()
    max_sim = weight.max().item()

    bin_edges = torch.linspace(min_sim, max_sim, num_bins + 1, device="cpu")  # 自定义 bin 的范围和数量
    bin_counts = torch.histc(weight, bins=num_bins, min=min_sim, max=max_sim)  # 使用 torch.histc 进行 bin 统计
    ax.bar(bin_edges[:-1], bin_counts, width=(bin_edges[1] - bin_edges[0]), align="edge", alpha=0.7)
    ax.set_xlabel("Weight Values")
    ax.set_ylabel("Density")
    ax.set_title(fig_name)

    fig.savefig(os.path.join(save_path, fig_name + ".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    compress_png_image(os.path.join(save_path, fig_name + ".png"), print_info=False)
    print(f'Results saved to "{os.path.join(save_path, fig_name + ".png")}"!')


model_path = "/mnt/petrelfs/share_data/quxiaoye/models/Mixtral-8x7B-v0.1"
save_path = "/mnt/petrelfs/dongdaize.d/workspace/compression/results_analysis/weight_distribution"

model = MixtralForCausalLM.from_pretrained(model_path)

for i in range(model.config.num_hidden_layers):
    layer = model.model.layers[i]
    for j in range(model.config.num_local_experts):
        w1 = layer.block_sparse_moe.experts[j].w1.weight.data
        w2 = layer.block_sparse_moe.experts[j].w2.weight.data
        w3 = layer.block_sparse_moe.experts[j].w3.weight.data

        plot_graph(w1, f"w1_expert{j}", os.path.join(save_path, f"layer{i}"))
        plot_graph(w2, f"w2_expert{j}", os.path.join(save_path, f"layer{i}"))
        plot_graph(w3, f"w3_expert{j}", os.path.join(save_path, f"layer{i}"))
print('Done')
