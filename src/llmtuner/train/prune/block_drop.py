import logging
import math
import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from argparse import Namespace
from torch import no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm

from global_utils.io import create_dir
from llmtuner.train.prune.utils import prepare_calibration_input, print_gpu_memory
from llmtuner.train.prune.wrapper import InputStatesRecordWrapper
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM

logger = logging.getLogger(__name__)


@no_grad()
def get_layer_similarities(model: MixtralForCausalLM, dataloader: DataLoader, accelerator: Accelerator, num_samples: int, cache_file=None):
    device = accelerator.device

    if cache_file is not None and os.path.exists(cache_file):
        # use cached file
        accelerator.print(f"Loading cached model from {cache_file}")
        similarities = torch.load(cache_file, map_location=device)

    else:
        # calculate similarities
        accelerator.print(f"No cached model found. Running model on {num_samples} samples for each device.")
        unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
        unwrapped_model.config.use_cache = False
        layers = unwrapped_model.model.layers

        accelerator.print("Getting features...")
        inputs, outputs, attention_mask, position_ids = prepare_calibration_input(unwrapped_model, dataloader, num_samples)  # 🔍

        # 🔍 Initialize the similarities.
        # Row: each layer
        # Column: similarity to the next n layer
        # Example: [[ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
        #           [ 0.5,  0.5,  0.5,  0.5,  0.5, -inf],
        #           [ 0.5,  0.5,  0.5,  0.5, -inf, -inf],
        #           [ 0.5,  0.5,  0.5, -inf, -inf, -inf],
        #           [ 0.5,  0.5, -inf, -inf, -inf, -inf],
        #           [ 0.5, -inf, -inf, -inf, -inf, -inf]]  # shape(6, 6)
        similarities = torch.full((len(layers), len(layers)), -math.inf, device=device)

        accelerator.print('Starting ...')
        wrapped_layers = []

        for i in tqdm(range(len(layers)), desc="Recording hidden states...", disable=not accelerator.is_main_process):
            layer = layers[i]
            wrapped_layer = InputStatesRecordWrapper(layer, record_output=i == len(layers) - 1)  # 🔍 Wrap layer
            wrapped_layers.append(wrapped_layer)

            # Forward hook for recording hidden states
            def record_states_hook(_, input, output):
                wrapped_layer.record(input[0].data, output[0].data)

            # Get states
            handle = layer.register_forward_hook(record_states_hook)
            for j in range(num_samples):
                outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
            handle.remove()

            # Update inputs & outputs
            inputs, outputs = outputs, inputs
            print_gpu_memory(accelerator)

        all_hidden_states = []
        for i in tqdm(range(len(layers)), desc="Concatenating hidden states...", disable=not accelerator.is_main_process):
            all_hidden_states.append(torch.cat(wrapped_layers[i].hidden_states, dim=0).to(device))  # (total_token_num, hidden_size)
        all_hidden_states.append(torch.cat(wrapped_layers[-1].output_hidden_states, dim=0).to(device))
        accelerator.print(f'Total {len(all_hidden_states)} hidden states concatenated.')

        for i in tqdm(range(len(all_hidden_states)), desc="Calculating similarities...", disable=not accelerator.is_main_process):
            for j in range(i + 1, len(all_hidden_states)):
                packed_hidden_states_layer_i = all_hidden_states[i]
                packed_hidden_states_layer_j = all_hidden_states[j]
                index_gap = j - i

                cos_sim = F.cosine_similarity(packed_hidden_states_layer_i, packed_hidden_states_layer_j, dim=-1)  # (total_token_num)
                cos_sim = cos_sim.mean()
                cos_sim = accelerator.reduce(cos_sim, reduction="mean")  # 🔍 All reduce across devices

                similarities[i, index_gap - 1] = cos_sim

        # Save to the cache file
        if cache_file is not None:
            if accelerator.is_main_process:
                create_dir(os.path.dirname(cache_file))
                torch.save(similarities.clone().cpu(), cache_file)
                print(f"Saving cached model to {cache_file}")
            accelerator.wait_for_everyone()

    accelerator.print("similarities\n", similarities)

    return similarities


def consecutive_dropping(args: Namespace, model: MixtralForCausalLM, dataloader: DataLoader, accelerator: Accelerator, num_samples: int):
    """
    🔍 Prune blocks in a consecutive order.
    E.g., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -> [0, 1, 7, 8, 9]
    """
    drop_n = args.drop_n

    similarities = get_layer_similarities(model, dataloader, accelerator, num_samples, cache_file=args.similarity_cache_file)

    max_similarity, begin_layer_id = torch.max(similarities, dim=1)  # shape(layer_num)
    max_similarity = max_similarity[drop_n - 1].item()
    begin_layer_id = begin_layer_id[drop_n - 1].item()
    end_layer_id = begin_layer_id + drop_n
    dropped_layer_list = [i for i in range(begin_layer_id, end_layer_id)]

    accelerator.print(f"Dropped layer: {dropped_layer_list}, max_similarity: {max_similarity}")
    return dropped_layer_list


def discrete_dropping(args: Namespace, model: MixtralForCausalLM, dataloader: DataLoader, accelerator: Accelerator, num_samples: int):
    """
    🔍 Prune blocks in a discrete order.
    E.g., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -> [0, 2, 6, 8, 9]
    """
    similarities = get_layer_similarities(model, dataloader, accelerator, num_samples, cache_file=args.similarity_cache_file)

    # 这段是自己加的，用贪心来搜索，每次剪掉 cosine 相似度最大的连续 k ( 1 <= k <= drop_n ) 层，不断迭代，直到剪完 drop_n 个层结束
    # 先不实现这个，把 paper 的搞定了再说
    raise NotImplementedError
