import logging
import math
import os
import sys
from argparse import Namespace

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm

from llmtuner.model.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralPreTrainedModel
from .io import create_dir
from .utils import print_gpu_memory, prepare_calibration_input, get_moe_model_information
from .wrapper import HiddenStatesRecordWrapper
from ...model.deepseek.modeling_deepseek import DeepseekPreTrainedModel

logger = logging.getLogger(__name__)


@no_grad()
def get_layer_similarities(model: MixtralForCausalLM, dataloader: DataLoader, accelerator: Accelerator, num_samples: int, drop_norm: bool, cache_file=None):
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

        # 🔍 Get MoE information
        _, num_layers, moe_layer_indices, valid_moe_layer_indices = get_moe_model_information(unwrapped_model, accelerator)

        # 🔍 Initialize the similarities.
        # Row: each layer
        # Column: similarity to the next n layer
        # Example: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # shape(6)
        similarities = torch.full((num_layers,), -math.inf, device=device)

        accelerator.print('Starting ...')
        for i in tqdm(range(num_layers), desc="Recording hidden states...", disable=not accelerator.is_main_process):
            sys.stderr.flush()
            torch.cuda.empty_cache()
            print_gpu_memory(accelerator)
            layer = layers[i]

            if i in moe_layer_indices:  # this block is MoE, not dense
                if isinstance(unwrapped_model, MixtralPreTrainedModel):  # 🔍
                    mlp_pre_norm = layer.post_attention_layernorm
                    mlp = layer.block_sparse_moe
                elif isinstance(unwrapped_model, DeepseekPreTrainedModel):  # 🔍
                    mlp_pre_norm = layer.post_attention_layernorm
                    mlp = layer.mlp
                else:
                    raise NotImplementedError

                if drop_norm:
                    wrapped_mlp_pre_norm = HiddenStatesRecordWrapper(mlp_pre_norm, record_input=True, record_output=False)  # 🔍 Wrap layer
                else:
                    wrapped_mlp_pre_norm = HiddenStatesRecordWrapper(mlp_pre_norm, record_input=False, record_output=True)  # 🔍 Wrap layer
                wrapped_mlp = HiddenStatesRecordWrapper(mlp, record_input=False, record_output=True)  # 🔍 Wrap layer

                # Forward hook for recording hidden states
                def record_mlp_pre_norm_states_hook(_, input, output):
                    wrapped_mlp_pre_norm.record(input[0].data, output[0].data)

                def record_mlp_states_hook(_, input, output):
                    wrapped_mlp.record(input[0].data, output[0].data)

                # Get hidden states
                handles = []
                handles.append(mlp_pre_norm.register_forward_hook(record_mlp_pre_norm_states_hook))
                handles.append(mlp.register_forward_hook(record_mlp_states_hook))
                for j in range(num_samples):
                    outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
                for handle in handles:
                    handle.remove()

                # 🔍 Automatically choose the dtype to prevent OOM
                dtype = torch.float32 if num_samples <= 64 else torch.bfloat16

                if drop_norm:
                    input_hidden_states = torch.cat(wrapped_mlp_pre_norm.input_hidden_states, dim=0).to(dtype).to(device)
                    output_hidden_states = input_hidden_states + torch.cat(wrapped_mlp.output_hidden_states, dim=0).to(dtype).to(device)
                else:
                    input_hidden_states = torch.cat(wrapped_mlp_pre_norm.output_hidden_states, dim=0).to(dtype).to(device)
                    output_hidden_states = torch.cat(wrapped_mlp.output_hidden_states, dim=0).to(dtype).to(device)

                # 🔍 Calculate similarity (output+input due to residual connection)
                cos_sim = F.cosine_similarity(input_hidden_states, output_hidden_states, dim=-1)  # (total_token_num)
                cos_sim = cos_sim.mean()
                cos_sim = accelerator.reduce(cos_sim, reduction="mean")  # 🔍 All reduce across devices
                accelerator.print(f'layer {i} similarity: {cos_sim.item()}')

                similarities[i] = cos_sim

            else:  # this layer is dense, not MoE
                for j in range(num_samples):
                    outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]

            # Update inputs & outputs
            inputs, outputs = outputs, inputs

        # Save to the cache file
        if cache_file is not None:
            if accelerator.is_main_process:
                create_dir(os.path.dirname(cache_file))
                torch.save(similarities.clone().cpu(), cache_file)
                print(f"Saving cached similarities to {cache_file}")
            accelerator.wait_for_everyone()

    accelerator.print("similarities\n", similarities)

    return similarities


def discrete_layer_dropping(args: Namespace, model: MixtralForCausalLM, dataloader: DataLoader, accelerator: Accelerator, num_samples: int):
    """
    🔍 Prune mlp layers in a discrete order.
    E.g., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -> [0, 2, 6, 8, 9]
    """
    drop_n = args.drop_n

    similarities = get_layer_similarities(model, dataloader, accelerator, num_samples, args.layer_drop_norm, cache_file=args.similarity_cache_file)
    sorted_similarities, sorted_layer_id = torch.sort(similarities, dim=0, descending=True)

    dropped_layer_list = sorted_layer_id[:drop_n].tolist()
    accelerator.print(f"Dropped layer: {dropped_layer_list}, similarities: {sorted_similarities[:drop_n].tolist()}")
    return dropped_layer_list


def post_layers_drop(compressed_model_save_path, model, tokenizer, reserved_layer_list, accelerator: Accelerator):
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    layers = unwrapped_model.model.layers

    # 🔍 Get MoE information
    _, _, moe_layer_indices, valid_moe_layer_indices = get_moe_model_information(unwrapped_model, accelerator)

    num_experts = []

    if accelerator.is_main_process:
        for layer_id, layer in tqdm(list(enumerate(layers)), desc='Dropping MLPs...'):
            if layer_id in moe_layer_indices:  # this block is MoE, not dense
                if layer_id in reserved_layer_list:
                    if isinstance(unwrapped_model, MixtralPreTrainedModel):  # 🔍
                        num_experts.append(unwrapped_model.config.num_local_experts[layer_id] if isinstance(unwrapped_model.config.num_local_experts, list) else unwrapped_model.config.num_local_experts)
                    elif isinstance(unwrapped_model, DeepseekPreTrainedModel):  # 🔍
                        num_experts.append(unwrapped_model.config.n_routed_experts[layer_id] if isinstance(unwrapped_model.config.n_routed_experts, list) else unwrapped_model.config.n_routed_experts)
                    else:
                        raise NotImplementedError
                else:
                    if isinstance(unwrapped_model, MixtralPreTrainedModel):  # 🔍
                        layer.post_attention_layernorm = None
                        layer.block_sparse_moe = None
                    elif isinstance(unwrapped_model, DeepseekPreTrainedModel):  # 🔍
                        layer.post_attention_layernorm = None
                        layer.mlp = None
                    else:
                        raise NotImplementedError
                    num_experts.append(-1)  # append -1 to mark that the layer has no MoE and Norm

                    if hasattr(unwrapped_model.config, "layer_experts_idx"):  # for compatibility with Expert Drop
                        unwrapped_model.config.layer_experts_idx[layer_id] = None

            else:  # this layer is dense, not MoE
                num_experts.append(None)

        if isinstance(unwrapped_model, MixtralPreTrainedModel):  # 🔍
            unwrapped_model.config.num_local_experts = num_experts
        elif isinstance(unwrapped_model, DeepseekPreTrainedModel):  # 🔍
            unwrapped_model.config.n_routed_experts = num_experts
        else:
            raise NotImplementedError

        accelerator.print("Saving...")
        unwrapped_model.save_pretrained(compressed_model_save_path)
        tokenizer.save_pretrained(compressed_model_save_path)

    accelerator.wait_for_everyone()
    accelerator.print(f"model: {model}")
