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

from global_utils.io import create_dir
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralPreTrainedModel
from .utils import print_gpu_memory, prepare_calibration_input
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

        # 🔍 Get MoE layer ids
        if isinstance(unwrapped_model, MixtralPreTrainedModel):
            num_layers = unwrapped_model.config.num_hidden_layers
            moe_layer_indices = list(range(num_layers))
        elif isinstance(unwrapped_model, DeepseekPreTrainedModel):
            num_layers = unwrapped_model.config.num_hidden_layers
            moe_layer_indices = [layer_idx for layer_idx in range(num_layers) if (unwrapped_model.config.n_routed_experts is not None and layer_idx >= unwrapped_model.config.first_k_dense_replace and layer_idx % unwrapped_model.config.moe_layer_freq == 0)]
        accelerator.print("moe_layer_indices", moe_layer_indices)

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

            if i in moe_layer_indices:
                if isinstance(unwrapped_model, MixtralPreTrainedModel):  # 🔍
                    mlp_pre_norm = layer.post_attention_layernorm
                    mlp = layer.block_sparse_moe
                elif isinstance(unwrapped_model, DeepseekPreTrainedModel):  # 🔍
                    mlp_pre_norm = layer.post_attention_layernorm
                    mlp = layer.mlp

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

                dtype = torch.float32 if num_samples <= 64 else torch.bfloat16

                if drop_norm:
                    input_hidden_states = torch.cat(wrapped_mlp_pre_norm.input_hidden_states, dim=0).to(dtype).to(device)
                    output_hidden_states = input_hidden_states + torch.cat(wrapped_mlp.output_hidden_states, dim=0).to(dtype).to(device)
                else:
                    input_hidden_states = torch.cat(wrapped_mlp_pre_norm.output_hidden_states, dim=0).to(dtype).to(device)
                    output_hidden_states = torch.cat(wrapped_mlp.output_hidden_states, dim=0).to(dtype).to(device)
                # accelerator.print('layer', i)
                # accelerator.print('input_hidden_states', input_hidden_states)
                # accelerator.print('output_hidden_states', output_hidden_states)

                # 🔍 Calculate similarity (output+input due to residual connection)
                cos_sim = F.cosine_similarity(input_hidden_states, output_hidden_states, dim=-1)  # (total_token_num)
                cos_sim = cos_sim.mean()
                cos_sim = accelerator.reduce(cos_sim, reduction="mean")  # 🔍 All reduce across devices
                accelerator.print(f'layer {i} similarity: {cos_sim.item()}')

                similarities[i] = cos_sim

            else:
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
    # similarities = get_layer_similarities(model, dataloader, accelerator, num_samples, args.layer_drop_norm, cache_file=None)

    sorted_similarities, sorted_layer_id = torch.sort(similarities, dim=0, descending=True)

    dropped_layer_list = sorted_layer_id[:drop_n].tolist()
    accelerator.print(f"Dropped layer: {dropped_layer_list}, similarities: {sorted_similarities[:drop_n].tolist()}")
    return dropped_layer_list


def post_layers_drop(prune_model_save_path, model, tokenizer, reserved_layer_list, accelerator: Accelerator):
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    layers = unwrapped_model.model.layers

    num_experts = []

    if accelerator.is_main_process:
        for layer_id, layer in tqdm(list(enumerate(layers)), desc='Dropping MLPs...'):
            if layer_id in reserved_layer_list:
                if isinstance(unwrapped_model, MixtralPreTrainedModel):  # 🔍
                    num_experts.append(unwrapped_model.config.num_local_experts[layer_id] if isinstance(unwrapped_model.config.num_local_experts, list) else unwrapped_model.config.num_local_experts)
                elif isinstance(unwrapped_model, DeepseekPreTrainedModel):  # 🔍
                    num_experts.append(unwrapped_model.config.n_routed_experts[layer_id] if isinstance(unwrapped_model.config.n_routed_experts, list) else unwrapped_model.config.n_routed_experts)
            else:
                if isinstance(unwrapped_model, MixtralPreTrainedModel):  # 🔍
                    layer.post_attention_layernorm = None
                    layer.block_sparse_moe = None
                elif isinstance(unwrapped_model, DeepseekPreTrainedModel):  # 🔍
                    layer.post_attention_layernorm = None
                    layer.mlp = None
                num_experts.append(-1)  # 🔍 append -1 to mark that the layer has no MoE and Norm

        if isinstance(unwrapped_model, MixtralPreTrainedModel):  # 🔍
            unwrapped_model.config.num_local_experts = num_experts
        elif isinstance(unwrapped_model, DeepseekPreTrainedModel):  # 🔍
            unwrapped_model.config.n_routed_experts = num_experts

        accelerator.print("Saving...")
        unwrapped_model.save_pretrained(prune_model_save_path)
        tokenizer.save_pretrained(prune_model_save_path)

    accelerator.wait_for_everyone()
    accelerator.print(f"model: {model}")

# def post_layer_drop(prune_model_save_path, model, tokenizer, layer_id_mapping, accelerator):
#     # get state dict
#     state_dict = model.state_dict()
#     accelerator.print(f"layer_id_mapping: {layer_id_mapping}")
#
#     # 🔍 update state dict for saving
#     if accelerator.is_main_process:
#         save_state_dict = {}
#         for state_name in sorted(list(state_dict.keys())):
#             for old_layer_id, new_layer_id in layer_id_mapping.items():
#                 if f"layers.{old_layer_id}." in state_name:  # convert old ids to new ones
#                     save_state_dict[state_name.replace(f"layers.{old_layer_id}", f"layers.{new_layer_id}")] = state_dict[state_name]
#                     accelerator.print(state_name, "-->", state_name.replace(f"layers.{old_layer_id}", f"layers.{new_layer_id}"))
#                     break
#                 elif f"layers." not in state_name:  # copy other states
#                     save_state_dict[state_name] = state_dict[state_name]
#                     accelerator.print(state_name, "-->", state_name)
#                     break
#
#         accelerator.print("Keys in save_state_dict:")
#         for key in save_state_dict.keys():
#             accelerator.print(key)
#
#         # 🔍 initialize a new model and save
#         accelerator.print("Initializing the new model...")
#         new_config = deepcopy(model.config)
#         new_config.num_hidden_layers = len(layer_id_mapping)
#         accelerator.print(new_config)
#         new_model = type(model)(new_config)
#         new_model.load_state_dict(save_state_dict, strict=True)
#         new_model.bfloat16()
#         accelerator.print("new_model", new_model)
#         accelerator.print("Saving...")
#         new_model.save_pretrained(prune_model_save_path)
#         tokenizer.save_pretrained(prune_model_save_path)
#
#     accelerator.wait_for_everyone()
#     accelerator.print(f"Model saved to {prune_model_save_path}")


# @torch.no_grad()
# def layer_pruning(args: Namespace, model: MixtralForCausalLM, dataloader: DataLoader, accelerator: Accelerator, num_samples: int):
#     cache_file = args.similarity_cache_file
#
#
#
#     device = accelerator.device
#     unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
#     use_cache = unwrapped_model.config.use_cache
#     unwrapped_model.config.use_cache = False
#     num_local_experts = unwrapped_model.config.num_local_experts
#     layers = unwrapped_model.model.layers
#
#
#     accelerator.print("Getting features...")
#     inputs, outputs, attention_mask, position_ids = prepare_calibration_input(unwrapped_model, dataloader, num_samples)  # 🔍
#     new_subset = {}
#     wrapped_layers = {}
#
#     # Wrap layers
#     accelerator.print('Starting ...')
#     for i in tqdm(range(len(layers)), desc="Dropping layers...", disable=not accelerator.is_main_process):
#         sys.stderr.flush()
#         torch.cuda.empty_cache()
#         print_gpu_memory(accelerator)
#         layer = layers[i]
#         subset = find_modules(layer, [MixtralSparseMoeBlock])  # 🔍
#
#         # Wrap MixtralSparseMoeBlock
#         name = "block_sparse_moe"
#         module_state_dict_name = f"model.layers.{i}.{name}"  # name = "block_sparse_moe"
#         wrapped_layers[module_state_dict_name] = MixtralLayerDropWrapper(subset[name])  # 🔍
#         new_subset[module_state_dict_name] = subset[name]
#
#         # Forward hook for recording metrics
#         def add_batch(name):
#             def hook(_, input, output):
#                 wrapped_layers[name].add_batch(input[0].data, output[0].data)  # output[1] is router_logits (before softmax)
#
#             return hook
#
#         # Get importance
#         handles = []
#         for name in wrapped_layers:
#             handles.append(new_subset[name].register_forward_hook(add_batch(name)))
#         for j in range(num_samples):
#             outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
#         for h in handles:
#             h.remove()
#
#         inputs, outputs = outputs, inputs
#
#     print_gpu_memory(accelerator)
#     torch.cuda.empty_cache()
#
#     # 🔍 Expert Drop
#     global_scores = []
#     update_num_local_experts_list = []
#     for module_state_dict_name in wrapped_layers:
#         accelerator.print(f"Dropping for {module_state_dict_name}")
#
#         # 🔍 sort total scores
#         # [IMPORTANT] all reduce across devices
#         hidden_states = torch.cat(wrapped_layers[module_state_dict_name].hidden_states, dim=0).to(device)
#         output_hidden_states = torch.cat(wrapped_layers[module_state_dict_name].output_hidden_states, dim=0).to(device)
#         cos_sim = F.cosine_similarity(hidden_states, output_hidden_states, dim=-1)  # (total_token_num)
#         cos_sim = cos_sim.mean()
#         # cos_sim = accelerator.reduce(cos_sim, reduction="mean")  # 🔍 All reduce across devices
#         # scores = wrapped_layers[module_state_dict_name].scores
#         accelerator.print(f"cos_sim: {cos_sim}")
#         # accelerator.print(f"scores: {scores.size()}")
#         # scores = accelerator.reduce(scores, reduction="sum")  # Here we use "sum" as the number of tokens processed by each device may be different.
#         # global_scores = torch.cat((global_scores, cos_sim), dim=0) if global_scores is not None else cos_sim  # 🔍 gather the scores.
#         global_scores.append(cos_sim.data)
#
#     accelerator.print(f"global_scores: {global_scores}")
#     if cache_file is not None:
#         if accelerator.is_main_process:
#             create_dir(os.path.dirname(cache_file))
#             torch.save(global_scores.clone().cpu(), cache_file)
#             print(f"Saving cached similarities to {cache_file}")
#         accelerator.wait_for_everyone()
#
#     # sorted_scores = sorted(global_scores)
#     _, layers_to_drop = torch.topk(torch.tensor(global_scores), int(args.sparsity_ratio * len(layers)), largest=False)
#     # accelerator.print(f"global_scores: {global_scores.size()}")
#
#     layers_to_drop = sorted(layers_to_drop.tolist())
#     for layer_id, layer in tqdm(list(enumerate(layers)), desc='Dropping Experts...'):
#         experts = num_local_experts if layer_id not in layers_to_drop else 0
#         update_num_local_experts_list.append(experts)
#
#     unwrapped_model.config.layer_experts_idx = update_num_local_experts_list
#     accelerator.print(f"layers_to_drop: {layers_to_drop}")
#     accelerator.print("Layer dropping done!")
#     unwrapped_model.config.use_cache = use_cache
#     torch.cuda.empty_cache()
