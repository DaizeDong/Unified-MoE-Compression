import logging
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from argparse import Namespace
from torch.utils.data import DataLoader
from tqdm import tqdm
from global_utils.io import create_dir

from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralSparseMoeBlock
from .utils import print_gpu_memory, prepare_calibration_input, find_moe_expert_linears_and_gate, find_modules
from .wrapper import PrunableMixtralSparseMoeLayerWrapper, WeightRecordWrapper

logger = logging.getLogger(__name__)


# 🔍 The final attempt that strictly follows the pruning pipeline
# Finally, the whole shit has been done. THANK GOD!!!!!!!!!!!!!!!!!!!!!


@torch.no_grad()
def layer_pruning(args: Namespace, model: MixtralForCausalLM, dataloader: DataLoader,
                   accelerator: Accelerator, num_samples: int, num_local_experts: int):

    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    num_local_experts = unwrapped_model.config.num_local_experts
    layers = unwrapped_model.model.layers
    cache_file = args.similarity_cache_file
    # 🔍 store the pruned parameters in CPU
    # update_state_dict = {}

    accelerator.print("Getting features...")
    inputs, outputs, attention_mask, position_ids = prepare_calibration_input(unwrapped_model, dataloader, num_samples)  # 🔍
    new_subset = {}
    # new_captured_weights_subset = {}
    wrapped_layers = {}
    # captured_weights_wrapped_layers = {}

    # Wrap layers
    accelerator.print('Starting ...')
    for i in tqdm(range(len(layers)), desc="Dropping layers...", disable=not accelerator.is_main_process):
        sys.stderr.flush()
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)
        layer = layers[i]
        subset = find_modules(layer, [MixtralSparseMoeBlock])  # 🔍
        
        # Wrap MixtralSparseMoeBlock
        # for name in subset:
        name = "block_sparse_moe"
        module_state_dict_name = f"model.layers.{i}.{name}"  # name = "block_sparse_moe"
        wrapped_layers[module_state_dict_name] = PrunableMixtralSparseMoeLayerWrapper(subset[name])  # 🔍
        new_subset[module_state_dict_name] = subset[name]

        # Forward hook for recording metrics
        def add_batch(name):
            def hook(_, input, output):
                wrapped_layers[name].add_batch(input[0].data, output[0].data)  # output[1] is router_logits (before softmax)

            return hook

        # def record_weight(name):  # 👆
        #     def hook(_, input, output):
        #         captured_weights_wrapped_layers[name].record(input, output)
        #     return hook

        # Get importance
        handles = []
        for name in wrapped_layers:
            handles.append(new_subset[name].register_forward_hook(add_batch(name)))
        # for name in captured_weights_wrapped_layers:  # 👆
        #     handles.append(new_captured_weights_subset[name].register_forward_hook(record_weight(name)))
        for j in range(num_samples):
            outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
        for h in handles:
            h.remove()

        inputs, outputs = outputs, inputs

    print_gpu_memory(accelerator)
    torch.cuda.empty_cache()

    # 🔍 Expert Drop
    global_scores = []
    update_num_local_experts_list = []
    # for layer_id, layer in tqdm(list(enumerate(layers)), desc='Dropping Experts...'):
    # module_state_dict_name = f"model.layers.{l}.block_sparse_moe"
    for module_state_dict_name in wrapped_layers:
        # module_state_dict_name = f"model.layers.{layer_id}.{name}" # name = "block_sparse_moe"
        accelerator.print(f"Dropping for {module_state_dict_name}")

        # 🔍 sort total scores
        # [IMPORTANT] all reduce across devices
        hidden_states = torch.cat(wrapped_layers[module_state_dict_name].hidden_states, dim=0).to(device)
        output_hidden_states = torch.cat(wrapped_layers[module_state_dict_name].output_hidden_states, dim=0).to(device)
        cos_sim = F.cosine_similarity(hidden_states, output_hidden_states, dim=-1)  # (total_token_num)
        cos_sim = cos_sim.mean()
        # cos_sim = accelerator.reduce(cos_sim, reduction="mean")  # 🔍 All reduce across devices
        # scores = wrapped_layers[module_state_dict_name].scores
        accelerator.print(f"cos_sim: {cos_sim}")
        # accelerator.print(f"scores: {scores.size()}")
        # scores = accelerator.reduce(scores, reduction="sum")  # Here we use "sum" as the number of tokens processed by each device may be different.
        # global_scores = torch.cat((global_scores, cos_sim), dim=0) if global_scores is not None else cos_sim  # 🔍 gather the scores.
        global_scores.append(cos_sim.data)
        
    accelerator.print(f"global_scores: {global_scores}")
    if cache_file is not None:
        if accelerator.is_main_process:
            create_dir(os.path.dirname(cache_file))
            torch.save(global_scores.clone().cpu(), cache_file)
            print(f"Saving cached similarities to {cache_file}")
        accelerator.wait_for_everyone()
    
    # sorted_scores = sorted(global_scores)
    _, layers_to_drop = torch.topk(torch.tensor(global_scores), int(args.sparsity_ratio * len(layers)), largest=False)
    # accelerator.print(f"global_scores: {global_scores.size()}")

    layers_to_drop = sorted(layers_to_drop.tolist())
    for layer_id, layer in tqdm(list(enumerate(layers)), desc='Dropping Experts...'):
        experts = num_local_experts if layer_id not in layers_to_drop else 0
        update_num_local_experts_list.append(experts)
        
    unwrapped_model.config.layer_experts_idx = update_num_local_experts_list
    accelerator.print(f"layers_to_drop: {layers_to_drop}")
    accelerator.print("Layer dropping done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def post_layers_drop(model, layer_experts_idx, accelerator):
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    layers = unwrapped_model.model.layers
    
    for layer_id, layer in tqdm(list(enumerate(layers)), desc='Dropping Experts...'):
        experts = layer_experts_idx[layer_id]
        if experts == 0:
            experts_to_preserve = []
            new_gate_weight = layer.block_sparse_moe.gate.weight.data[list(experts_to_preserve)]
            layer.block_sparse_moe.gate = torch.nn.Linear(in_features=layer.block_sparse_moe.gate.in_features, out_features=len(experts_to_preserve), bias=False, device=layer.block_sparse_moe.gate.weight.device, dtype=torch.bfloat16)
            layer.block_sparse_moe.gate.weight.data = new_gate_weight
            # 🔍 drop experts.
            layer.block_sparse_moe.experts = torch.nn.ModuleList([layer.block_sparse_moe.experts[i] for i in experts_to_preserve])
            layer.num_experts = 0

    unwrapped_model.config.num_local_experts = layer_experts_idx

