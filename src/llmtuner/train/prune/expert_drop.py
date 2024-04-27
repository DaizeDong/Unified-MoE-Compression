import logging
import numpy as np
import sys
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from argparse import Namespace
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
from .utils import find_moe_experts, print_gpu_memory, prepare_calibration_input, find_moe_expert_linears_and_gate
from .wrapper import PrunableMixtralSparseMoeBlockWrapper, WeightRecordWrapper

logger = logging.getLogger(__name__)


# @torch.no_grad()
# def layerwise_pruning(args, model, calib_loader, accelerator: Accelerator, num_samples: int):
#     accelerator.print(f"model: {type(model)}")

#     device = accelerator.device
#     unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
#     use_cache = unwrapped_model.config.use_cache
#     unwrapped_model.config.use_cache = False
#     layers = unwrapped_model.model.layers

#     # 🔍 store the pruned parameters in CPU
#     update_state_dict = {}

#     for i, batch in enumerate(tqdm(calib_loader, desc='Model forwarding on sample set...', disable=not accelerator.is_main_process)):
#         if i > num_samples:
#             break
#         accelerator.print(f"batch: {batch.keys()} {batch}")
#         model_inputs = model.prepare_inputs_for_generation(**batch)
#         accelerator.print(f"model_inputs: {model_inputs.keys()} {model_inputs}")
#         # outputs = 
#         model(**model_inputs)
#         # assert outputs is not None

#     torch.cuda.empty_cache()

#     # Find the optimal expert combination
#     global_loss_history = dict()
#     for l, layer in tqdm(list(enumerate(layers)), desc='Enumerating loss on sample set...', disable=not accelerator.is_main_process):
#         with FSDP.summon_full_params(model):  # 🔍
#             layer.to(device)
#             accelerator.print(layer)
#             moe_module = layer.block_sparse_moe
#             if not hasattr(moe_module, 'cache_space'):
#                 continue

#             # Get the L2 loss on outputs for every possible combination of expert drop
#             # loss_history: dict = moe_module.enumerate()
#             # for key, value in loss_history.items():  # 🔍 all reduce across devices
#             #     loss_history[key] = accelerator.reduce(value, reduction="sum")  # Here we use "sum" as the number of tokens processed by each device may be different.

#             # Get the optimal expert drop combination
#             # moe_module.update_dropped_experts(loss_history)
#             # global_loss_history[l] = loss_history

#             # Update the state_dict
#             moe_module.prune(update_state_dict, layer_id=l)
#             torch.cuda.empty_cache()
#     accelerator.print(global_loss_history)

#     return update_state_dict

# @torch.no_grad()
# def layerwise_pruning(args: Namespace, model: MixtralForCausalLM, model_intact, calib_loader: DataLoader,
#                       accelerator: Accelerator, num_samples: int, num_local_experts: int):
#     # device = accelerator.device
#     unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
#     unwrapped_model.config.use_cache = False
#     layers = unwrapped_model.model.layers
#
#     # 🔍 store the pruned parameters in CPU
#     update_state_dict = {}
#
#     # Wrap layers
#     for layer_id, layer in enumerate(layers):
#         subset = find_moe_experts(layer)  # MixtralSparseMoeBlock
#         # accelerator.print(subset)
#         for name, module in subset.items():  # set all modules in the model as wrapped_modules
#             wrapped_module = PrunableMixtralSparseMoeBlockWrapper(module, r=args.r)
#             wrapped_module.cache_X = True
#             wrapped_module.cache_Z = True
#             setattr(layer, name, wrapped_module)
#
#     # Forward to record scores
#     for i, batch in tqdm(enumerate(calib_loader), desc='Model forwarding on sample set...'):
#         if i > num_samples:
#             break
#         model_inputs = model.prepare_inputs_for_generation(**batch)
#         model(**model_inputs)
#
#     print_gpu_memory(accelerator)
#     torch.cuda.empty_cache()
#
#     # Expert Drop
#     for layer_id, layer in tqdm(list(enumerate(layers)), desc='Dropping Experts...'):
#         # module_state_dict_name = f"model.layers.{l}.block_sparse_moe"
#         wrapped_module = layer.block_sparse_moe
#
#         if hasattr(wrapped_module, 'cache_space'):
#             # 🔍 [IMPORTANT] all reduce across devices
#             wrapped_module.cache_space.scores = accelerator.reduce(wrapped_module.cache_space.scores, reduction="sum")  # Here we use "sum" as the number of tokens processed by each device may be different.
#             wrapped_module.enumerate()
#             update_state_dict = wrapped_module.prune(update_state_dict, model_intact, layer_id)
#
#         layer.block_sparse_moe = wrapped_module.layer
#         accelerator.print(f"layer {layer_id}: {layer.block_sparse_moe}")
#
#     accelerator.print(update_state_dict)
#
#     return update_state_dict


# 🔍 The final attempt that strictly follows the pruning pipeline
# Finally, the whole shit has been done. THANK GOD!!!!!!!!!!!!!!!!!!!!!
@torch.no_grad()
def layerwise_pruning(args: Namespace, model: MixtralForCausalLM, dataloader: DataLoader, accelerator: Accelerator, num_samples: int, num_local_experts: int):
    """
    :param num_samples: samples on each device, calculated as "num_samples = n_calibration_samples // num_processes"
    """
    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    num_local_experts = unwrapped_model.config.num_local_experts
    layers = unwrapped_model.model.layers

    # 🔍 store the pruned parameters in CPU
    update_state_dict = {}

    accelerator.print("Getting features...")
    inputs, outputs, attention_mask, position_ids = prepare_calibration_input(unwrapped_model, dataloader, num_samples)  # 🔍

    accelerator.print('Starting ...')
    update_num_local_experts_list = []
    
    for i in tqdm(range(len(layers)), desc="Dropping layers...", disable=not accelerator.is_main_process):
        sys.stderr.flush()
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)
        layer = layers[i]
        subset = find_moe_experts(layer)  # 🔍 Find MixtralSparseMoeBlock
        captured_weights_subset = find_moe_expert_linears_and_gate(layer)  # 👆 Find weights to capture (here the gate & w1 & w2 & w3)
        # accelerator.print(subset)
        # accelerator.print(captured_weights_subset)

        # Wrap MixtralSparseMoeBlock
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = PrunableMixtralSparseMoeBlockWrapper(subset[name])  # 🔍

        # 👆 Wrap weights to record during forward
        # (WHY: DeepSpeed will temporarily collect intact weights to GPU during forward, so we can capture them using forward hooks)
        captured_weights_wrapped_layers = {}
        for name in captured_weights_subset:
            captured_weights_wrapped_layers[name] = WeightRecordWrapper(captured_weights_subset[name], layer_name=name)

        # Forward hook for recording metrics
        def add_batch(name):
            def hook(_, input, output):
                wrapped_layers[name].add_batch(input[0].data, output[1].data)  # output[1] is router_logits (before softmax)

            return hook

        def record_weight(name):  # 👆
            def hook(_, input, output):
                captured_weights_wrapped_layers[name].record(input, output)

            return hook

        # Get importance
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for name in captured_weights_wrapped_layers:  # 👆
            handles.append(captured_weights_subset[name].register_forward_hook(record_weight(name)))
        for j in range(num_samples):
            outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
        for h in handles:
            h.remove()

        # 🔍 Expert Drop
        for name in subset:
            module_state_dict_name = f"model.layers.{i}.{name}"
            accelerator.print(f"Dropping for {module_state_dict_name}")

            # 🔍 sort total scores
            # [IMPORTANT] all reduce across devices
            scores = wrapped_layers[name].scores
            scores = accelerator.reduce(scores, reduction="sum")  # Here we use "sum" as the number of tokens processed by each device may be different.

            _, experts_to_drop = torch.topk(scores, num_local_experts - args.r, largest=False)
            experts_to_drop = experts_to_drop.tolist()
            experts_to_preserve = sorted(list(set(range(num_local_experts)) - set(experts_to_drop)))
        
        accelerator.print(f"layer {i} scores: {scores}")
        accelerator.print(f"layer {i} experts_to_drop: {experts_to_drop}")
        update_num_local_experts_list.append(len(experts_to_preserve))

            # 🔍 update the state dict
            # 👆 get weights from the "captured_weights_wrapped_layers"
            # update_state_dict[f"{module_state_dict_name}.gate.weight"] = captured_weights_wrapped_layers[f"{name}.gate"].weight[list(experts_to_preserve)]
            # for new_expert_id, old_expert_id in enumerate(experts_to_preserve):
            #     update_state_dict[f"{module_state_dict_name}.experts.{new_expert_id}.w1.weight"] = captured_weights_wrapped_layers[f"{name}.experts.{old_expert_id}.w1"].weight
            #     update_state_dict[f"{module_state_dict_name}.experts.{new_expert_id}.w2.weight"] = captured_weights_wrapped_layers[f"{name}.experts.{old_expert_id}.w2"].weight
            #     update_state_dict[f"{module_state_dict_name}.experts.{new_expert_id}.w3.weight"] = captured_weights_wrapped_layers[f"{name}.experts.{old_expert_id}.w3"].weight

            # update_state_dict[f"{module_state_dict_name}.gate.weight"] = wrapped_layers[name].gate[list(experts_to_preserve)].clone().bfloat16().cpu()
            # for new_expert_id, old_expert_id in enumerate(experts_to_preserve):
            #     update_state_dict[f"{module_state_dict_name}.experts.{new_expert_id}.w1.weight"] = wrapped_layers[name].w1.clone().bfloat16().cpu()
            #     update_state_dict[f"{module_state_dict_name}.experts.{new_expert_id}.w2.weight"] = wrapped_layers[name].w2.clone().bfloat16().cpu()
            #     update_state_dict[f"{module_state_dict_name}.experts.{new_expert_id}.w3.weight"] = wrapped_layers[name].w3.clone().bfloat16().cpu()

        # Update inputs & outputs
        inputs, outputs = outputs, inputs

    accelerator.print("Expert dropping done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    setattr(accelerator.unwrap_model(model).config, "num_local_experts", args.r)
    unwrapped_model.config.num_local_experts = args.r

    # for name, weight in update_state_dict.items():
    #     accelerator.print(name, weight.shape)

    # 🔍 return the state dict
    return update_state_dict


@torch.no_grad()
def global_pruning(args: Namespace, model: MixtralForCausalLM, dataloader: DataLoader,
                   accelerator: Accelerator, num_samples: int, num_local_experts: int):
    # device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    num_local_experts = unwrapped_model.config.num_local_experts
    layers = unwrapped_model.model.layers

    # 🔍 store the pruned parameters in CPU
    # update_state_dict = {}

    accelerator.print("Getting features...")
    inputs, outputs, attention_mask, position_ids = prepare_calibration_input(unwrapped_model, dataloader, num_samples)  # 🔍
    new_subset = {}
    new_captured_weights_subset = {}
    wrapped_layers = {}
    captured_weights_wrapped_layers = {}

    # Wrap layers
    accelerator.print('Starting ...')
    for i in tqdm(range(len(layers)), desc="Dropping layers...", disable=not accelerator.is_main_process):
        sys.stderr.flush()
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)
        layer = layers[i]
        subset = find_moe_experts(layer)  # 🔍 Find MixtralSparseMoeBlock

        captured_weights_subset = find_moe_expert_linears_and_gate(layer)  # 👆 Find weights to capture (here the gate & w1 & w2 & w3)
        # accelerator.print(captured_weights_subset)

        # Wrap MixtralSparseMoeBlock
        for name in subset:
            module_state_dict_name = f"model.layers.{i}.{name}"  # name = "block_sparse_moe"
            wrapped_layers[module_state_dict_name] = PrunableMixtralSparseMoeBlockWrapper(subset[name])  # 🔍
            new_subset[module_state_dict_name] = subset[name]

        # accelerator.print(f"new_subset: {new_subset.keys()}")
        # accelerator.print(f"warpped_layers: {wrapped_layers.keys()}")

        # 👆 Wrap weights to record during forward
        # (WHY: DeepSpeed will temporarily collect intact weights to GPU during forward, so we can capture them using forward hooks)
        for name in captured_weights_subset:
            module_state_dict_name = f"model.layers.{i}.{name}"  # name = "block_sparse_moe"
            captured_weights_wrapped_layers[module_state_dict_name] = WeightRecordWrapper(captured_weights_subset[name], layer_name=name)
            new_captured_weights_subset[module_state_dict_name] = captured_weights_subset[name]

        # Forward hook for recording metrics
        def add_batch(name):
            def hook(_, input, output):
                wrapped_layers[name].add_batch(input[0].data, output[1].data)  # output[1] is router_logits (before softmax)

            return hook

        def record_weight(name):  # 👆
            def hook(_, input, output):
                captured_weights_wrapped_layers[name].record(input, output)

            return hook

        # Get importance
        handles = []
        for name in wrapped_layers:
            handles.append(new_subset[name].register_forward_hook(add_batch(name)))
        for name in captured_weights_wrapped_layers:  # 👆
            handles.append(new_captured_weights_subset[name].register_forward_hook(record_weight(name)))
        for j in range(num_samples):
            outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
        for h in handles:
            h.remove()

        inputs, outputs = outputs, inputs

    # for layer_id, layer in enumerate(layers):
    #     subset = find_moe_experts(layer)  # MixtralSparseMoeBlock
    #     # accelerator.print(subset)
    #     for name, module in subset.items():  # set all modules in the model as wrapped_modules
    #         wrapped_module = PrunableMixtralSparseMoeBlockWrapper(module)
    #         wrapped_module.cache_X = True
    #         wrapped_module.cache_Z = True
    #         setattr(layer, name, wrapped_module)

    # Forward to record scores
    # for i, batch in tqdm(enumerate(calib_loader), desc='Model forwarding on sample set...'):
    #     if i > num_samples:
    #         break
    #     model_inputs = model.prepare_inputs_for_generation(**batch)
    #     model(**model_inputs)

    print_gpu_memory(accelerator)
    torch.cuda.empty_cache()

    # 🔍 Expert Drop
    global_scores = None
    # for layer_id, layer in tqdm(list(enumerate(layers)), desc='Dropping Experts...'):
    # module_state_dict_name = f"model.layers.{l}.block_sparse_moe"
    for module_state_dict_name in wrapped_layers:
        # module_state_dict_name = f"model.layers.{layer_id}.{name}" # name = "block_sparse_moe"
        accelerator.print(f"Dropping for {module_state_dict_name}")

        # 🔍 sort total scores
        # [IMPORTANT] all reduce across devices
        scores = wrapped_layers[module_state_dict_name].scores
        scores = accelerator.reduce(scores, reduction="sum")  # Here we use "sum" as the number of tokens processed by each device may be different.

        # if hasattr(wrapped_module, 'cache_space'):
        # 🔍 [IMPORTANT] all reduce across devices
        # wrapped_module.cache_space.scores = accelerator.reduce(wrapped_module.cache_space.scores, reduction="sum")  # Here we use "sum" as the number of tokens processed by each device may be different.
        # wrapped_module.enumerate()
        # update_state_dict = wrapped_module.prune(update_state_dict, layer_id)

        # layer.block_sparse_moe = wrapped_module.layer
        # accelerator.print(f"layer {layer_id}: {layer.block_sparse_moe}")
        global_scores = torch.cat((global_scores, scores), dim=0) if global_scores is not None else scores  # 🔍 gather the scores.

    accelerator.print(f"global_scores: {global_scores}")
    _, experts_to_drop = torch.topk(global_scores, (num_local_experts - args.r) * len(layers), largest=False)
    experts_to_drop = sorted(experts_to_drop.tolist())
    # accelerator.print(f"experts_to_drop: {experts_to_drop}")
    # accelerator.print(f"captured_weights_wrapped_layers: {captured_weights_wrapped_layers}")
    # experts_to_drop = list(int(i) for i in experts_to_drop.data)

    update_num_local_experts_list = []  # 🔍
    layer_experts_idx = []
    for layer_id, layer in tqdm(list(enumerate(layers)), desc='Dropping Experts...'):
        # accelerator.print(f"sorted(list(set(range(num_local_experts) * (layer_id + 1)): {sorted(list(set(range(num_local_experts * layer_id, num_local_experts))))}")
        experts_to_preserve = sorted(list(set(range(num_local_experts * layer_id, num_local_experts * (layer_id + 1))) - set(experts_to_drop)))
        experts_to_preserve = [i - layer_id * num_local_experts for i in experts_to_preserve]
        accelerator.print(f"layer {layer_id} experts_to_preserve: {experts_to_preserve}")
        # module_state_dict_name = f"model.layers.{layer_id}.block_sparse_moe"  # name = "block_sparse_moe"
        # update_state_dict[f"{module_state_dict_name}.gate.weight"] = captured_weights_wrapped_layers[f"{module_state_dict_name}.gate"].weight[list(experts_to_preserve)]
        # for new_expert_id, old_expert_id in enumerate(experts_to_preserve):
        #     update_state_dict[f"{module_state_dict_name}.experts.{new_expert_id}.w1.weight"] = captured_weights_wrapped_layers[f"{module_state_dict_name}.experts.{old_expert_id}.w1"].weight
        #     update_state_dict[f"{module_state_dict_name}.experts.{new_expert_id}.w2.weight"] = captured_weights_wrapped_layers[f"{module_state_dict_name}.experts.{old_expert_id}.w2"].weight
        #     update_state_dict[f"{module_state_dict_name}.experts.{new_expert_id}.w3.weight"] = captured_weights_wrapped_layers[f"{module_state_dict_name}.experts.{old_expert_id}.w3"].weight

        update_num_local_experts_list.append(len(experts_to_preserve))
        layer_experts_idx.append(experts_to_preserve)
        
        # wrapped_module = layer.block_sparse_moe
        # 🔍 select the experts for each layer. 
        # wrapped_module.experts_to_drop = [i for i in experts_to_drop if i >= layer_id * num_local_experts and i < (layer_id + 1) * num_local_experts]
        # update_state_dict = wrapped_module.prune(update_state_dict, layer_id)
        # layer.block_sparse_moe = wrapped_module.layer

    unwrapped_model.config.num_local_experts = update_num_local_experts_list
    unwrapped_model.config.layer_experts_idx = layer_experts_idx
    accelerator.print("Expert dropping done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    # for name, weight in update_state_dict.items():
    #     accelerator.print(name, weight.shape)

    # 🔍 return the idx of remaining experts
    # return update_num_local_experts_list
    # 🔍 return the state dict
    # return update_state_dict


@torch.no_grad()
def progressive_pruning(model: MixtralForCausalLM, calib_loader: DataLoader, args: Namespace):
    assert isinstance(
        model, MixtralForCausalLM), 'Currently only `Mixtral` is supported'

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = PrunableMixtralSparseMoeBlockWrapper(
            layer.block_sparse_moe, r=args.r)
        layer.block_sparse_moe.cache_Z = True

    for i, batch in enumerate(tqdm(calib_loader, desc='Computing Z activations on sample set...')):
        model_inputs = model.prepare_inputs_for_generation(**batch)
        outputs = model(**model_inputs)
        assert outputs is not None

    del model_inputs
    del outputs
    torch.cuda.empty_cache()

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe.cache_Z = False

    # Drop
    global_loss_history = dict()

    for l, layer in tqdm(list(enumerate(model.model.layers)), desc='Dropping layers...'):
        b = layer.block_sparse_moe

        b.cache_X = True
        with torch.inference_mode():
            for i, batch in enumerate(calib_loader):
                print(f"batch.keys(): {batch.keys()}")
                model_inputs = model.prepare_inputs_for_generation(**batch)
                outputs = model(**model_inputs)
                assert outputs is not None

        del model_inputs
        del outputs
        torch.cuda.empty_cache()
        b.cache_X = False

        loss_history = b.enumerate()
        global_loss_history[l] = loss_history

        b.prune()
        layer.block_sparse_moe = b.model

    # Prune & save
    model.num_experts = args.r
    # model.config.num_local_experts = args.r


@torch.no_grad()
def dynamic_skipping(model: MixtralForCausalLM, calib_loader: DataLoader, args: Namespace):
    assert isinstance(
        model, MixtralForCausalLM), 'Currently only `Mixtral` is supported'

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = PrunableMixtralSparseMoeBlockWrapper(
            layer.block_sparse_moe)
        layer.block_sparse_moe.cache_logits = True
        layer.block_sparse_moe.cache_X = True
        layer.block_sparse_moe.cache_Z = True

    for i, batch in enumerate(tqdm(calib_loader, desc='Model forwarding on sample set...')):
        model_inputs = model.prepare_inputs_for_generation(**batch)
        outputs = model(**model_inputs)
        assert outputs is not None

    res_median = {}
    res_mean = {}

    for layer_idx in range(len(model.model.layers)):
        b = model.model.layers[layer_idx].block_sparse_moe
        b.cache_space.prepare_for_loader()
        dataloader = torch.utils.data.DataLoader(
            b.cache_space,
            batch_size=args.batch_size,
            shuffle=True,
        )
        logger.info(len(dataloader))

        ana_list = []
        for i, (router_logits, X, Z) in enumerate(dataloader):
            routing_weights = F.softmax(
                router_logits, dim=-1, dtype=torch.float).view(-1, b.model.num_experts)
            for j in range(len(routing_weights)):
                sorted_weights, sort_indices = torch.sort(
                    routing_weights[j], descending=True)
                ana_list.append(float(sorted_weights[1] / sorted_weights[0]))

        median = np.median(ana_list)
        mean = np.mean(ana_list)
        logger.info(f'layer {layer_idx} | mean: {mean}, median: {median}')
        res_median[str(layer_idx)] = median
        res_mean[str(layer_idx)] = mean

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = layer.block_sparse_moe.model

    model.config.betas = res_median
    return model, (res_median, res_mean)


def post_experts_drop(model, layer_experts_idx, accelerator):
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    layers = unwrapped_model.model.layers
    for layer_id, layer in tqdm(list(enumerate(layers)), desc='Dropping Experts...'):

        experts_to_preserve = layer_experts_idx[layer_id]
        accelerator.print(f"experts_to_preserve: {experts_to_preserve}")
        r = len(experts_to_preserve)
        # accelerator.print(f"layer.block_sparse_moe.gate.weight.data, {layer.block_sparse_moe.gate.weight.data}")
        # 🔍 rewrite gate. 
        new_gate_weight = layer.block_sparse_moe.gate.weight.data[list(experts_to_preserve)]
        layer.block_sparse_moe.gate = torch.nn.Linear(in_features=layer.block_sparse_moe.gate.in_features, out_features=r, bias=False, device=layer.block_sparse_moe.gate.weight.device, dtype=torch.bfloat16)
        layer.block_sparse_moe.gate.weight.data = new_gate_weight
        # 🔍 drop experts.
        layer.block_sparse_moe.experts = torch.nn.ModuleList([layer.block_sparse_moe.experts[i] for i in experts_to_preserve])
        layer.num_experts = r