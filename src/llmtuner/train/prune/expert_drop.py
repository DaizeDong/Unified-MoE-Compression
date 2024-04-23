import logging
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from argparse import Namespace
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
from .wrapper import PrunableMixtralSparseMoeBlockWrapper

logger = logging.getLogger(__name__)


@torch.no_grad()
def layerwise_pruning(args, model, calib_loader, accelerator: Accelerator, num_samples: int):
    accelerator.print(f"model: {type(model)}")

    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    # 🔍 store the pruned parameters in CPU
    update_state_dict = {}

    for i, batch in enumerate(tqdm(calib_loader, desc='Model forwarding on sample set...'), disable=not accelerator.is_main_process):
        if i > num_samples:
            break
        accelerator.print(f"batch: {batch.keys()} {batch}")
        model_inputs = model.prepare_inputs_for_generation(**batch)
        accelerator.print(f"model_inputs: {model_inputs.keys()} {model_inputs}")
        outputs = model(**model_inputs)
        assert outputs is not None

    torch.cuda.empty_cache()

    # Find the optimal expert combination
    global_loss_history = dict()
    for l, layer in tqdm(list(enumerate(layers)), desc='Enumerating loss on sample set...', disable=not accelerator.is_main_process):
        moe_module = layer.block_sparse_moe
        if not hasattr(moe_module, 'cache_space'):
            continue

        # Get the L2 loss on outputs for every possible combination of expert drop
        loss_history: dict = moe_module.enumerate()
        for key, value in loss_history.items():  # 🔍 all reduce across devices
            loss_history[key] = accelerator.reduce(value, reduction="sum")  # Here we use "sum" as the number of tokens processed by each device may be different.

        # Get the optimal expert drop combination
        moe_module.update_dropped_experts(loss_history)
        global_loss_history[l] = loss_history

        # Update the state_dict
        moe_module.prune(update_state_dict, layer_id=l)
        torch.cuda.empty_cache()
    accelerator.print(global_loss_history)

    # 🔍 change the config
    update_config = deepcopy(unwrapped_model.config)
    update_config.num_local_experts = args.r

    return update_config, update_state_dict


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
    model.config.num_local_experts = args.r

    return model, (global_loss_history,)


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


Expert_Drop_Methods = {
    'layerwise_pruning': layerwise_pruning,
    'progressive_pruning': progressive_pruning,
    'dynamic_skipping': dynamic_skipping,
}
