import sys
import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator
from tqdm import tqdm

print("transformers", transformers)

from transformers.models.mixtral.modeling_mixtral import Expert
from .ablate import AblateGPT
from .data import get_loaders
from .info import print_gpu_memory
from .wrapper import WandaWrapper, SparseGPTWrapper


def find_layers(module, layers=[nn.Linear, Expert], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    # print(name, type(module), type(module) in layers)
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def find_layers_for_moe(module, layers=[nn.Linear, Expert], name=''):
    # 🔍 exclude the gate networks for MoE
    res = find_layers(module, layers, name)
    for key in list(res.keys()):
        if "gate" in key:
            res.pop(key)
        # if 'experts' in key:
        #     res.pop(key)
    return res


@torch.no_grad()
def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params


@torch.no_grad()
def check_sparsity_from_state_dict(state_dict):
    """
    🔍 This function has been rewritten to calculate sparsity from "state_dict".
    """
    # Get corresponding names for each layer
    layer_params = {}
    for name in sorted(list(state_dict.keys())):
        # Example: model.layers.5.block_sparse_moe.experts.2.w3.weight
        if "layers" in name:
            layer_id = int(name.split(".")[2])
            if layer_id not in layer_params:
                layer_params[layer_id] = [name]
            else:
                layer_params[layer_id].append(name)

    layer_num = max(list(layer_params.keys())) + 1

    # Calculate sparsity
    count = 0
    total_params = 0
    for i in range(layer_num):
        sub_count = 0
        sub_params = 0
        for name in layer_params[i]:
            count += (state_dict[name] == 0).sum().item()
            total_params += state_dict[name].numel()

            sub_count += (state_dict[name] == 0).sum().item()
            sub_params += state_dict[name].numel()

        print(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")

    return float(count) / total_params


@torch.no_grad()
def prepare_calibration_input(model, dataloader, accelerator: Accelerator, num_samples=16):
    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍

    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers
    cache = {'inputs': [], 'attention_mask': [], "position_ids": []}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, input, **kwargs):
            # print(input.shape)
            cache['inputs'].append(input)
            cache['attention_mask'].append(kwargs['attention_mask'])
            cache['position_ids'].append(kwargs['position_ids'])
            raise ValueError

    layers[0] = Catcher(layers[0])
    for index, batch in enumerate(dataloader):
        if index >= num_samples:  # 🔍 limit the number of samples in each device, batch_size must be 1
            break
        try:
            model(**batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    outputs = [None] * len(cache['inputs'])
    unwrapped_model.config.use_cache = use_cache

    return cache['inputs'], outputs, cache['attention_mask'], cache['position_ids']


@torch.no_grad()
def prune_magnitude(args, model, accelerator, prune_n=0, prune_m=0):
    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    # 🔍 store the pruned parameters in CPU
    update_state_dict = {}

    print('Starting ...')
    for i in tqdm(range(len(layers)), desc="Pruning layers..."):
        sys.stderr.flush()
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)
        layer = layers[i]
        subset = find_layers_for_moe(layer)  # 🔍 Find layers to prune

        # Prune
        layer.to(device)  # 🔍

        for name in subset:
            module_state_dict_name = f"model.layers.{i}.{name}"
            print(f"Pruning module {module_state_dict_name}")
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            # print(f"W_metric: {W_metric}")
            if prune_n != 0:
                W_mask = (torch.zeros_like(W) == 1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten())[0][int(W.numel() * args.sparsity_ratio)]
                W_mask = (W_metric <= thresh)

            # 🔍 update the state dict
            # 🔍 the weights would not change if directly updating them using "W[W_mask] = 0"
            update_state_dict[module_state_dict_name + ".weight"] = (W * W_mask).bfloat16().cpu()

        layer.to("cpu")  # 🔍

    print("Pruning done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    # 🔍 return the state dict
    return update_state_dict


@torch.no_grad()
def prune_wanda(args, model, dataloader, accelerator: Accelerator, num_samples, prune_n=0, prune_m=0):
    """
    :param num_samples: samples on each device, calculated as "num_samples = n_calibration_samples // num_processes"
    """
    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    # 🔍 store the pruned parameters in CPU
    update_state_dict = {}

    accelerator.print("Getting features...")
    inputs, outputs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, accelerator, num_samples)  # 🔍

    accelerator.print('Starting ...')
    for i in tqdm(range(len(layers)), desc="Pruning layers...", disable=not accelerator.is_main_process):
        sys.stderr.flush()
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)
        layer = layers[i]
        subset = find_layers_for_moe(layer)  # 🔍 Find layers to prune
        separate = True
        # separate = False

        if separate:
            experts_subset = [name for name in subset if "experts" in name]
            vanilla_subset = [name for name in subset if name not in experts_subset]
            w1 = [name for name in experts_subset if "w1" in name]
            w2 = [name for name in experts_subset if "w2" in name]
            w3 = [name for name in experts_subset if "w3" in name]

        else:
            vanilla_subset = subset
            w1 = []
            w2 = []
            w3 = []

        accelerator.print(f"w1, w2, w3: {w1, w2, w3}")
        accelerator.print(f"subset: {subset}")

        # Wrap layers
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WandaWrapper(subset[name], layer_name=name)

        # Forward hook for recording row importance
        def add_batch(name):
            def hook(_, input, output):
                wrapped_layers[name].add_batch(input[0].data, output.data)

            def moe_hook(_, input, output):
                wrapped_layers[name].add_batch(input[0].data, output.data, input[1].data)  # 🔍 input[1] is routing scores.

            if 'experts' in name:
                return moe_hook
            else:
                return hook

                # accelerator.print(f'subset: {subset}')

        # Get importance
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(num_samples):
            outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
        for h in handles:
            h.remove()

        # 🔍 Prune non-moe weights
        for name in vanilla_subset:
            # for name in subset:
            module_state_dict_name = f"model.layers.{i}.{name}"
            accelerator.print(f"Pruning module {module_state_dict_name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            W_metric = accelerator.reduce(W_metric, reduction="sum")  # 🔍 all reduce across devices
            W_mask = torch.zeros_like(W_metric)  # initialize a mask to be all 0

            # accelerator.print(f"W_metric: {W_metric}")
            def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
                thres_cumsum = sum_before * alpha
                sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
                thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
                W_mask = (W_metric <= thres)
                cur_sparsity = (W_mask == True).sum() / W_mask.numel()
                return W_mask, cur_sparsity

            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    accelerator.print(f"Alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            # 🔍 update the state dict
            # 🔍 the weights would not change if directly updating them using "subset[name].weight.data[W_mask] = 0"
            update_state_dict[module_state_dict_name + ".weight"] = (subset[name].weight * (torch.ones_like(W_mask) - W_mask)).bfloat16().cpu()

        # 🔍 prune moe keys. 
        print(f"w1, w2, w3: {w1, w2, w3}")
        for weights in [w1, w2, w3]:
            if len(weights) == 0:
                continue
            # accelerator.print(f"weights: {weights}")
            All_experts_metric = torch.zeros((len(weights),) + subset[weights[0]].weight.data.size())
            All_mask = torch.zeros((len(weights),) + subset[weights[0]].weight.data.size())

            for ei in range(len(weights)):
                name = weights[ei]
                module_state_dict_name = f"model.layers.{i}.{name}"
                accelerator.print(f"Pruning module {module_state_dict_name}")

                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                W_metric = accelerator.reduce(W_metric, reduction="sum")  # 🔍 all reduce across devices
                if All_experts_metric.device != W_metric.device:
                    All_experts_metric = All_experts_metric.to(W_metric.device)
                All_experts_metric[ei, :, :] += W_metric

            if prune_n != 0:
                # structured n:m sparsity
                raise NotImplementedError
            else:
                if args.use_variant:
                    # wanda variant
                    raise NotImplementedError
                else:
                    # unstructured pruning
                    accelerator.print("All_experts_metric", All_experts_metric.shape)
                    experts, din, dout = All_experts_metric.size()
                    All_experts_metric = All_experts_metric.transpose(0, 1).reshape(din, experts * dout)
                    All_mask = All_mask.transpose(0, 1).reshape(din, experts * dout)

                    sort_res = torch.sort(All_experts_metric, dim=-1, stable=True)
                    indices = sort_res[1][:, :int(All_experts_metric.shape[-1] * args.sparsity_ratio)]

                    if All_mask.device != indices.device:
                        All_mask = All_mask.to(indices.device)
                    All_mask.scatter_(-1, indices, True)
                    All_mask = All_mask.reshape(din, experts, dout).transpose(0, 1)

                    for ei in range(len(weights)):
                        name = weights[ei]
                        module_state_dict_name = f"model.layers.{i}.{name}"

                        accelerator.print(f"{module_state_dict_name}: {All_mask[ei].float().mean()}")
                        update_state_dict[module_state_dict_name + ".weight"] = (subset[name].weight
                                                                                 * (torch.ones_like(All_mask[ei]) - All_mask[ei])).bfloat16().cpu()

        # Update inputs & outputs
        for j in range(num_samples):
            outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]

        inputs, outputs = outputs, inputs

    accelerator.print("Pruning done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    # 🔍 return the state dict
    return update_state_dict


@torch.no_grad()
def prune_sparsegpt(args, model, dataloader, accelerator: Accelerator, num_samples, prune_n=0, prune_m=0, blocksize=128, percdamp=0.01):
    """
        SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
        :param num_samples: samples on each device, calculated as "num_samples = n_calibration_samples // num_processes"
    """
    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    # 🔍 store the pruned parameters in CPU
    update_state_dict = {}

    accelerator.print("Getting features...")
    inputs, outputs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, accelerator, num_samples)  # 🔍

    accelerator.print('Starting ...')
    for i in tqdm(range(len(layers)), desc="Pruning layers...", disable=not accelerator.is_main_process):
        sys.stderr.flush()
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)
        layer = layers[i]
        subset = find_layers_for_moe(layer)  # 🔍 Find layers to prune

        # Wrap layers
        gpts = {}
        for name in subset:
            gpts[name] = SparseGPTWrapper(subset[name])

        def add_batch(name):
            def hook(_, input, output):
                gpts[name].add_batch(input[0].data, output.data)

            return hook

        # Get importance
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(num_samples):
            outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
        for h in handles:
            h.remove()

        # Prune
        for name in gpts:
            module_state_dict_name = f"model.layers.{i}.{name}"
            accelerator.print(f"Pruning module {module_state_dict_name}")

            W = subset[name].weight.data.clone()
            if isinstance(subset[name], nn.Conv2d):
                W = W.flatten(1)
            if isinstance(subset[name], transformers.Conv1D):
                W = W.t()
            W = W.float()  # this makes no sense

            H = gpts[name].H
            H = accelerator.reduce(H, reduction="mean")  # 🔍 all reduce across devices
            # gpts[name].H = None
            # torch.cuda.empty_cache()

            dead = (torch.diag(H) == 0)
            H[dead, dead] = 1
            W[:, dead] = 0

            Losses = torch.zeros(gpts[name].rows, device=gpts[name].device)

            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(gpts[name].columns, device=gpts[name].device)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

            mask = None

            # formally begin
            for i1 in range(0, gpts[name].columns, blocksize):
                i2 = min(i1 + blocksize, gpts[name].columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                if prune_n == 0:
                    if mask is not None:
                        mask1 = mask[:, i1:i2]
                    else:
                        tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                        thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * args.sparsity_ratio)]
                        mask1 = (tmp <= thresh)
                else:
                    mask1 = (torch.zeros_like(W1) == 1)

                for j in range(count):
                    w = W1[:, j]
                    d = Hinv1[j, j]

                    if prune_n != 0 and j % prune_m == 0:
                        tmp = W1[:, j:(j + prune_m)] ** 2 / (torch.diag(Hinv1)[j:(j + prune_m)].reshape((1, -1))) ** 2
                        mask1.scatter_(1, j + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                    q = w.clone()
                    q[mask1[:, j]] = 0

                    Q1[:, j] = q
                    Losses1[:, j] = (w - q) ** 2 / d ** 2

                    err1 = (w - q) / d
                    W1[:, j:] -= err1.unsqueeze(1).matmul(Hinv1[j, j:].unsqueeze(0))
                    Err1[:, j] = err1

                W[:, i1:i2] = Q1
                Losses += torch.sum(Losses1, 1) / 2
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if isinstance(subset[name], transformers.Conv1D):
                W = W.t()

            # 🔍 update the state dict
            # 🔍 the weights would not change if directly applying them
            update_state_dict[module_state_dict_name + ".weight"] = W.reshape(subset[name].weight.shape).bfloat16().cpu()
            # subset[name].weight.data = W.reshape(subset[name].weight.shape).to(subset[name].weight.data.dtype)

        # Update inputs & outputs
        for j in range(num_samples):
            outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
        inputs, outputs = outputs, inputs

    accelerator.print("Pruning done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    # 🔍 return the state dict
    return update_state_dict


@torch.no_grad()
def prune_template(args, model, dataloader, accelerator: Accelerator, num_samples, prune_n=0, prune_m=0):
    """Template for pruning methods"""
    raise NotImplementedError("Please copy this function and implement the full method.")

    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    # 🔍 store the pruned parameters in CPU
    update_state_dict = {}

    accelerator.print("Getting features...")
    inputs, outputs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, accelerator, num_samples)  # 🔍

    accelerator.print('Starting ...')
    for i in tqdm(range(len(layers)), desc="Pruning layers...", disable=not accelerator.is_main_process):
        sys.stderr.flush()
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)
        layer = layers[i]
        subset = find_layers_for_moe(layer)  # 🔍 Find layers to prune

        # Wrap layers
        wrapped_layers = {}
        for name in subset:
            # TODO
            pass

        # Forward hook for recording row importance
        def add_batch(name):
            def hook(_, input, output):
                wrapped_layers[name].add_batch(input[0].data, output.data)

            return hook

        # Get importance
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(num_samples):
            outputs[j] = layer(inputs[j], attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        # Prune
        for name in subset:
            module_state_dict_name = f"model.layers.{i}.{name}"  # 🔍
            accelerator.print(f"Pruning module {module_state_dict_name}")

            # TODO

            xxxxxx = None

            # 🔍 update the state dict
            # 🔍 the weights would not change if directly applying them
            update_state_dict[module_state_dict_name + ".weight"] = xxxxxx.bfloat16().cpu()  # TODO

    accelerator.print("Pruning done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    # 🔍 return the state dict
    return update_state_dict


@torch.no_grad()
def prune_ablate(args, model, tokenizer, device, seqlen=2048, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4", nsamples=args.n_calibration_samples, seed=args.prune_seed, seqlen=seqlen, tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inputs = torch.zeros(
        (args.n_calibration_samples, seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, input, **kwargs):
            inputs[cache['i']] = input
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outputs = torch.zeros_like(inputs)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:
            device = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {device}")
            inputs, outputs, attention_mask, position_ids = inputs.to(device), outputs.to(device), attention_mask.to(device), position_ids.to(device)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, input, output):
                gpts[name].add_batch(input[0].data, output.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.n_calibration_samples):
            outputs[j] = layer(inputs[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.n_calibration_samples):
            outputs[j] = layer(inputs[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inputs, outputs = outputs, inputs

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
