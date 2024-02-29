import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator

from .ablate import AblateGPT
from .data import get_loaders
from .info import print_gpu_memory
from .layerwrapper import WrappedGPT
from .sparsegpt import SparseGPT


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def find_layers_for_moe(module, layers=[nn.Linear], name=''):
    # 🔍 excluded the gate networks for MoE
    res = find_layers(module, layers, name)
    for key in list(res.keys()):
        if "gate" in key:
            res.pop(key)
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

    layer_num = max(list(layer_params.keys()))

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
def prepare_calibration_input(model, dataloader, accelerator: Accelerator, num_samples=128, seqlen=2048):
    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍

    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    dtype = next(iter(unwrapped_model.parameters())).dtype
    inputs = torch.zeros((num_samples, seqlen, unwrapped_model.config.hidden_size), dtype=dtype, device=device)
    inputs.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, input, **kwargs):
            inputs[cache['i']] = input  # 🔍 the batch size must be 1
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for index, batch in enumerate(dataloader):
        if index >= num_samples:  # 🔍 limit the number of samples in each device
            break
        try:
            model(**batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    outputs = torch.zeros_like(inputs)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    unwrapped_model.config.use_cache = use_cache

    return inputs, outputs, attention_mask, position_ids


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W) == 1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel() * args.sparsity_ratio)].cpu()
                W_mask = (W_metric <= thresh)

            W[W_mask] = 0


@torch.no_grad()
def prune_wanda(args, model, dataloader, accelerator: Accelerator, num_samples, seqlen, prune_n=0, prune_m=0):
    """
    :param num_samples: samples on each device, calculated as "num_samples = n_calibration_samples // num_processes"
    """
    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    # 🔍 store the pruned parameters in CPU on the main process
    # if accelerator.is_main_process:
    update_state_dict = {}

    accelerator.print("Getting features...")
    inputs, outputs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, accelerator, num_samples, seqlen)  # 🔍
    # inputs/outputs: shape(num_samples, seqlen, hidden_size)

    accelerator.print('Starting ...')
    for i in range(len(layers)):
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)  # GPU Memory
        layer = layers[i]
        subset = find_layers_for_moe(layer)  # 🔍 Find layers to prune

        # Wrap layers
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):  # Forward hook for recording row importance
            def hook(_, input, output):
                wrapped_layers[name].add_batch(input[0].data, output.data)

            return hook

        # Get importance
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(num_samples):  # 🔍 n_calibration_samples -> num_samples
            outputs[j] = layer(inputs[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        # Prune
        for name in subset:
            state_dict_name = f"model.layers.{i}.{name}"  # 🔍
            accelerator.print(f"Pruning layer {state_dict_name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            W_metric = accelerator.reduce(W_metric, reduction="mean")  # 🔍 all reduce across devices
            W_mask = (torch.zeros_like(W_metric) == 1)  # initialize a mask to be all False

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
            # 🔍 the weights would not change if directly update them using "subset[name].weight.data[W_mask] = 0"
            # if accelerator.is_main_process:
            update_state_dict[state_dict_name] = (subset[name].weight * W_mask).cpu()
            accelerator.wait_for_everyone()

        # Update inputs & outputs
        for j in range(num_samples):  # 🔍 n_calibration_samples -> num_samples
            outputs[j] = layer(inputs[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inputs, outputs = outputs, inputs

    accelerator.print("Pruning done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    # 🔍 return the state dict
    # if accelerator.is_main_process:
    #     return update_state_dict
    # else:
    #     return None
    return update_state_dict


@torch.no_grad()
def prune_sparsegpt(args, model, dataloader, accelerator: Accelerator, num_samples, seqlen, prune_n=0, prune_m=0, blocksize=128, percdamp=0.01):
    """
        SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
        :param num_samples: samples on each device, calculated as "num_samples = n_calibration_samples // num_processes"
    """
    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    # 🔍 store the pruned parameters in CPU on the main process
    update_state_dict = {}

    accelerator.print("Getting features...")
    inputs, outputs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, accelerator, num_samples, seqlen)  # 🔍
    # inputs/outputs: shape(num_samples, seqlen, hidden_size)

    accelerator.print('Starting ...')
    for i in range(len(layers)):
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)  # GPU Memory
        layer = layers[i]
        subset = find_layers_for_moe(layer)  # 🔍 Find layers to prune

        # Wrap layers
        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def hook(_, input, output):
                gpts[name].add_batch(input[0].data, output.data)

            return hook

        # Get importance
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(num_samples):  # 🔍 n_calibration_samples -> num_samples
            outputs[j] = layer(inputs[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        # Prune
        for name in gpts:
            state_dict_name = f"model.layers.{i}.{name}"  # 🔍
            accelerator.print(f"Pruning layer {state_dict_name}")

            W = subset[name].layer.weight.data.clone()
            if isinstance(subset[name].layer, nn.Conv2d):
                W = W.flatten(1)
            if isinstance(subset[name].layer, transformers.Conv1D):
                W = W.t()
            W = W.float()

            H = subset[name].H
            H = accelerator.reduce(H, reduction="mean")  # 🔍 all reduce across devices
            subset[name].H = None
            torch.cuda.empty_cache()
            dead = (torch.diag(H) == 0)
            H[dead, dead] = 1
            W[:, dead] = 0

            Losses = torch.zeros(subset[name].rows, device=subset[name].dev)

            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(subset[name].columns, device=subset[name].dev)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

            mask = None

            # formally begin
            for i1 in range(0, subset[name].columns, blocksize):
                i2 = min(i1 + blocksize, subset[name].columns)
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
                        mask1 = tmp <= thresh
                else:
                    mask1 = (torch.zeros_like(W1) == 1)

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if prune_n != 0 and i % prune_m == 0:
                        tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                        mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                    q = w.clone()
                    q[mask1[:, i]] = 0

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d ** 2

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                W[:, i1:i2] = Q1
                Losses += torch.sum(Losses1, 1) / 2
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            # torch.cuda.synchronize()
            if isinstance(subset[name].layer, transformers.Conv1D):
                W = W.t()

            # 🔍 update the state dict
            # 🔍 the weights would not change if directly apply them
            update_state_dict[state_dict_name] = W.reshape(subset[name].layer.weight.shape).cpu()
            # subset[name].layer.weight.data = W.reshape(subset[name].layer.weight.shape).to(subset[name].layer.weight.data.dtype)


            # gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            # gpts[name].free()

        for j in range(num_samples):  # 🔍 n_calibration_samples -> num_samples
            outputs[j] = layer(inputs[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer
        inputs, outputs = outputs, inputs

    accelerator.print("Pruning done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    # 🔍 return the state dict
    return update_state_dict


@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, seqlen=2048, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4", nsamples=args.n_calibration_samples, seed=args.prune_seed, seqlen=seqlen, tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inputs = torch.zeros(
        (args.n_calibration_samples, seqlen, model.config.hidden_size), dtype=dtype, device=dev
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
            model(batch[0].to(dev))
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
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inputs, outputs, attention_mask, position_ids = inputs.to(dev), outputs.to(dev), attention_mask.to(dev), position_ids.to(dev)

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
