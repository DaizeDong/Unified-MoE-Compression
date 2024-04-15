import math
import random
import sys
import torch
from accelerate import Accelerator
from torch import nn as nn
from tqdm import tqdm
from typing import Optional

from llmtuner.train.prune.utils import find_modules_for_moe, print_gpu_memory


def low_rank_decomposition(
        weight: torch.Tensor,
        rank_ratio: Optional[float] = 0.1,
        parameter_ratio: Optional[float] = 0.15,
        remove_criteria: Optional[str] = 'max_eigenvalue',
        return_dict: Optional[bool] = False,
        output_device: Optional[str] = "cpu",
):
    """
    Parameters
    ----------
    weight: torch.Tensor
        The matrix to decompose, of shape (H, W)
    rank_ratio: float, optional, default 0.1
        The ratio of the reduced rank to the original rank:
            rank_of_decomposed_matrix / rank_of_input_weight
    parameter_ratio: float, optional, default 0.15
        The ratio of the number of parameters of the decomposed matrix to the original matrix:
            parameter_num_of_decomposed_matrix / (H * W).
        If specify, override rank_ratio
    remove_criteria: str, optional, default 'max_eigenvalue'
        The criteria to remove the small eigenvalues, of ['max_eigenvalue', 'random', 'min_eigenvalue']
    return_dict: bool, optional, default False
        Return a dict if True, else return a tuple (L, R)
    debug: bool, optional, default False
        Print debug information if True
    """
    matrix_dimension = len(weight.size())
    assert matrix_dimension == 2, "Only Support 2D matrix"
    height, width = weight.size()

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    dtype = weight.dtype
    weight = weight.float()
    u_, s_, v_ = torch.linalg.svd(weight, full_matrices=False)
    u_ = u_.to(dtype)
    s_ = s_.to(dtype)
    v_ = v_.to(dtype)
    rank = torch.count_nonzero(s_)

    # return None
    if parameter_ratio is not None:
        reduced_rank = math.ceil(parameter_ratio * (height * width) / (height + width))
    else:
        reduced_rank = math.ceil(rank * rank_ratio)

    if remove_criteria == 'max_eigenvalue':
        l_ = u_ @ (torch.sqrt(torch.diag(s_)[:, 0:reduced_rank]))
        r_ = torch.sqrt(torch.diag(s_)[0:reduced_rank, :]) @ v_
    elif remove_criteria == 'random':
        selected_index = random.choices(range(len(s_)), k=reduced_rank)
        l_ = u_ @ (torch.sqrt(torch.diag(s_)[:, selected_index]))
        r_ = torch.sqrt(torch.diag(s_)[selected_index, :]) @ v_
    elif remove_criteria == 'min_eigenvalue':
        len_s = len(s_)
        l_ = u_ @ (torch.sqrt(torch.diag(s_)[:, len_s - reduced_rank:]))
        r_ = torch.sqrt(torch.diag(s_)[len_s - reduced_rank:, :]) @ v_
    else:
        raise NameError("remove criteria not support")

    if return_dict:
        # return {"L": l_, "R": r_, "U": u_, "S": s_, "Vh": v_, 'reduced_rank': reduced_rank}
        return {"L": l_, "R": r_, 'reduced_rank': reduced_rank}
    else:
        return l_, r_


def _substitute_single_linear_weight(
        module: nn.Module,
        accelerator: Accelerator,
        parameter_ratio: float,
        has_sparse: bool,
        use_svd: bool,
        update_state_dict: dict,
        module_state_dict_name: str,
        device,
        **kwargs
) -> nn.Module:
    """
    Substitute a single Linear weight with to LinearLoSparse

    Examples
    --------
    >>> linear = nn.Linear(16, 32)
    >>> linear = _substitute_single_linear_weight(linear, parameter_ratio=0.15, has_sparse=True, use_svd=True)
    Reduced Rank: 2 | Num Parameters: 96
    >>> linear
    LinearLoSparse(
      (right): Linear(in_features=16, out_features=2, bias=False)
      (left): Linear(in_features=2, out_features=32, bias=False)
      (sparse): Linear(in_features=16, out_features=32, bias=False)
    )
    """
    has_bias = module.bias is not None

    if use_svd:
        # Decompose a matrix by SVD
        weight_tensor = module.weight.data.to(device)  # 🔍 here to device
        output = low_rank_decomposition(weight_tensor, parameter_ratio=parameter_ratio, return_dict=True, **kwargs)
        l_, r_, reduced_rank = output['L'], output['R'], output['reduced_rank']
        s_ = weight_tensor - torch.mm(l_, r_)
    else:
        height, width = module.weight.shape
        reduced_rank = math.ceil(parameter_ratio * (height * width) / (height + width))
        l_ = torch.zeros(height, reduced_rank, requires_grad=False)
        r_ = torch.zeros(reduced_rank, width, requires_grad=False)
        s_ = torch.zeros(height, width, requires_grad=False)

    # Create a nn.Module and assign decomposed weights to the parameters
    # in_features, out_features = module.in_features, module.out_features
    # module = LoSparseLinear(in_features, out_features, reduced_rank, has_bias=has_bias, has_sparse=has_sparse)
    # l_, r_, s_ = l_.to("cpu"), r_.to("cpu"), s_.to("cpu")
    # module.initialize_weight(l_, r_, s_)

    update_state_dict[module_state_dict_name + ".left.weight"] = l_.bfloat16().cpu()  # 🔍 to cpu
    update_state_dict[module_state_dict_name + ".right.weight"] = r_.bfloat16().cpu()
    if has_sparse:
        update_state_dict[module_state_dict_name + ".sparse.weight"] = s_.bfloat16().cpu()

    torch.cuda.empty_cache()
    accelerator.free_memory()


@torch.no_grad()
def decompose_moe(model, accelerator: Accelerator, parameter_ratio=0.15, has_sparse=True, use_svd=True):
    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    # 🔍 store the pruned parameters in CPU
    update_state_dict = {}
    accelerator.print('Starting ...')

    #####################################
    # 🔍 set parameter ratio for saving.
    setattr(model.config, "parameter_ratio", parameter_ratio)

    #####################################
    for i in tqdm(range(len(layers)), desc="Decomposing layers...", disable=not accelerator.is_main_process):
        sys.stderr.flush()
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)
        layer = layers[i]
        subset = find_modules_for_moe(layer)  # 🔍 Find layers to prune

        # Wrap layers
        for name in subset:
            module_state_dict_name = f"model.layers.{i}.{name}"
            accelerator.print(f"Decomposing layer {i} {name}...")
            # 🔍 TODO: parallel
            _substitute_single_linear_weight(
                module=subset[name].w1,
                accelerator=accelerator,
                parameter_ratio=parameter_ratio,
                has_sparse=has_sparse,
                use_svd=use_svd,
                update_state_dict=update_state_dict,
                module_state_dict_name=module_state_dict_name + ".w1",
                device=device,
                # **kwargs
            )
            _substitute_single_linear_weight(
                module=subset[name].w2,
                accelerator=accelerator,
                parameter_ratio=parameter_ratio,
                has_sparse=has_sparse,
                use_svd=use_svd,
                update_state_dict=update_state_dict,
                module_state_dict_name=module_state_dict_name + ".w2",
                device=device,
                # **kwargs
            )
            _substitute_single_linear_weight(
                module=subset[name].w3,
                accelerator=accelerator,
                parameter_ratio=parameter_ratio,
                has_sparse=has_sparse,
                use_svd=use_svd,
                update_state_dict=update_state_dict,
                module_state_dict_name=module_state_dict_name + ".w3",
                device=device,
                # **kwargs
            )

    accelerator.print(f"update_state_dict: {update_state_dict.keys()}")
    accelerator.print("Decomposition done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    return update_state_dict
