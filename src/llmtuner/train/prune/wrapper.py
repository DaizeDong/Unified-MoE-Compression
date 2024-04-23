import itertools
import logging
import math
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.utils.data import Dataset
from typing import Optional

import transformers
from transformers.models.mixtral.modeling_mixtral import Expert
from transformers.models.mixtral.modeling_mixtral import (
    # MixtralForCausalLM,
    MixtralSparseMoeBlock,
    # MixtralBLockSparseTop2MLP
)

# from data import CacheDataset

logger = logging.getLogger(__name__)


class WandaWrapper:
    def __init__(self, layer, layer_id=0, layer_name="none", multiply_score=True, p=2):
        self.layer = layer
        self.device = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.device)  # importance for each row
        self.nsamples = 0

        self.multiply_score = multiply_score
        self.score_memery = torch.zeros((1,), device=self.device, dtype=torch.float32)  # the summation of (score ** p)

        self.p = p
        self.layer_id = layer_id
        self.layer_name = layer_name

    def numel(self):
        return self.rows * self.columns

    # 🔍 compute scores to obtain sparse ratios. 
    def add_scores(self, routing_scores):
        self.score_memery += (routing_scores ** self.p).sum().clone().float()

    def add_batch(self, input, output, routing_scores=None):
        # print(f"routing_scores: {routing_scores.shape}")
        # print(f"routing_scores: {routing_scores}")
        # print(type(self.layer))
        # print(self.p)
        # print(self.layer_name, type(self.layer), isinstance(self.layer, Expert))

        if isinstance(self.layer, Expert):
            if self.multiply_score:
                # 🔍 multiple routing_scores to inputs
                routing_scores = (routing_scores ** (self.p / 2))  # dividing 2 as the latter "scaler_row" will calculate the squared value
                input = input * routing_scores
            else:
                # 🔍 add routing_scores to memory
                # routing_scores = (routing_scores ** self.p)
                # self.score_memery += routing_scores.numel().clone().float()  # number of tokens
                self.score_memery += (routing_scores ** self.p).sum().float()  # scores

        if len(input.shape) == 2:
            input = input.unsqueeze(0)  # 🔍 input: shape(1, tokens, hidden_size)
        tmp = input.shape[0]

        if isinstance(self.layer, (nn.Linear, Expert)):  # 🔍 for both Linear and Expert
            if len(input.shape) == 3:
                input = input.reshape((-1, input.shape[-1]))  # input: shape(batch_size * seq_len, hidden_size)
            input = input.t()
        input = input.type(torch.float32)

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.scaler_row += (torch.norm(input, p=2, dim=1) ** 2) / self.nsamples  # 🔍 determined by the number of input tokens
        # Description: torch.norm(input, p=2, dim=1) ** 2 <==> (input * input).sum(1), which is $\sum_{x_i\in X} x_i^2$


class SparseGPTWrapper:
    def __init__(self, layer):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        self.layer = layer
        self.device = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.device)
        self.nsamples = 0

    def add_batch(self, input, output):
        if len(input.shape) == 2:
            input = input.unsqueeze(0)  # input: shape(batch_size, seq_len, hidden_size)
        batch_size = input.shape[0]

        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(input.shape) == 3:
                input = input.reshape((-1, input.shape[-1]))  # input: shape(batch_size * seq_len, hidden_size)
            input = input.t()

        # Estimate the mean Hessian through iterative updates
        self.H *= self.nsamples / (self.nsamples + batch_size)  # shrink old mean values
        self.nsamples += batch_size
        input = math.sqrt(2 / self.nsamples) * input.float()
        self.H += input.matmul(input.t())  # update mean values by adding values from new samples


class GateRemapWrapper:
    def __init__(self, layer, layer_id=0, layer_name="none", record_input=True, record_output=True):
        self.layer = layer
        self.device = self.layer.weight.device

        self.record_input = record_input
        self.record_output = record_output

        self.inputs = []
        self.outputs = []

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, input, output):
        if self.record_input:
            self.inputs.append(input.reshape(-1, input.shape[-1]).float())  # (token_num, dim)

        if self.record_output:
            self.outputs.append(output.reshape(-1, output.shape[-1]).float())  # (token_num, dim)


class PrunableMixtralSparseMoeBlockWrapper(nn.Module):
    def __init__(self, layer: MixtralSparseMoeBlock, r: Optional[int] = None):
        super().__init__()
        self.layer = layer
        self.r = r

        self.experts_to_drop = None
        self.cache_space = CacheDataset()
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False

    # Forward uses topk
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.layer.gate(hidden_states)

        # 🔍 set the weights to "-inf" for dropped experts, however this doesn't change the selected num
        if self.experts_to_drop is not None:
            for e in self.experts_to_drop:
                router_logits[:, e] = -float('inf')

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.layer.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.layer.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the layer and perform the computation on each expert
        for expert_idx in range(self.layer.num_experts):
            expert_layer = self.layer.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None])

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        if self.experts_to_drop is not None and (self.cache_logits or self.cache_X or self.cache_Z):
            logger.warning(f'Already dropped {self.experts_to_drop} but still storing activations.')
        self.cache_space.append(
            alpha=(router_logits if self.cache_logits else None),
            X=(hidden_states if self.cache_X else None),
            Z=(final_hidden_states if self.cache_Z else None)
        )

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    @torch.no_grad()
    def enumerate(self):
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False
        device = self.layer.gate.weight.data.device
        loss_history = dict()

        for name, params in self.layer.named_parameters():
            print(f"name: {name}, params: {params.size(), params.device}")

        for dropped in itertools.combinations(range(self.layer.num_experts), self.layer.num_experts - self.r):
            # 🔍 O(n!) time complexity. NB.
            # 🔍 Here the loss measures the L2 deviation of the output.
            self.experts_to_drop = dropped
            loss = torch.zeros((1,), device=device)

            for (hidden_states, final_hidden_states) in zip(self.cache_space.Xs, self.cache_space.Zs):
                print(f"hidden_states: {hidden_states.size()}")
                hidden_states = hidden_states.to(device=device, non_blocking=True)
                final_hidden_states = final_hidden_states.to(device=device, non_blocking=True)
                # final_hidden_states = final_hidden_states.to(dtype=torch.float64, device=device, non_blocking=True)

                # 🔍 why to float64? seems unnecessary.
                final_hidden_states_e, _ = self.forward(hidden_states.unsqueeze(0))
                loss += torch.norm(final_hidden_states - final_hidden_states_e.squeeze(0), p=2).item()
                # loss += torch.norm(final_hidden_states - final_hidden_states_e.squeeze(0).to(torch.float64), p=2).item()

            loss_history[dropped] = loss

        return loss_history

    @torch.no_grad()
    def update_dropped_experts(self, loss_history):
        self.experts_to_drop = min(loss_history, key=loss_history.get)

    # @torch.no_grad()
    # def prune(self):
    #     assert self.experts_to_drop is not None
    #     assert len(self.experts_to_drop) == self.layer.num_experts - self.r
    #     del self.cache_space
    #     self.cache_X = False
    #     self.cache_Z = False
    #
    #     experts_to_reserve = sorted(set(range(self.layer.num_experts)) - set(self.experts_to_drop))
    #
    #     gate_new = torch.nn.Linear(in_features=self.layer.gate.in_features, out_features=self.r, bias=False, device='cpu', dtype=torch.bfloat16)
    #     gate_new.weight.data = self.layer.gate.weight.data[list(experts_to_reserve)]
    #     self.layer.gate = gate_new
    #
    #     self.layer.experts = torch.nn.ModuleList([self.layer.experts[i] for i in experts_to_reserve])
    #     self.layer.num_experts = self.r

    @torch.no_grad()
    def prune(self, update_state_dict, layer_id):
        assert self.experts_to_drop is not None
        assert len(self.experts_to_drop) == self.layer.num_experts - self.r
        del self.cache_space
        self.cache_X = False
        self.cache_Z = False

        experts_to_reserve = sorted(set(range(self.layer.num_experts)) - set(self.experts_to_drop))

        update_state_dict[f"model.layers.{layer_id}.block_sparse_moe.gate.weight"] = self.layer.gate.weight.data[experts_to_reserve, :].clone().float16().cpu()
        for new_expert_id, old_expert_id in enumerate(experts_to_reserve):
            update_state_dict[f"model.layers.{layer_id}.block_sparse_moe.experts.{new_expert_id}.w1.weight"] = self.layer.experts[old_expert_id].w1.weight.data.clone().float16().cpu()
            update_state_dict[f"model.layers.{layer_id}.block_sparse_moe.experts.{new_expert_id}.w2.weight"] = self.layer.experts[old_expert_id].w2.weight.data.clone().float16().cpu()
            update_state_dict[f"model.layers.{layer_id}.block_sparse_moe.experts.{new_expert_id}.w3.weight"] = self.layer.experts[old_expert_id].w3.weight.data.clone().float16().cpu()


class DynamicSkippingMixtralSparseMoeBlockWrapper(nn.Module):
    def __init__(self, model: MixtralSparseMoeBlock, beta: float):
        super().__init__()
        assert isinstance(model, MixtralSparseMoeBlock)
        assert model.top_k == 2
        self.hidden_dim = model.hidden_dim
        self.ffn_dim = model.ffn_dim
        self.num_experts = model.num_experts
        self.top_k = model.top_k
        self.gate = model.gate
        self.experts = model.experts

        self.beta = beta

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1)

        # (batch * sequence_length)
        mask_top1 = (routing_weights[:, 1] < self.beta * routing_weights[:, 0])
        routing_weights[mask_top1, 1] = 0

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        # (batch * sequence_length, self.top_k, n_experts)
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts)

        expert_mask[mask_top1, 1, :] = 0
        expert_mask = expert_mask.permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            top_x, indices = torch.where(expert_mask[expert_idx])

            if indices.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            indices_list = indices.tolist()
            top_x_list = top_x.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None,
            indices_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(
                current_state, routing_weights[indices_list, top_x_list, None])

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, indices, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class CacheDataset(Dataset):
    def __init__(self):
        self.alphas = []
        self.Xs = []
        self.Zs = []
        self.prepared = False

    def __len__(self):
        if not self.prepared:
            self.prepare_for_loader()
        return len(self.alphas)

    def __getitem__(self, index):
        if not self.prepared:
            self.prepare_for_loader()
        if isinstance(index, list):
            return [(self.alphas[idx], self.Xs[idx], self.Zs[idx]) for idx in index]
        elif isinstance(index, int):
            return self.alphas[index], self.Xs[index], self.Zs[index]

    def append(self, alpha=None, X=None, Z=None):
        if alpha is not None:
            self.alphas.append(alpha.detach().to('cpu', non_blocking=True))
        if X is not None:
            self.Xs.append(X.detach().to('cpu', non_blocking=True))
        if Z is not None:
            self.Zs.append(Z.detach().to('cpu', non_blocking=True))
        self.prepared = False

    def prepare_for_loader(self):
        if self.prepared:
            return
        self.prepared = True
        self.alphas = torch.concat(self.alphas)
        self.Xs = torch.concat(self.Xs)
        self.Zs = torch.concat(self.Zs)
        assert len(self.Xs) == len(self.Zs)
