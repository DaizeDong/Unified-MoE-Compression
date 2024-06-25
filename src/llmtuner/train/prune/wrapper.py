import logging
import math

import torch
import torch.nn.functional as F
import transformers
from torch import nn as nn

from llmtuner.model.deepseek.modeling_deepseek import MoEGate
from llmtuner.model.mixtral.modeling_mixtral import MixtralSparseMoeBlock

logger = logging.getLogger(__name__)

"""For pruning"""


class WandaWrapper:
    def __init__(self, layer, layer_id=0, layer_name="none", multiply_score=True, p=2):
        self.layer = layer
        self.layer_id = layer_id
        self.layer_name = layer_name
        self.device = self.layer.weight.device
        # print(layer_name, layer.weight.data.shape)

        self.scaler_row = None  # importance for each row
        self.weight = None  # 👆 record weight
        self.nsamples = 0

        # 🔍 for dynamic sparsity using scores
        self.multiply_score = multiply_score
        self.score_memery = torch.zeros((1,), device=self.device, dtype=torch.float32)  # the summation of (score ** p)
        self.p = p

    def add_batch(self, input, output, routing_scores=None):
        # 🔍 rescale inputs with scores
        if routing_scores is not None:
            if self.multiply_score:
                # multiple routing_scores to inputs
                routing_scores = (routing_scores ** (self.p / 2))  # dividing 2 as the latter "scaler_row" will calculate the squared value
                input = input * routing_scores
            else:
                # add routing_scores to memory
                # 🔍 compute scores to obtain sparse ratios.
                self.score_memery += (routing_scores ** self.p).sum().float()  # add the token scores

        self.add_batch_no_score(input, output)

    def add_batch_no_score(self, input, output):
        # 👆 capture the intact weights when possible!!!!!!!!!!!!!!!!!!!!!!
        if self.weight is None and self.layer.weight.data.shape[0] > 0:
            self.weight = self.layer.weight.data.clone().cpu()
            self.rows = self.weight.shape[0]
            self.columns = self.weight.shape[1]
            # print(f"record {self.layer_name}, {self.weight.data.shape}")

        if len(input.shape) == 2:
            input = input.unsqueeze(0)  # 🔍 shape(1, tokens, hidden_size)
        tmp = input.shape[0]

        if len(input.shape) == 3:
            input = input.reshape((-1, input.shape[-1]))  # shape(hidden_size, batch_size * seq_len)
        input = input.t()  # shape(hidden_size, batch_size * seq_len)
        input = input.type(torch.float32)

        if self.scaler_row is None:
            self.scaler_row = torch.zeros((input.shape[0],), device=self.device)
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.scaler_row += (torch.norm(input, p=2, dim=1) ** 2) / self.nsamples  # 🔍 determined by the number of input tokens
        # Description: torch.norm(input, p=2, dim=1) ** 2 <==> (input * input).sum(1), which is $\sum_{x_i\in X} x_i^2$


class SparseGPTWrapper:
    def __init__(self, layer):
        self.layer = layer
        self.device = self.layer.weight.device

        self.H = None  # importance for each row
        self.weight = None  # 👆 record weight
        self.nsamples = 0

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def add_batch(self, input, output):
        # 👆 capture the intact weights when possible!!!!!!!!!!!!!!!!!!!!!!
        if self.weight is None and self.layer.weight.data.shape[0] > 0:
            self.weight = self.layer.weight.data.clone().cpu()
            self.rows = self.weight.shape[0]
            self.columns = self.weight.shape[1]
            # print(f"record {self.layer_name}, {self.weight.data.shape}")

        if len(input.shape) == 2:
            input = input.unsqueeze(0)  # shape(batch_size, seq_len, hidden_size)
        batch_size = input.shape[0]

        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(input.shape) == 3:
                input = input.reshape((-1, input.shape[-1]))  # shape(batch_size * seq_len, hidden_size)
            input = input.t()  # shape(hidden_size, batch_size * seq_len)
        input = input.type(torch.float32)

        # Estimate the mean Hessian through iterative updates
        if self.H is None:
            self.H = torch.zeros((input.shape[0], input.shape[0]), device=self.device)
        self.H *= self.nsamples / (self.nsamples + batch_size)  # shrink old mean values
        self.nsamples += batch_size
        input = math.sqrt(2 / self.nsamples) * input.float()
        self.H += input.matmul(input.t())  # update mean values by adding values from new samples


"""For recording weights"""


class HiddenStatesRecordWrapper:
    def __init__(self, layer, layer_name="none", record_input=True, record_output=True):
        self.layer = layer
        self.layer_name = layer_name

        self.record_input = record_input
        self.record_output = record_output

        if record_input:
            self.input_hidden_states = []
        if record_output:
            self.output_hidden_states = []

    def record(self, input, output):
        # input: (1, seq_len, hidden_size)
        if self.record_input:
            self.input_hidden_states.append(input.squeeze(0).clone().cpu())
        if self.record_output:
            self.output_hidden_states.append(output.squeeze(0).clone().cpu())


class WeightRecordWrapper:
    def __init__(self, layer, layer_name="none"):
        self.layer = layer
        self.layer_name = layer_name
        self.weight = None

    def record(self, input, output):
        if self.weight is None and self.layer.weight.data.shape[0] > 0:
            # capture the intact weights when possible!!!!!!!!!!!!!!!!!!!!!!
            self.weight = self.layer.weight.data.clone().cpu()
            # print(f"record {self.layer_name}, {self.weight.data.shape}")


"""For expert drop"""


class MixtralExpertDropWrapper:
    def __init__(self, layer: MixtralSparseMoeBlock):
        self.layer = layer
        self.scores = None
        self.nsamples = 0
        self.top_k = layer.top_k
        self.ana_list = []

    def add_batch(self, input, router_logits):
        if len(input.shape) == 2:
            batch_size = 1
        else:
            batch_size = input.shape[0]

        # Record scores
        routing_weights = router_logits.reshape(-1, router_logits.shape[-1])  # router_logits: shape(batch_size * seq_len, n_experts)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        mask = torch.zeros_like(router_logits, device=router_logits.device)
        mask.scatter_(-1, selected_experts, 1)
        # print(f"routing_weights: {routing_weights}")

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        # routing_weights = routing_weights * mask
        # print(f"routing_weights: {routing_weights}")

        for j in range(len(routing_weights)):
            sorted_weights, sort_indices = torch.sort(routing_weights[j], descending=True)
            self.ana_list.append(float(sorted_weights[1] / sorted_weights[0]))

        # The above code is reshaping the `router_logits` array into a 2D array with a shape of
        # `(batch_size * seq_len, n_experts)`. This means that it is rearranging the elements of the
        # `router_logits` array into a new shape where the first dimension is the product of
        # `batch_size` and `seq_len`, and the second dimension is `n_experts`.
        # print("routing_weights", routing_weights.shape)

        if self.scores is None:
            self.nsamples += batch_size
            self.scores = routing_weights.float().sum(0) / self.nsamples

        else:
            self.scores *= self.nsamples / (self.nsamples + batch_size)  # shrink old mean values
            self.nsamples += batch_size
            self.scores += routing_weights.float().sum(0) / self.nsamples  # update mean values by adding values from new samples


class DeepseekExpertDropWrapper:
    def __init__(self, layer: MoEGate):
        self.layer = layer
        self.scores = None
        self.nsamples = 0
        self.n_routed_experts = layer.n_routed_experts

    def add_batch(self, input, topk_idx, topk_weight):
        if len(input.shape) == 2:
            batch_size = 1
        else:
            batch_size = input.shape[0]

        # Record scores
        topk_idx = topk_idx.reshape(-1, topk_idx.shape[-1])  # shape(batch_size * seq_len, num_selects)
        topk_weight = topk_weight.reshape(-1, topk_weight.shape[-1])  # shape(batch_size * seq_len, num_selects)

        routing_weights = torch.zeros((topk_weight.shape[0], self.n_routed_experts), device=topk_weight.device, dtype=topk_weight.dtype)
        routing_weights = torch.scatter(routing_weights, dim=1, index=topk_idx, src=topk_weight)

        if self.scores is None:
            self.nsamples += batch_size
            self.scores = routing_weights.float().sum(0) / self.nsamples
        else:
            self.scores *= self.nsamples / (self.nsamples + batch_size)  # shrink old mean values
            self.nsamples += batch_size
            self.scores += routing_weights.float().sum(0) / self.nsamples  # update mean values by adding values from new samples
