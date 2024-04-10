import math
import torch
from torch import nn as nn

from src import transformers
from src.transformers.models.mixtral.modeling_mixtral import Expert


class WandaWrapper:

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.device = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.device)  # importance for each row
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, input, output, routing_scores=None):
        if isinstance(self.layer, Expert):
            input = input * routing_scores  # 🔍 add routing_scores

        if len(input.shape) == 2:
            input = input.unsqueeze(0)  # 🔍 input: shape(1, tokens, hidden_size)
        tmp = input.shape[0]

        if isinstance(self.layer, (nn.Linear, Expert)):  # 🔍 for both Linear and Expert
            if len(input.shape) == 3:
                input = input.reshape((-1, input.shape[-1]))  # input: shape(batch_size * seq_len, hidden_size)
            input = input.t()
        input = input.type(torch.float32)
        # print(f"input: {input.size()}, nsamples: {self.nsamples}")

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.scaler_row += torch.norm(input, p=2, dim=1) ** 2 / self.nsamples  # 🔍 determined by the number of input tokens


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
