import torch
import torch.nn as nn


# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.device = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.device)  # importance for each row
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, input, output):
        if len(input.shape) == 2:
            input = input.unsqueeze(0)
        tmp = input.shape[0]

        if isinstance(self.layer, nn.Linear):
            if len(input.shape) == 3:
                input = input.reshape((-1, input.shape[-1]))
            input = input.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        input = input.type(torch.float32)
        self.scaler_row += torch.norm(input, p=2, dim=1) ** 2 / self.nsamples
