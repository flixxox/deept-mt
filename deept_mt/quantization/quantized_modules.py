
import torch
import torch.nn as nn
import torch.nn.functional as F

from deept.util.globals import Settings

from deept_mt.quantization.quantizer import (
    MinMaxFakeWeightQuantizer
)


class QuantizedLinear(nn.Module):

    def __init__(self, in_dim, out_dim, bits, bias=True):
        super().__init__()
        
        self.weight = nn.Parameter(torch.empty((out_dim, in_dim)), requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.empty((out_dim)), requires_grad=True)

        self.weight_quantizer = MinMaxFakeWeightQuantizer(bits, torch.quint8)

    def __call__(self, x):
        weight = self.weight_quantizer(self.weight)
        return F.linear(x, weight, self.bias)


class QuantizedLinearRelu(nn.Module):

    def __init__(self, in_dim, out_dim, bits, bias=True):
        super().__init__()
        
        self.weight = nn.Parameter(torch.empty((out_dim, in_dim)), requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.empty((out_dim)), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.weight_quantizer = MinMaxFakeWeightQuantizer(bits, torch.quint8)

    def __call__(self, x):
        weight = self.weight_quantizer(self.weight)
        return F.relu(F.linear(x, weight, self.bias))