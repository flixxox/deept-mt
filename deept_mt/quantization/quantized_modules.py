
import torch
import torch.nn as nn
import torch.nn.functional as F

from deept.util.globals import Settings


from deept_mt.quantization.quant_utils import get_dtype_from_string

from deept_mt.quantization.quantizer import (
    FakeWeightQuantizer,
    FakeActivationQuantizer
)

class LinearBase(nn.Module):

    def __init__(self, in_dim, out_dim, 
        bits,
        weight_quant_dtype,
        weight_quant_method,
        activation_quant_dtype,
        activation_quant_method,
        bias
    ):
        super().__init__()
        
        self.weight = nn.Parameter(torch.empty((out_dim, in_dim)), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty((out_dim)), requires_grad=True)

        self.quant_weight_dtype = get_dtype_from_string(weight_quant_dtype)
        self.quant_activation_dtype = get_dtype_from_string(activation_quant_dtype)

        self.weight_quantizer = FakeWeightQuantizer(bits, self.quant_weight_dtype, weight_quant_method)
        self.input_quantizer = FakeActivationQuantizer(bits, self.quant_activation_dtype, activation_quant_method, channel_axis=2)


    def __call__(self, x):
        raise NotImplementedError


class QuantizedLinear(LinearBase):

    def __init__(self, in_dim, out_dim, 
        bits = 8,
        weight_quant_dtype = 'quint8',
        weight_quant_method = 'per_tensor',
        activation_quant_dtype = 'quint8',
        activation_quant_method = 'per_tensor',
        bias = True
    ):
        super().__init__(in_dim, out_dim, 
            bits,
            weight_quant_dtype,
            weight_quant_method,
            activation_quant_dtype,
            activation_quant_method,
            bias
        )

    def __call__(self, x):
        return F.linear(self.input_quantizer(x), self.weight_quantizer(self.weight), self.bias)


class QuantizedLinearRelu(LinearBase):

    def __init__(self, in_dim, out_dim, 
        bits = 8,
        weight_quant_dtype = 'quint8',
        weight_quant_method = 'per_tensor',
        activation_quant_dtype = 'quint8',
        activation_quant_method = 'per_tensor',
        bias = True
    ):
        super().__init__(in_dim, out_dim, 
            bits,
            weight_quant_dtype,
            weight_quant_method,
            activation_quant_dtype,
            activation_quant_method,
            bias
        )

    def __call__(self, x):
        return F.relu(F.linear(self.input_quantizer(x), self.weight_quantizer(self.weight), self.bias))