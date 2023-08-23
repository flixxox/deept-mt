
import torch

def get_quantization_range_from_bits(bits, dtype):

    if dtype == torch.qint8:
        quant_min = -(2**(bits-1))
        quant_max = (2**(bits-1))-1

    elif dtype == torch.quint8:
        quant_min = 0
        quant_max = (2**bits)-1

    else:
        raise ValueError(f'Unrecognized dtype {dtype}')

    return quant_min, quant_max

def is_quantized(x):
    return x.is_quantized()

def create_activation_quantizer(bits, argument):
    from deept_mt.quantization.quantizer import (
        MinMaxFakeActivationQuantizer
    )

    if argument == 'MinMax':
        return MinMaxFakeActivationQuantizer(bits, torch.quint8)
    else:
        raise ValueError(f'Unrecognized activation quantizer {argument}!')