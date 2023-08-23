
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

def get_dtype_from_string(dtype_string):
    if dtype_string == 'quint8':
        return torch.quint8
    elif dtype_string == 'qint8':
        return torch.qint8
    else:
        raise ValueError(f'Unrecognized dtype string {dtype_string}!')

def is_quantized(x):
    return x.is_quantized()