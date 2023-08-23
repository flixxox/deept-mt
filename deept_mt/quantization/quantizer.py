
import torch
import torch.nn as nn
import torch.ao.quantization as torchquant

from deept.util.globals import Settings

from deept_mt.quantization.quant_utils import get_quantization_range_from_bits


class MinMaxFakeWeightQuantizer(nn.Module):

    """
    During training:
        WeightQuantizer resets the qparams in every call 
        since the weight may have changed.
    After training:
        In calibration mode or not, the weights and 
        thus qparams are fixed.
    """

    def __init__(self, bits, dtype):
        super().__init__()

        self.dtype = dtype
        self.quant_min, self.quant_max = get_quantization_range_from_bits(bits, dtype)

        self.observer = torchquant.observer.MinMaxObserver(
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            dtype=dtype
        )
        
        self.args = None
    
    def __call__(self, x):
        
        if Settings.is_training():
            self.observer.reset_min_max_vals()
            x = self.observer(x)
            args = self.observer.calculate_qparams()
        else:
            if self.args is None:
                self.observer(x)
                args = self.observer.calculate_qparams()
            else:
                args = self.args

        x = torch.fake_quantize_per_tensor_affine(x, *args, self.quant_min, self.quant_max)

        return x


class MinMaxFakeActivationQuantizer(nn.Module):

    def __init__(self, bits, dtype):
        super().__init__()

        self.dtype = dtype
        self.quant_min, self.quant_max = get_quantization_range_from_bits(bits, dtype)

        self.observer = torchquant.observer.MinMaxObserver(
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            dtype=dtype
        )
    
    def __call__(self, x):

        x = self.observer(x)
        scale, zp = self.observer.calculate_qparams()

        x = torch.fake_quantize_per_tensor_affine(x, scale, zp, self.quant_min, self.quant_max)

        return x