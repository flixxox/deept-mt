
import torch
import torch.nn as nn
import torch.ao.quantization as torchquant

from deept.util.globals import Settings

from deept_mt.quantization.quant_utils import get_quantization_range_from_bits


class FakeWeightQuantizer(nn.Module):

    """
    During training:
        WeightQuantizer resets the qparams in every call 
        since the weight may have changed.
    After training:
        In calibration mode or not, the weights and 
        thus qparams are fixed.
    """

    def __init__(self, bits, dtype, method):
        super().__init__()

        self.dtype = dtype
        self.quant_min, self.quant_max = get_quantization_range_from_bits(bits, dtype)
        self.args = None

        self.quant_fn, self.observer = self.__get_quant_fn_and_observer_for_method(method)

    def __get_quant_fn_and_observer_for_method(self, method):

        if method == 'per_tensor':

            quant_fn = torch.fake_quantize_per_tensor_affine

            observer = torchquant.observer.MinMaxObserver(
                quant_min=self.quant_min,
                quant_max=self.quant_max,
                dtype=self.dtype,
                reduce_range=False
            )
        
        else:
            raise ValueError(f'Unrecognized quantization method {method}!')

        return quant_fn, observer
    
    def __call__(self, x):
        
        if Settings.is_training():
            self.observer.reset_min_max_vals()
            x = self.observer(x)
            args = self.observer.calculate_qparams()
        else:
            if self.args is None:
                x = self.observer(x)
                args = self.observer.calculate_qparams()
            else:
                args = self.args

        x = self.quant_fn(x, *args, self.quant_min, self.quant_max)

        return x


class FakeActivationQuantizer(nn.Module):

    def __init__(self, bits, dtype, method,
        channel_axis = None    
    ):
        super().__init__()

        self.dtype = dtype
        self.quant_min, self.quant_max = get_quantization_range_from_bits(bits, dtype)
        self.channel_axis = channel_axis

        self.quant_fn, self.static_args, self.observer = self.__get_quant_fn_and_observer_for_method(method)

    def __get_quant_fn_and_observer_for_method(self, method):

        if method == 'per_tensor':

            quant_fn = torch.fake_quantize_per_tensor_affine

            static_args = [self.quant_min, self.quant_max]

            observer = torchquant.observer.MovingAverageMinMaxObserver(
                averaging_constant=0.01,
                quant_min=self.quant_min,
                quant_max=self.quant_max,
                dtype=self.dtype,
                reduce_range=False
            )

        elif method == 'per_channel':

            assert self.channel_axis is not None

            quant_fn = torch.fake_quantize_per_channel_affine

            static_args = [self.channel_axis, self.quant_min, self.quant_max]

            observer = torchquant.observer.MovingAveragePerChannelMinMaxObserver(
                averaging_constant=0.01,
                quant_min=self.quant_min,
                quant_max=self.quant_max,
                dtype=self.dtype,
                ch_axis=self.channel_axis,
                reduce_range=False
            )
        
        else:
            raise ValueError(f'Unrecognized quantization method {method}!')

        return quant_fn, static_args, observer
    
    def __call__(self, x):

        x = self.observer(x)
        args = self.observer.calculate_qparams()

        x = self.quant_fn(x, *args, *self.static_args)

        return x