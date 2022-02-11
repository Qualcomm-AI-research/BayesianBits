# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from torch import nn

from anonymized_compression_package.quantization.quantized_folded_bn import (
    BatchNormQScheme,
)
from anonymized_compression_package.quantization.straight_through import (
    QuantizationHijacker,
)


class QuantizedModule(nn.Module):
    """
    Parent class for a quantized module. It adds the basic functionality of switching the module
    between quantized and full precision mode. It also defines the cached parameters and handles
    the reset of the cache properly.

    """

    def __init__(
        self,
        *args,
        method="asymmetric",
        n_bits=8,
        n_bits_act=None,
        act_momentum=0.1,
        percentile=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.method = method
        self.n_bits = n_bits
        self.n_bits_act = n_bits_act or n_bits
        self.act_momentum = act_momentum
        self.percentile = percentile

        self.cached_params = None
        self.quant_params = None
        self._quant_w = True
        self._quant_a = True

    def quantized_weights(self):
        self.cached_params = None
        self._quant_w = True

    def full_precision_weights(self):
        self.cached_params = None
        self._quant_w = False

    def quantized_acts(self):
        self._quant_a = True

    def full_precision_acts(self):
        self._quant_a = False

    def quantized(self):
        self.quantized_weights()
        self.quantized_acts()

    def full_precision(self):
        self.full_precision_weights()
        self.full_precision_acts()

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.cached_params = None
        return self


class QuantizedModel(nn.Module):
    """
    Parent class for a quantized model. This allows you to have convenience functions to put the
    whole model into quantization or full precision or to freeze BN. Otherwise it does not add any
    further functionality, so it is not a necessity that a quantized model uses this class.

    """

    def __getattribute__(self, name):
        if name.startswith("quantized") or name.startswith("full_precision"):

            def the_func(layer):
                if isinstance(layer, (QuantizationHijacker, QuantizedModule)):
                    layer.__getattribute__(name)()

        elif name.endswith("freeze_batch_norm"):

            def the_func(layer):
                if isinstance(layer, BatchNormQScheme):
                    layer.__getattribute__(name)()

        else:
            return super(QuantizedModel, self).__getattribute__(name)

        return lambda: self.apply(the_func)
