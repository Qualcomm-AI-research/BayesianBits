# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.


import torch
from torch import nn

from anonymized_compression_package.quantization.base_classes import QuantizedModel
from anonymized_compression_package.quantization.straight_through import (
    QuantizationHijacker,
)


class QConvReLU(QuantizationHijacker, nn.Conv2d):
    def __init__(self, *args, quant_params, **kwargs):
        super(QConvReLU, self).__init__(
            *args, activation="relu", **quant_params, **kwargs
        )


class QLinearReLU(QuantizationHijacker, nn.Linear):
    def __init__(self, *args, quant_params, **kwargs):
        super(QLinearReLU, self).__init__(
            *args, activation="relu", **quant_params, **kwargs
        )


class QuantLinear(QuantizationHijacker, nn.Linear):
    def __init__(self, *args, quant_params, **kwargs):
        super(QuantLinear, self).__init__(*args, **quant_params, **kwargs)


class LeNet5Quantized(QuantizedModel):
    def __init__(self, **quant_params):
        super(LeNet5Quantized, self).__init__()

        features = nn.Sequential(
            QConvReLU(1, 32, 5, quant_params=quant_params),
            nn.MaxPool2d(2),
            QConvReLU(32, 64, 5, quant_params=quant_params),
            nn.MaxPool2d(2),
        )
        self.features = features

        self.classifier = nn.Sequential(
            QLinearReLU(1024, 512, quant_params=quant_params),
            QuantLinear(512, 10, quant_params=quant_params),
        )

    def set_full_precision_logits(self):
        print("Using full precision logits.")
        for name, module in self.classifier.named_modules():
            if isinstance(module, QuantLinear):
                module.full_precision_acts()
                if module.weight_quantizer.include_pruning:
                    module.weight_quantizer.include_pruning = False

    def forward(self, input):
        h = self.features(input)
        h = h.view(h.size(0), -1)
        return self.classifier(h)
