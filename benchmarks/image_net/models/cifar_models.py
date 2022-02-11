# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.


import torch
from torch import nn


from anonymized_compression_package.quantization.autoquant_utils import quantize_model
import torch.nn.functional as F

from anonymized_compression_package.quantization.base_classes import QuantizedModel
from anonymized_compression_package.quantization.straight_through import (
    QuantizationHijacker,
)

from anonymized_compression_package.quantization.quantized_folded_bn import (
    BatchNormQScheme,
)
from benchmarks.cifar.cifar_models import resnet56, BasicBlock, LambdaLayer


class BNQConvReLU(BatchNormQScheme, nn.Conv2d):
    def __init__(self, *args, quant_params, **kwargs):
        super(BNQConvReLU, self).__init__(
            *args, activation="relu", **quant_params, **kwargs
        )


class BNQConv(BatchNormQScheme, nn.Conv2d):
    def __init__(self, *args, quant_params, **kwargs):
        super(BNQConv, self).__init__(
            *args, activation="none", **quant_params, **kwargs
        )


class BNQLinearReLU(BatchNormQScheme, nn.Linear):
    def __init__(self, *args, quant_params, **kwargs):
        super(BNQLinearReLU, self).__init__(
            *args, activation="relu", **quant_params, **kwargs
        )


class QuantLinear(QuantizationHijacker, nn.Linear):
    def __init__(self, *args, quant_params, **kwargs):
        super(QuantLinear, self).__init__(*args, **quant_params, **kwargs)


conv_layer, convrelu_layer, linear_layer = BNQConv, BNQConvReLU, BNQLinearReLU


class VGGQuantized(QuantizedModel):
    def __init__(self, **quant_params):
        super(VGGQuantized, self).__init__()

        k = 1
        features = nn.Sequential(
            convrelu_layer(3, int(k * 128), 3, padding=1, quant_params=quant_params),
            convrelu_layer(
                int(k * 128), int(k * 128), 3, padding=1, quant_params=quant_params
            ),
            nn.MaxPool2d(2),
            convrelu_layer(
                int(k * 128), int(k * 256), 3, padding=1, quant_params=quant_params
            ),
            convrelu_layer(
                int(k * 256), int(k * 256), 3, padding=1, quant_params=quant_params
            ),
            nn.MaxPool2d(2),
            convrelu_layer(
                int(k * 256), int(k * 512), 3, padding=1, quant_params=quant_params
            ),
            convrelu_layer(
                int(k * 512), int(k * 512), 3, padding=1, quant_params=quant_params
            ),
            nn.MaxPool2d(2),
        )
        self.features = features

        input_feats = 8192
        self.classifier = nn.Sequential(
            linear_layer(input_feats, int(k * 1024), quant_params=quant_params),
            QuantLinear(int(k * 1024), 10, quant_params=quant_params),
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


default_quant_params = {
    "method": "asymmetric",
    "n_bits": 8,
    "per_channel_weights": False,
}


class QuantizedBlock(QuantizationHijacker):
    def __init__(self, block, **quant_params):
        super(QuantizedBlock, self).__init__(**quant_params)

        if isinstance(block, BasicBlock):
            features = nn.Sequential(
                block.conv1, block.bn1, nn.ReLU(), block.conv2, block.bn2
            )
        else:
            raise ValueError("Unknown block type: {}".format(type(BasicBlock)))

        self.features = quantize_model(features, **quant_params)
        if isinstance(block.shortcut, nn.Sequential):
            self.downsample = quantize_model(block.shortcut, **quant_params)
        elif isinstance(block.shortcut, LambdaLayer):
            self.downsample = block.shortcut
        else:
            raise ValueError(
                "Something weird is happening: {}".format(type(block.shortcut))
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.features(x)
        out += self.downsample(x)
        out = self.relu(out)

        return self.quantize_activations(out)


class QuantizedResNet(QuantizedModel):
    def __init__(self, resnet, **quant_params):
        super(QuantizedResNet, self).__init__()
        specials = {BasicBlock: QuantizedBlock}
        features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            nn.ReLU(),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

        self.features = quantize_model(features, specials=specials, **quant_params)

        self.fc = quantize_model(resnet.linear, **quant_params)

    def set_full_precision_logits(self):
        print("Using full precision logits.")
        for name, module in self.fc.named_modules():
            if isinstance(module, QuantLinear):
                module.full_precision_acts()
                # if module.weight_quantizer.include_pruning:
                module.weight_quantizer.include_pruning = False
                module.weight_quantizer.prune_only = False

    def forward(self, x):
        x = self.features(x)

        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet56_quantized(quant_params):
    """Constructs a quantized ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """
    model = resnet56()
    print("New quantization parameters:", quant_params)
    quant_model = QuantizedResNet(model, **quant_params)
    return quant_model
