# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
from collections import OrderedDict

import torch.nn as nn
import torch
import math
import re

from anonymized_compression_package.quantization.base_classes import QuantizedModel
from anonymized_compression_package.quantization.quantized_folded_bn import (
    BatchNormQScheme,
    QuantizationHijacker,
)

quant_params = {"method": "asymmetric", "n_bits": 8, "per_channel_weights": False}


class BNQConv(BatchNormQScheme, nn.Conv2d):
    def __init__(self, *args, activation=None, **kwargs):
        super(BNQConv, self).__init__(
            *args, activation=activation, **quant_params, **kwargs
        )


class BNQConvReLU6(BNQConv):
    def __init__(self, *args, **kwargs):
        super(BNQConvReLU6, self).__init__(*args, activation="relu6", **kwargs)


class QuantLinear(QuantizationHijacker, nn.Linear):
    def __init__(self, *args, **kwargs):
        super(QuantLinear, self).__init__(*args, **quant_params, **kwargs)


def conv_bn(inp, oup, stride):
    return BNQConvReLU6(inp, oup, 3, stride=stride, padding=1, bias=False)


def conv_1x1_bn(inp, oup):
    return BNQConvReLU6(inp, oup, 1, stride=1, padding=0, bias=False)


class QuantizedInvertedResidual(QuantizationHijacker, nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        # TODO remove inheritance from hijacker
        super(QuantizedInvertedResidual, self).__init__(**quant_params)
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                BNQConvReLU6(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                # pw-linear
                BNQConv(
                    hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False
                ),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                BNQConvReLU6(
                    inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False
                ),
                # dw
                BNQConvReLU6(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                # pw-linear
                BNQConv(
                    hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False
                ),
            )

    def forward(self, x):
        if self.use_res_connect:
            result = x + self.conv(x)
            return self.quantize_activations(result)
        else:
            return self.conv(x)  # quantization happens in convolutions


class QuantizedMobileNetV2(QuantizedModel):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.0, dropout=0.0):
        super(QuantizedMobileNetV2, self).__init__()
        block = QuantizedInvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = (
            int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        )
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(input_channel, output_channel, s, expand_ratio=t)
                    )
                else:
                    self.features.append(
                        block(input_channel, output_channel, 1, expand_ratio=t)
                    )
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # TODO: Quantize?
        self.features.append(nn.AvgPool2d(input_size // 32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), QuantLinear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def set_full_precision_logits(self):
        print("Using full precision logits.")
        for name, module in self.classifier.named_modules():
            if isinstance(module, QuantLinear):
                module.full_precision_acts()
                if module.weight_quantizer.include_pruning:
                    module.weight_quantizer.include_pruning = False

    def forward(self, x):
        x = self.features(x)
        x = x.view((x.shape[0], -1))
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def load_model(model, model_path):
    Warning(
        "There are known issues with loading a model and then fine-tuning in that it doesn't work. "
        "Evaluation with a preloaded model has no problems."
    )
    state_dict = torch.load(model_path)
    new_dict = OrderedDict()
    sd_keys = list(state_dict.keys())

    def replace_conv_bn_module(i, infix):
        old_key = sd_keys[i]
        new_key = re.sub(r"\.[0-9]\.weight", ".{}weight".format(infix), old_key)
        new_dict[new_key.replace("module.", "")] = state_dict[old_key]

        old_key = sd_keys[i + 1]
        new_key = re.sub(r"\.[0-9]\.weight", ".{}gamma".format(infix), old_key)
        new_dict[new_key.replace("module.", "")] = state_dict[old_key]

        old_key = sd_keys[i + 2]
        new_key = re.sub(r"\.[0-9]\.bias", ".{}beta".format(infix), old_key)
        new_dict[new_key.replace("module.", "")] = state_dict[old_key]

        old_key = sd_keys[i + 3]
        new_key = re.sub(
            r"\.[0-9]\.running_mean", ".{}running_mean".format(infix), old_key
        )
        new_dict[new_key.replace("module.", "")] = state_dict[old_key]

        old_key = sd_keys[i + 4]
        new_key = re.sub(
            r"\.[0-9]\.running_var", ".{}running_var".format(infix), old_key
        )
        new_dict[new_key.replace("module.", "")] = state_dict[old_key]

    replace_conv_bn_module(0, "")

    if "num_batches_tracked" in sd_keys[5]:
        pars = 6
    else:
        pars = 5

    last_layer = None
    for i in range(pars, len(sd_keys) - (pars + 2), pars):
        # account for weird keys in Harris' pre-trained mobile net params
        l = (
            len("module.features.")
            if sd_keys[i].startswith("module.")
            else len("features.")
        )
        e = sd_keys[i].find(".", l)
        layer = sd_keys[i][l:e]
        layer_module = 0 if layer != last_layer else layer_module + 1
        last_layer = layer
        infix = "{}.".format(layer_module)

        replace_conv_bn_module(i, infix)

    replace_conv_bn_module(len(sd_keys) - (pars + 2), "")

    new_dict[sd_keys[-2].replace("module.", "")] = state_dict[sd_keys[-2]]
    new_dict[sd_keys[-1].replace("module.", "")] = state_dict[sd_keys[-1]]

    model.load_state_dict(new_dict)


def mobilenetv2_quantized(pretrained=False, qparams=None, **kwargs):
    """Constructs a MobileNetV2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        model_file (str):  If pretrained=True, this path is used to find/load the model file

    """
    if qparams:
        quant_params.update(qparams)
        print("New quantization parameters:", quant_params)
    model = QuantizedMobileNetV2()
    if pretrained:
        model_path = kwargs.get(
            "model_path",
            # '/prj/neo_lv/scratch/compression/model_files/mobilenetv2/model_model_120.pth'
            # '/local/mnt/workspace/model_model_120.pth'
            # 'mobilenetv2_tony.pth.tar'
        )
        load_model(model, model_path)
    # For some reason the model is not by default quantized
    model.quantized()
    return model
