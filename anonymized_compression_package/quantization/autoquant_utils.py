# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.


import copy

from torch import nn

from anonymized_compression_package.quantization.straight_through import (
    QuantizationHijacker,
)
from anonymized_compression_package.quantization.quantized_folded_bn import (
    BatchNormQScheme,
)


class QuantConv(QuantizationHijacker, nn.Conv2d):
    def __init__(self, *args, activation=None, **kwargs):
        super(QuantConv, self).__init__(*args, activation=activation, **kwargs)


class QuantLinear(QuantizationHijacker, nn.Linear):
    def __init__(self, *args, **kwargs):
        super(QuantLinear, self).__init__(*args, **kwargs)


class BNQConv(BatchNormQScheme, nn.Conv2d):
    def __init__(self, *args, activation=None, **kwargs):
        super(BNQConv, self).__init__(*args, activation=activation, **kwargs)


class BNQLinear(BatchNormQScheme, nn.Linear):
    def __init__(self, *args, activation=None, **kwargs):
        super(BNQLinear, self).__init__(*args, activation=activation, **kwargs)


class IdentityModule(nn.Module):
    def forward(self, *args):
        return args[0] if len(args) == 1 else args


non_bn_module_map = {nn.Conv2d: QuantConv, nn.Linear: QuantLinear}
bn_module_map = {nn.Conv2d: BNQConv, nn.Linear: BNQLinear}
act_map = {
    nn.ReLU: "relu",
    nn.ReLU6: "relu6",
    nn.Hardtanh: "hardtanh",
    nn.Sigmoid: "sigmoid",
}


def next_bn(module, i):
    return len(module) > i + 1 and isinstance(
        module[i + 1], (nn.BatchNorm2d, nn.BatchNorm1d)
    )


def get_act(module, i):
    result, act_idx = None, None
    for i in range(i + 1, len(module)):
        if type(module[i]) in bn_module_map:
            break
        if type(module[i]) in act_map:
            result = act_map[type(module[i])]
            act_idx = i
            break

    return result, act_idx


def get_conv_args(module):
    args = dict(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
    )
    return args


def get_linear_args(module):
    args = dict(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None,
    )
    return args


def get_module_args(mod, act):
    if isinstance(mod, nn.Conv2d):
        kwargs = get_conv_args(mod)
    elif isinstance(mod, nn.Linear):
        kwargs = get_linear_args(mod)
    else:
        raise ValueError

    kwargs["activation"] = act

    return kwargs


def fold_bn(module, i, **quant_params):
    bn = next_bn(module, i)
    act, act_idx = get_act(module, i)
    modmap = bn_module_map if bn else non_bn_module_map
    modtype = modmap[type(module[i])]

    kwargs = get_module_args(module[i], act)
    new_module = modtype(**kwargs, **quant_params)
    new_module.weight.data = module[i].weight.data.clone()

    if bn:
        new_module.gamma.data = module[i + 1].weight.data.clone()
        new_module.beta.data = module[i + 1].bias.data.clone()
        new_module.running_mean.data = module[i + 1].running_mean.data.clone()
        new_module.running_var.data = module[i + 1].running_var.data.clone()
        if module[i].bias is not None:
            new_module.running_mean.data -= module[i].bias.data
            print("Warning: bias in conv/linear before batch normalization.")

    elif module[i].bias is not None:
        new_module.bias.data = module[i].bias.data.clone()

    return new_module, i + int(bool(act)) + int(bn) + 1


def quantize_sequential(model, specials=None, **quant_params):
    specials = specials or dict()

    i = 0
    quant_modules = []
    while i < len(model):
        if isinstance(model[i], QuantizationHijacker):
            quant_modules.append(model[i])

        elif type(model[i]) in (nn.Conv2d, nn.Linear):
            new_module, new_i = fold_bn(model, i, **quant_params)
            quant_modules.append(new_module)
            i = new_i
            continue

        elif type(model[i]) in specials:
            quant_modules.append(specials[type(model[i])](model[i], **quant_params))

        else:
            quant_modules.append(
                quantize_model(model[i], specials=specials, **quant_params)
            )
        i += 1
    return nn.Sequential(*quant_modules)


def quantize_model(model, specials=None, **quant_params):
    specials = specials or dict()

    if isinstance(model, nn.Sequential):
        quant_model = quantize_sequential(model, specials, **quant_params)

    elif type(model) in specials:
        quant_model = specials[type(model)](model, **quant_params)

    elif type(model) in (nn.Conv2d, nn.Linear):
        # If we do isinstance() then we might run into issues with modules that inherit from
        # one of these classes, for whatever reason
        modtype = non_bn_module_map[type(model)]
        kwargs = get_module_args(model, None)
        quant_model = modtype(**kwargs, **quant_params)

        quant_model.weight.data = model.weight.data
        if model.bias is not None:
            quant_model.bias.data = model.bias.data

    else:
        # Unknown type, try to quantize all child modules
        quant_model = copy.copy(model)
        for name, module in quant_model._modules.items():
            new_model = quantize_model(module, specials=specials, **quant_params)
            if new_model is not None:
                setattr(model, name, new_model)

    return quant_model
