# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
from torch import nn

from benchmarks.image_net.graph_utils import inspect_graph
from anonymized_compression_package.quantization.straight_through import (
    Quantizer,
    QuantizationHijacker,
)
from benchmarks.cifar.cifar_models import LambdaLayer


def mobilenet_v2_quantizer_groups(mobilenet_v2_quantized):
    anf = "features.{}.activation_quantizer"
    anc = "features.{}.conv.{}.activation_quantizer"
    wnc = "features.{}.conv.{}.weight_quantizer"

    groups = [("features.0.weight_quantizer",)]
    last_quantizer = "features.0.activation_quantizer"

    for f in range(1, len(mobilenet_v2_quantized.features) - 2):
        for i in range(len(mobilenet_v2_quantized.features[f].conv)):
            groups.append((last_quantizer, wnc.format(f, i)))
            last_quantizer = anc.format(f, i)
        if mobilenet_v2_quantized.features[f].use_res_connect:
            last_quantizer = anf.format(f)

    groups.append((last_quantizer, "features.18.weight_quantizer"))
    groups.append(("features.18.activation_quantizer", "classifier.1.weight_quantizer"))

    qdict = {}
    for name, module in mobilenet_v2_quantized.named_modules():
        if isinstance(module, Quantizer):
            qdict[name] = module

    qgroups = [(qgroup, [qdict[q] for q in qgroup]) for qgroup in groups]

    return qgroups


def resnet18_quantizer_groups(resnet18_quantized, model):
    block_act_q = "features.{}.{}.activation_quantizer"
    groups = [("features.0.weight_quantizer",)]
    last_quantizer = "features.0.activation_quantizer"

    start = 2 if model == "resnet18" else 1

    for i, superblock in enumerate(resnet18_quantized.features[start:], start=start):
        for j, block in enumerate(superblock):
            start = 0
            if block.downsample is not None and not (
                isinstance(block.downsample, LambdaLayer)
                or (
                    isinstance(block.downsample, nn.Sequential)
                    and len(block.downsample) == 0
                )
            ):
                groups.append(
                    (
                        last_quantizer,
                        "features.{}.{}.features.0.weight_quantizer".format(i, j),
                        "features.{}.{}.downsample.0.weight_quantizer".format(i, j),
                    )
                )
                last_quantizer = "features.{}.{}.features.{}.activation_quantizer".format(
                    i, j, 0
                )
                start = 1
            for k, bnqconv in enumerate(block.features[start:], start=start):
                groups.append(
                    (
                        last_quantizer,
                        "features.{}.{}.features.{}.weight_quantizer".format(i, j, k),
                    )
                )
                last_quantizer = "features.{}.{}.features.{}.activation_quantizer".format(
                    i, j, k
                )
            last_quantizer = block_act_q.format(i, j)

    groups.append((last_quantizer, "fc.weight_quantizer"))

    module_dict = {name: module for name, module in resnet18_quantized.named_modules()}
    qgroups = [(qgroup, [module_dict[q] for q in qgroup]) for qgroup in groups]

    return qgroups


def vgg_quantizer_groups(rn):
    last_act_quantizer, last_act_quant_name = None, None
    qgroups = []
    for n, m in rn.named_modules():
        if not isinstance(m, QuantizationHijacker):
            continue

        key, val = tuple(), []
        if last_act_quantizer is not None:
            key += (last_act_quant_name,)
            val += [last_act_quantizer]
        key += (n + ".weight_quantizer",)
        val += [m.weight_quantizer]

        qgroups.append((key, val))
        last_act_quantizer = m.activation_quantizer
        last_act_quant_name = n + ".activation_quantizer"

    return qgroups


def get_prev_layers(model, rn):
    if isinstance(rn, torch.nn.DataParallel):
        rn = rn.module
    prev_layers = {}
    prev_layer = None

    if model in ["vggquantized", "lenet5quantized"]:
        for n, m in rn.named_modules():
            if not isinstance(m, QuantizationHijacker):
                continue
            prev_layers[n + ".weight_quantizer"] = (
                [prev_layer] if prev_layer is not None else []
            )
            prev_layer = m.weight_quantizer
        return prev_layers

    elif model == "resnet18":
        for n, m in rn.named_modules():
            if not isinstance(m, QuantizationHijacker):
                continue
            if n.endswith("features.1"):
                prev_layers[n + ".weight_quantizer"] = (
                    [] if prev_layer is None else [prev_layer]
                )
            else:
                prev_layers[n + ".weight_quantizer"] = []

            prev_layer = m.weight_quantizer
        return prev_layers

    elif model == "mobilenetv2":
        prev_layers["features.1.conv.0.weight_quantzizer"] = [
            rn.features[0].weight_quantizer
        ]
        n = "features.{}.conv.2.weight_quantizer"
        for i in range(2, 18):
            prev_layers[n.format(i, 2)] = rn.features[i].conv[0].weight_quantizer

    else:
        return None


def get_per_layer_macs(rn, input_shape, replace_names=None):
    replace_names = replace_names or []
    i = inspect_graph(rn, input_shape=input_shape)
    per_layer_macs = {}
    for n in i.yield_node_names():
        stats = i.get_node_stats(n)
        if stats["macs"] == 0:
            continue
        k = n[:]
        for before, after in replace_names:
            k = k.replace(before, after)
        per_layer_macs[k] = stats["macs"]

    return per_layer_macs


def get_groups_and_macs(rn, experiment, model):
    replace_names = [(model + ".", "")]
    if experiment == "cifar10":
        input_shape = [1, 3, 32, 32]
        replace_names += [("features.0", "features.0.0")]
    elif experiment == "mnist":
        input_shape = [1, 1, 28, 28]
        replace_names += [("features.0", "features.0.0")]
    elif experiment == "imagenet":
        input_shape = [1, 3, 223, 223]
        if model == "resnet18":
            replace_names += [("quantizedresnet.", "")]
        elif model == "mobilenetv2":
            replace_names += [("quantized", "")]
    else:
        raise ValueError("Invalid experiment:", experiment)

    per_layer_macs = get_per_layer_macs(rn, input_shape, replace_names)

    if model == "vggquantized":
        groups = vgg_quantizer_groups(rn)
    elif model in ["resnet18", "resnet56_quantized"]:
        groups = resnet18_quantizer_groups(rn, model)
    elif model == "mobilenetv2":
        groups = mobilenet_v2_quantizer_groups(rn)
    elif model == "lenet5quantized":
        groups = vgg_quantizer_groups(rn)
    else:
        raise ValueError("Invalid model:", model)

    return per_layer_macs, groups


def fix_name(n, model=None):
    n = n.replace(".weight_quantizer", "")
    if model != "mobilenetv2":
        n = n.replace(".conv.2", ".conv.6").replace(".conv.1", ".conv.3")
    if model in ["vggquantized", "lenet5quantized"]:
        if n == "features.0":
            n += ".0"
    if model == "resnet56_quantized":
        n = "quantizedresnet." + n
        if n.endswith("features.0"):
            n += ".0"

    return n


def get_bw(quantizer, prune_only=False):
    if prune_only:
        if quantizer.quantizer is None or not (
            quantizer.quantizer.fixed_8bit or quantizer.quantizer.fixed_4bit
        ):
            return 16
    if "bayesian_bits" in quantizer.method:
        assert quantizer.quantizer is not None
        if quantizer.quantizer.fixed_8bit:
            return 8
        elif quantizer.quantizer.fixed_4bit:
            return 4

        train = quantizer.quantizer.training
        if train:
            quantizer.quantizer.eval()

        fix_type = lambda g: int(g.item()) if isinstance(g, torch.Tensor) else int(g)
        q4, q8, q16, q32 = [fix_type(g) for g in quantizer.quantizer.get_gates()[1:]]
        n = 1 + q4 + q4 * (q8 + q8 * (q16 + q16 * q32))

        if train:
            quantizer.quantizer.train()

        return int(2 ** n)
    else:
        return quantizer.n_bits


def print_and_log(*s, logfile=None, **kwargs):
    print(*s, **kwargs)
    if logfile:
        print(*s, **kwargs, file=open(logfile, "a"))


def l2_loss(rn):
    regularizer = 0.0
    for name, param in rn.named_parameters():
        if "quantizer" not in name:
            regularizer += 0.5 * torch.pow(param, 2).sum()
    return regularizer
