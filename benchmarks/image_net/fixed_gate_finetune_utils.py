# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from time import time

import math
import torch
from torch import nn
from torch.nn import functional as F

from anonymized_compression_package.quantization.quantized_folded_bn import (
    BatchNormQScheme,
)
from anonymized_compression_package.quantization.straight_through import (
    BayesianBitsQuantizer,
    PACTQuantizer,
    QuantizationHijacker,
)
from benchmarks.image_net.mixed_precision_utils import get_bw, print_and_log, l2_loss


class QConvReLU(QuantizationHijacker, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(QConvReLU, self).__init__(*args, activation="relu", **kwargs)


def static_fold_bn(submod):
    assert nn.Conv2d in submod.__class__.mro()

    bias = submod.beta - submod.gamma * submod.running_mean / submod.sigma
    weight_folded = (
        submod.weight.transpose(0, -1) * submod.gamma / submod.sigma
    ).transpose(0, -1)

    if submod.bias is not None:
        bias += submod.bias

    new_conv = QConvReLU(
        in_channels=submod.in_channels,
        out_channels=submod.out_channels,
        kernel_size=submod.kernel_size,
        stride=submod.stride,
        padding=submod.padding,
        dilation=submod.dilation,
        groups=submod.groups,
        bias=submod.bias is not False,
        padding_mode=submod.padding_mode,
    )

    new_conv.weight.data.copy_(weight_folded.data)
    new_conv.bias.data.copy_(bias.data)

    new_conv.activation_function = submod.activation_function
    new_conv.activation_quantizer = submod.activation_quantizer
    new_conv.weight_quantizer = submod.weight_quantizer
    new_conv._quant_w = submod._quant_w
    new_conv._quant_a = submod._quant_a

    return new_conv


def get_prune_mask(module):
    gates = None
    module = module.quantizer
    if module.include_pruning:
        # prune_mask = torch.ones((module.gamma_2.shape[0], 1, 1, 1)).to(module.gamma_2.device)
        offset = math.log(-module.hc_gamma / module.hc_zeta) * module.hc_beta
        gates = (torch.sigmoid(offset - module.gamma_2) < module.hc_thres).float()
        # prune_mask[:, 0, 0, 0].copy_(gates)
    return gates


def compare_quantizers(old, new):
    import numpy as np

    signed, x_max = old.signed, old.x_max
    x_min = -int(signed) * x_max

    if old.gamma_2 is not None:
        x = (
            torch.rand((old.gamma_2.shape[0], 1000, 1, 1)).cuda() * (x_max - x_min)
            + x_min
        )
    else:
        x = torch.rand(10000).cuda() * (x_max - x_min) + x_min
    # print(x.shape)
    # old.eval()
    # print(old.x_max.item(), new.x_max.item())
    # print(old.compute_base_scale(), new.compute_base_scale())

    # old_q = old(x).data.cpu().numpy()
    # new_q = new(x).data.cpu().numpy()

    # ac = np.allclose(old_q, new_q, atol=1e-2, rtol=1e-2)
    # print('Old and new all close:', ac)
    # print(np.unique(old_q).shape, np.unique(new_q).shape)
    # assert ac
    # print('-------------------------------------------')


def replace_bb_quantizers(rn, force_no_prune=False):
    for name, module in rn.named_modules():
        if hasattr(module, "quantizer") and isinstance(
            module.quantizer, BayesianBitsQuantizer
        ):
            n_bits = get_bw(module)
            # print(name, n_bits)
            prune_mask = get_prune_mask(module)
            # if prune_mask is not None:
            #     print((prune_mask.sum() / prune_mask.squeeze().shape[0] * 100).item())
            if force_no_prune:
                prune_mask = None
            new_quantizer = PACTQuantizer(
                n_bits,
                module.quantizer.x_max.item(),
                module.quantizer.signed,
                True,
                prune_mask=prune_mask,
            )

            compare_quantizers(module.quantizer, new_quantizer)

            module.quantizer = new_quantizer
            module.method = "pact_only"
            module.n_bits = n_bits


def static_fold_bn_convs(rn):
    for module in rn.modules():
        for name, submod in module.named_children():
            if isinstance(submod, BatchNormQScheme):
                new_conv = static_fold_bn(submod)
                setattr(module, name, new_conv)


def train_finetune_epoch(
    rn,
    train_loader,
    optimizers,
    epoch,
    max_epoch,
    logfile=None,
    stopafter=None,
    schedulers=None,
    weight_decay=0.0,
):
    rn = rn.train()

    avg_loss = avg_ce_loss = avg_c_loss = 0.0
    tstart = time()
    for idx, (x, y) in enumerate(train_loader):
        if stopafter is not None and idx > stopafter:
            break
        t0 = time()
        rn.zero_grad()

        for opt in optimizers:
            opt.zero_grad()

        x, y = x.cuda(), y.cuda()
        yhat = rn(x)

        loss = F.cross_entropy(yhat, y)
        if weight_decay > 0:
            loss = loss + weight_decay * l2_loss(rn)
        loss.backward()

        # print('\n'.join([str(p.grad.flatten()[0]).replace('\n', ' ') for p in quant_parameters]))

        for opt in optimizers:
            opt.step()

        avg_loss = 1 / (idx + 1) * float(loss.item()) + idx / (idx + 1) * avg_loss
        t = time()

        batch_time = t - t0
        epoch_time = t - tstart
        avg_time = epoch_time / (idx + 1)
        remaining_time = (len(train_loader) - idx - 1) * avg_time

        s = (
            "[Epoch: {:>4}/{:>4}]: {:>5} / {}; {:>7.2f}; {:>7.2f}; "
            "{:>7.2f}; [{:>6.2f}s; {:>6.2f}s; {:>6.2f}s]".format(
                epoch,
                max_epoch,
                idx + 1,
                len(train_loader),
                avg_loss,
                avg_ce_loss,
                avg_c_loss,
                epoch_time,
                batch_time,
                remaining_time,
            )
        )
        notlast = (idx + 1) < len(train_loader)
        print_and_log(
            s,
            flush=True,
            end=("\r" if notlast else "\n"),
            logfile=(None if notlast else logfile),
        )

        if schedulers:
            for s in schedulers:
                s.step()

        # torch.cuda.empty_cache()


def get_finetune_optimizers(
    experiment, learning_rate, learning_rate_s, optimizer, optimizer_s, rn
):
    assert experiment == "imagenet"
    assert optimizer_s is not None
    assert (optimizer_s is None) == (learning_rate_s is None)

    parameters, scale_parameters = [], []
    for n, p in rn.named_parameters():
        if any([s in n for s in ["x_min", "x_max", "s_2"]]):
            scale_parameters.append(p)
        else:
            parameters.append(p)

    all_parameters, lr_list, optim_list, extra_args_list = [], [], [], []

    lr_list.append(learning_rate)
    optim_list.append(optimizer)
    extra_args_list.append({"weight_decay": 0.0})
    all_parameters.append(parameters)

    all_parameters.append(scale_parameters)
    lr_list.append(learning_rate_s)
    optim_list.append(optimizer_s)
    extra_args_list.append({"weight_decay": 0.0})

    for optim, extra_args in zip(optim_list, extra_args_list):
        if optim == "SGD":
            extra_args["momentum"] = 0.9
            extra_args["nesterov"] = True

    optimizers = [
        getattr(torch.optim, optimizer_str)(params, lr, **extra_args)
        for optimizer_str, params, lr, extra_args in zip(
            optim_list, all_parameters, lr_list, extra_args_list
        )
    ]

    return optimizers


def get_finetune_schedulers(optimizers, lr_schedule, epochs, train_loader=None):
    schedulers = None
    if lr_schedule.startswith("multistep"):
        epochs = [int(s) for s in lr_schedule.split(":")[1:]]
        schedulers = [
            torch.optim.lr_scheduler.MultiStepLR(opt, epochs) for opt in optimizers
        ]
    elif lr_schedule.startswith("cosine"):
        eta_min = float(lr_schedule.split(":")[1])
        if train_loader is not None:
            epochs = epochs * len(train_loader)
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers[0], epochs, eta_min=eta_min
            )
        ] + [
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
            for opt in optimizers[1:]
        ]

    return schedulers
