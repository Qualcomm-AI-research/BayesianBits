# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.


import os
from time import time
import traceback
import warnings

import click
import numpy as np
import torch
from torch.nn import functional as F
import math

import socket

from benchmarks.image_net.fixed_gate_finetune_utils import (
    train_finetune_epoch,
    replace_bb_quantizers,
    static_fold_bn_convs,
    get_finetune_schedulers,
    get_finetune_optimizers,
)
from benchmarks.image_net.models.mobilenetv2.mobilenet_v2_quantized import (
    mobilenetv2_quantized,
)
from benchmarks.image_net.models.resnet_quantized import resnet18_quantized
from benchmarks.image_net.models.cifar_models import VGGQuantized, resnet56_quantized
from benchmarks.image_net.models.mnist_models import LeNet5Quantized
from benchmarks.image_net.mixed_precision_dataloader import ImageNetDataLoaders
from benchmarks.image_net.mixed_precision_utils import (
    get_groups_and_macs,
    fix_name,
    get_prev_layers,
    get_bw,
    print_and_log,
    l2_loss,
)

from benchmarks.cifar.cifar10_data_loader import Cifar10DataLoader
from benchmarks.mnist.mnist_data_loader import MnistDataLoader

from anonymized_compression_package.quantization.straight_through import (
    BayesianBitsQuantizer,
    PACTQuantizer,
)

from anonymized_compression_package.quantization.utils import (
    equalize_residual_block,
    equalize_pair,
    replace_relu6_quantized,
)


def gate_loss(rn):
    regularizer = 0.0
    for name, module in rn.named_modules():
        if isinstance(module, BayesianBitsQuantizer):
            regularizer += module.regularizer()
    return regularizer


def train_epoch(
    rn,
    train_loader,
    lmb,
    optimizers,
    epoch,
    max_epoch,
    logfile=None,
    all_quantized=True,
    stopafter=None,
    weight_decay=0.0,
    log_every=None,
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

        c_loss = gate_loss(rn) if all_quantized else torch.Tensor([0.0])
        ce_loss = F.cross_entropy(yhat, y)
        loss = ce_loss + lmb * c_loss
        if weight_decay > 0:
            loss = loss + weight_decay * l2_loss(rn)
        loss.backward()

        # print('\n'.join([str(p.grad.flatten()[0]).replace('\n', ' ') for p in quant_parameters]))

        for opt in optimizers:
            opt.step()

        avg_loss = 1 / (idx + 1) * float(loss.item()) + idx / (idx + 1) * avg_loss
        avg_ce_loss = (
            1 / (idx + 1) * float(ce_loss.item()) + idx / (idx + 1) * avg_ce_loss
        )
        avg_c_loss = 1 / (idx + 1) * float(c_loss) + idx / (idx + 1) * avg_c_loss

        t = time()

        batch_time = t - t0
        epoch_time = t - tstart
        avg_time = epoch_time / (idx + 1)
        remaining_time = (len(train_loader) - idx - 1) * avg_time

        if not log_every or ((idx + 1) % log_every) == 0:
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
    print_and_log(logfile=logfile)


def get_gate_sum(rn):
    # sum all gates so we can get the gradient w.r.t. each of the bernoulli parameters
    gate_sum = 0.0
    for name, m in rn.named_modules():
        if isinstance(m, BayesianBitsQuantizer):
            if m.gate_4 is not None:
                gate_sum += m.gate_4 + m.gate_8 + m.gate_16

    return gate_sum


def print_gammas(rn):
    maxlen = max([len(name) for name, _ in rn.named_modules()])
    template = "{{:>{}}}: {{:>7.3f}} {{:>7.3f}}".format(maxlen)
    for name, module in rn.named_modules():
        if isinstance(module, BayesianBitsQuantizer):
            print(template.format(name, module.gamma_8.item(), module.gamma_16.item()))


def set_rn_q(rn, weight_quant, act_quant):
    rn.full_precision()
    if weight_quant:
        rn.quantized_weights()
    if act_quant:
        rn.quantized_acts()


def validate(rn, val_loader, epoch, logfile=None):
    N = correct = 0
    rn = rn.eval()
    with torch.no_grad():
        for idx, (x, y) in enumerate(val_loader):
            x, y = x.cuda(), y.data.cpu().numpy()

            yhat = torch.argmax(rn(x), dim=1).data.cpu().numpy()

            correct += (yhat == y).sum()
            N += yhat.shape[0]

            print(
                "{:>5} / {}; {:.2f}%".format(
                    idx + 1, len(val_loader), 100 * (correct / N)
                ),
                end="\r",
                flush=True,
            )

    print_and_log(
        "Epoch {} Top-1-accuracy: {:.2f} %".format(epoch, 100 * (correct / N)),
        logfile=logfile,
    )

    return 100 * (correct / N)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def get_on_channels(quantizer):
    q = quantizer.quantizer
    on_channels = prune_probs = None
    if q.include_pruning:
        if isinstance(q, PACTQuantizer):
            on_channels = q.prune_mask.squeeze().data.cpu().numpy()
            prune_probs = 1 - np.mean(on_channels)
        else:
            gamma_2 = q.gamma_2.data.cpu().numpy()

            if q.gating_method == "l0":
                offset = math.log(-q.hc_gamma / q.hc_zeta) * q.hc_beta
                prune_probs = sigmoid(offset - gamma_2)
                on_channels = prune_probs < q.hc_thres
            else:
                keep_prob = sigmoid(gamma_2)
                on_channels = keep_prob > 0.5
                prune_probs = 1 - keep_prob

    return on_channels, prune_probs


def get_keep_ratio_and_prune_prob(quantizer, name, model, rn):
    keep_ratio = 1.0
    avg_prune_prob = 0.0
    on_channels, prune_probs = get_on_channels(quantizer)

    if on_channels is not None:
        keep_ratio, avg_prune_prob = on_channels.mean(), prune_probs.mean()

    if None not in [name, model, rn]:
        prev_layers = get_prev_layers(model, rn)  # returns None for non VGG networks
        if prev_layers is not None:
            # Adjust keep ratio for input channel pruning.
            prev_layer_list = prev_layers[name]
            on_channels_union = 0.0
            for prev_layer_q in prev_layer_list:
                on_channels, _ = get_on_channels(prev_layer_q)
                on_channels = on_channels if on_channels is not None else 1.0
                on_channels_union = 1 - (1 - on_channels_union) * (1 - on_channels)

            if not np.isscalar(on_channels_union):
                input_channels_p = on_channels_union.mean()
            else:
                # This is only a scalar if there were no gammas for the previous layer
                input_channels_p = 1.0

            keep_ratio *= input_channels_p

    return keep_ratio, avg_prune_prob


def pretty_print_quantization(
    relevant_quantizers, method, logfile=None, prune_only=False
):
    apply_sigmoid = method == "bayesian_bits"
    n_bits_txt = ""
    max_len = max([len(nm) for nm, _ in relevant_quantizers])

    template = (
        "| {{:<{ml}}} | {{:<7}} || {{:>{ns}}} | {{:>{ns}}} || "
        "{{:>{ns}.4f}} | {{:>{ns}.4f}} | {{:>{ns}.4f}} "
        "| {{:>{ns}.4f}} |"
    )
    template += "| {{x_min:>8.4f}} | {{x_max:>8.4f}} |\n"
    template = template.format(ml=max_len, ns=6 + 2 * int(not apply_sigmoid), lpi=8)

    dummy_zeros = [0] * 4 * (1 + (method == "bayesian_bits"))
    dummy_line = template.format("a", 8, "", "", *dummy_zeros, x_min=0.0, x_max=0.0)
    hline = "|" + "-" * (len(dummy_line) - 3) + "|"

    for name, quantizer in relevant_quantizers:
        bw = get_bw(quantizer, prune_only)
        if not prune_only and method is not None:
            gams = [
                getattr(quantizer.quantizer, "gamma_{}".format(2 ** i)).item()
                for i in range(2, 6)
            ]
        else:
            gams = [0] * 4
        if apply_sigmoid:
            gams = [sigmoid(g) for g in gams]
        if quantizer.quantizer is not None:
            x_min, x_max = (
                quantizer.quantizer.x_min.item(),
                quantizer.quantizer.x_max.item(),
            )
        else:
            x_min = x_max = float("nan")

        keep_ratio_str = on_prob_str = ""
        if quantizer.quantizer is not None and quantizer.quantizer.include_pruning:
            keep_ratio, avg_on_prob = get_keep_ratio_and_prune_prob(
                quantizer, None, None, None
            )
            keep_ratio_str = "{:.1f}".format(keep_ratio * 100)
            on_prob_str = "{:.4f}".format(avg_on_prob)

        n_bits_txt += template.format(
            name,
            int(np.log2(bw)) * "*",
            on_prob_str,
            keep_ratio_str,
            *gams,
            x_min=x_min,
            x_max=x_max
        )

    print_and_log(hline.replace("|", "-"), logfile=logfile)

    # Make header: | Quantizer | ln(B) || g2 | ... | g32 | x_min | x_max |
    # make title elements for header, put in list:
    hs = ["g"]
    header = ["Quantizer", "log2(B)", "P(off)", "% On"] + [
        h + str(2 ** i) for h in hs for i in range(2, 6)
    ]

    # format header with title elements:
    print_and_log(
        template.replace(".4f", "")
        .replace(".2f", "")
        .format(*header, x_min="x_min", x_max="x_max"),
        end="",
        logfile=logfile,
    )
    print_and_log(hline, logfile=logfile)

    # Add the rest of the text
    print_and_log(n_bits_txt, end="", logfile=logfile)
    print_and_log(hline.replace("|", "-"), logfile=logfile)
    print_and_log(logfile=logfile)


def assign_macs(per_layer_macs, quantizer_groups, model, prune_only=False):
    found_input = False
    max_macs = max(per_layer_macs.values())

    for qg, quantizers in quantizer_groups:
        act_bw, weights = 0, []
        found_act = False
        if len(qg) == 1:
            assert not found_input
            assert not found_act
            found_act = found_input = True

        act_quantizer = None
        for name, quantizer in zip(qg, quantizers):
            if "activation" in name:
                assert not found_act
                found_act = True
                act_quantizer = quantizer

            elif "weight" in name:
                weights.append((name, quantizer))
            else:
                raise ValueError("Unknown quantizer:", name)

        assert weights
        # assert len(weights) == 1
        for weight_name, quantizer in weights:
            macs = per_layer_macs[fix_name(weight_name, model=model)]

            if act_quantizer is not None and not prune_only:
                if act_quantizer.quantizer.mac_count is not None:
                    act_quantizer.quantizer.mac_count += macs
                else:
                    act_quantizer.quantizer.mac_count = macs
                    act_quantizer.quantizer.max_macs = max_macs

            assert quantizer.quantizer.mac_count is None
            quantizer.quantizer.mac_count = macs
            quantizer.quantizer.max_macs = max_macs


def print_bitops(
    per_layer_macs,
    quantizer_groups,
    model,
    rn,
    logfile=None,
    no_print=False,
    prune_only=False,
):

    # We assume input has 8bit color values per channel (even if it's FP32 encoded)
    total_bitops = 0
    n_act, n_weight = ([0] * 5), ([0] * 5)

    baseline = sum([v * (32 ** 2) for n, v in per_layer_macs.items()])
    found_input = False

    for qg, quantizers in quantizer_groups:
        act_bw, weights = 0, []
        found_act = False
        if len(qg) == 1:
            assert not found_input
            assert not found_act
            found_act = found_input = True
            act_name, act_bw = "input", 8

        for name, quantizer in zip(qg, quantizers):
            if "activation" in name:
                assert not found_act
                found_act = True
                act_bw = get_bw(quantizer, prune_only)

            elif "weight" in name:
                weight_bw = get_bw(quantizer, prune_only)
                prune_ratio, _ = get_keep_ratio_and_prune_prob(
                    quantizer, name, model, rn
                )
                weights.append((name, weight_bw, prune_ratio))

            else:
                raise ValueError("Unknown quantizer:", name)

        act_idx = int(np.log2(act_bw) - 1)
        n_act[act_idx] += 1
        assert weights

        for weight_name, weight_bw, prune_ratio in weights:
            total_bitops += (
                act_bw
                * weight_bw
                * prune_ratio
                * per_layer_macs[fix_name(weight_name, model=model)]
            )
            weight_idx = int(np.log2(weight_bw) - 1)
            weight_idx = max(0, min(4, weight_idx))  # bw = 2 ** (weight_idx + 1)
            n_weight[weight_idx] += 1

    args = []
    for i in range(len(n_weight)):
        args.append(n_weight[i])
        args.append(n_act[i])
    args += [total_bitops / baseline * 100]
    nbits_summary_str = (
        "bits: weights:   activations:\n"
        "  2:  {:>5}      {:>5}\n"
        "  4:  {:>5}      {:>5}\n"
        "  8:  {:>5}      {:>5}\n"
        " 16:  {:>5}      {:>5}\n"
        " 32:  {:>5}      {:>5}\n"
        "relative bops: {:.2f}%".format(*args)
    )

    if not no_print:
        print_and_log(nbits_summary_str, logfile=logfile)
        print_and_log(
            "GBOPs: {:<6.2}, GBOPs baseline: {:<6.2}".format(
                total_bitops * 10 ** (-9), baseline * 10 ** (-9)
            ),
            logfile=logfile,
        )
    return total_bitops / baseline * 100, nbits_summary_str


def get_exp_name(
    method,
    n_img,
    smallval,
    gating_lambda,
    gating_method,
    learning_rate,
    gamma_8_init,
    gamma_16_init,
    weight_quant,
    act_quant,
    quant_tricks,
    bias_corr,
    learned_scale,
):
    name = (
        "{method}-{n_img}img-sv={smallval}-lmb={gating_lambda:.4f}-{gating_method}-lr={lr:.4f}-"
        "g8={gamma_8_init:.4f}-g16={gamma_16_init:.4f}-wq={weight_quant}-aq={act_quant}-"
        "qt={quant_tricks}-bc={bias_corr}-learned_scale={ls}".format(
            method=method,
            n_img=n_img,
            smallval=int(smallval),
            gating_lambda=gating_lambda,
            gating_method=gating_method,
            lr=learning_rate,
            gamma_8_init=gamma_8_init,
            gamma_16_init=gamma_16_init,
            weight_quant=int(weight_quant),
            act_quant=int(act_quant),
            quant_tricks=quant_tricks,
            bias_corr=int(bias_corr),
            ls=0 if learned_scale is None else learned_scale[0],
        )
    )

    return name


def make_lr_scheduler(learning_rate, epochs):
    def lr_schedule(epoch):
        # linearly decay to zero
        ep_sched = int((2.0 / 3.0) * epochs)
        if epoch > ep_sched:
            norm = float(ep_sched - epochs)
            lr = (learning_rate / norm) * (epoch - epochs)
            return lr / learning_rate
        else:
            return 1.0

    return lr_schedule


def get_dataloaders_and_model(
    experiment,
    data_dir,
    model,
    batch_size,
    qparams,
    n_img,
    smallval,
    n_workers,
    pretrained=False,
    fold_bn=True,
    cuda=True,
):
    error_msg_model = "{} not available for {}".format(model, experiment)
    host = socket.gethostname()

    if experiment == "imagenet":
        assert model in ["resnet18", "mobilenetv2"], error_msg_model
        if n_img is not None:
            imagenet_base = os.path.join(data_dir, "imagenet_subsampled")
            imagenet_base += "_smallval/" if smallval else "/"
            imagenet_base += "sub_{}_images".format(n_img)
        else:
            assert not smallval
            # Modify this to point to imagenet
            imagenet_base = os.path.join(data_dir, "imagenet_raw")

        dataloaders = ImageNetDataLoaders(imagenet_base, 224, batch_size, n_workers)
        if model == "mobilenetv2":
            # Adapt this to your path
            model_path = "/local/mnt/workspace/quantization/mobilenet_v2_tony.pth.tar"
            rn = (
                mobilenetv2_quantized(
                    pretrained=pretrained, model_path=model_path, qparams=qparams
                )
                .cuda()
                .train()
            )

            if qparams.get("include_pruning", False):
                for name, module in rn.named_modules():
                    # Don't prune output channels of dw separable convs:
                    if (
                        isinstance(module, torch.nn.Conv2d)
                        and "features.1" not in name
                        and module.groups > 1
                    ):
                        module.include_pruning = False

        else:
            rn = (
                resnet18_quantized(pretrained=pretrained, qparams=qparams)
                .cuda()
                .train()
            )

    elif experiment == "cifar10":
        assert model in ["vggquantized", "resnet56_quantized"], error_msg_model
        dataloaders = Cifar10DataLoader(data_dir, batch_size)
        if model == "vggquantized":
            rn = VGGQuantized(**qparams)
        else:
            rn = resnet56_quantized(qparams)
        if cuda:
            rn = rn.cuda()

    elif experiment == "mnist":
        assert model in ["lenet5quantized"], error_msg_model
        dataloaders = MnistDataLoader(data_dir, batch_size)
        rn = LeNet5Quantized(**qparams)
        if cuda:
            rn = rn.cuda()
    else:
        raise ValueError("Unknown experiment:", experiment)

    rn.train()

    return dataloaders, rn


def get_relevant_quantizers_from_groups(quantizer_groups):
    result = []
    for qg, quantizers in quantizer_groups:
        result += list(zip(qg, quantizers))
    return result


def save_checkpoint(model, optimizers, epoch, logdir, logfile):
    if logdir is None:
        return

    log_fn = os.path.join(logdir, "epoch_{}.pth".format(epoch))
    chk_data = dict(
        model_state=model.state_dict(), param_optim=optimizers[0].state_dict()
    )
    if len(optimizers) >= 2:
        chk_data.update(quant_params_optim=optimizers[1].state_dict())
    if len(optimizers) >= 3:
        chk_data.update(scale_params_optim=optimizers[2].state_dict())
    print_and_log("Saving checkpoint to", log_fn, logfile=logfile)
    torch.save(chk_data, log_fn)


def set_quantizer_setting(m, b):
    assert b in {2, 4, 8, 16}
    for i in [4, 8, 16, 32]:
        g = getattr(m, "gamma_{}".format(i))
        if i > b:
            g.data.fill_(-6.0)
        else:
            g.data.fill_(6.0)
        print("{}: {}".format(i, float(g.data)), "\t", end="")
    print()


def run_naive_25_baseline(
    dataloaders, model, rn, bws, logfile, per_layer_macs, quantizer_groups
):
    # init everything to INT16
    gmp_quantizers = []
    rn.freeze_batch_norm()
    for n, m in rn.named_modules():
        if isinstance(m, BayesianBitsQuantizer):
            set_quantizer_setting(m, 16)
            gmp_quantizers.append((n, m))

    bws = [int(i) for i in bws.strip().split(",")]
    assert all([i in [4, 8] for i in bws])
    assert len(set(bws)) == len(bws)
    assert len(bws) > 0

    template = "{{:<{}}} {{:<3}}".format(max([len(n) for n, m in gmp_quantizers]))
    results = []
    print_and_log("START SENSITIVITY TESTING", logfile=logfile)
    for n, m in gmp_quantizers:
        for b in bws:
            print_and_log(template.format(n, b), end="", flush=True, logfile=logfile)
            set_quantizer_setting(m, b)
            top1_reduction = -validate(rn, dataloaders.train_loader, 0.0, None)
            print_and_log(top1_reduction, logfile=logfile)
            results.append((n, m, b, top1_reduction))
        set_quantizer_setting(m, 16)

    print_and_log("START PARETO GENERATION", logfile=logfile)
    results = sorted(results, key=lambda x: x[3])
    for n, m, b, r in results:
        bitops, _ = print_bitops(
            per_layer_macs, quantizer_groups, model, rn, no_print=True
        )
        set_quantizer_setting(m, b)
        mixedprec_top1 = validate(rn, dataloaders.val_loader, 0.0, None)
        print_and_log(template.format(n, b), bitops, mixedprec_top1, logfile=logfile)


def get_optimizers(
    experiment,
    model,
    pretrained,
    epochs,
    learning_rate,
    learning_rate_q,
    learning_rate_s,
    optimizer,
    optimizer_q,
    optimizer_s,
    rn,
    all_quantized,
    lr_schedule,
    learn="wgs",
):
    schedulers = None

    assert (optimizer_s is None) == (learning_rate_s is None)
    assert (optimizer_q is None) == (learning_rate_q is None)
    assert (optimizer_s is None) or (optimizer_q is not None)

    parameters, scale_parameters, quant_parameters, bias_parameters = [], [], [], []
    for n, p in rn.named_parameters():
        if any([s in n for s in ["x_min", "x_max", "s_2"]]):
            scale_parameters.append(p)
        elif "gamma_" in n:
            quant_parameters.append(p)
        else:
            parameters.append(p)
        if any([n.endswith(s) for s in ["bias", "beta"]]):
            bias_parameters.append(p)

    print(len(scale_parameters))
    print(len(quant_parameters))
    print(len(parameters))

    if not all_quantized:
        learning_rate_q = optimizer_q = learning_rate_s = optimizer_s = None
        quant_parameters = scale_parameters = []

    assert len(set(learn)) == len(learn), "Duplicate arguments in " + learn
    assert not ("w" in learn) or not ("b" in learn), "w and b are mutually exclusive"

    all_parameters, lr_list, optim_list, extra_args_list = [], [], [], []

    print("Learning this: {}".format(learn))
    if "w" in learn or "b" in learn:
        lr_list.append(learning_rate)
        optim_list.append(optimizer)
        extra_args_list.append({"weight_decay": 0.0})
        if "w" in learn:
            all_parameters.append(parameters)
        else:  # mutually exclusive
            all_parameters.append(bias_parameters)

    if "g" in learn:
        all_parameters.append(quant_parameters)
        lr_list.append(learning_rate_q)
        optim_list.append(optimizer_q)
        extra_args_list.append({"weight_decay": 0.0})

    if "s" in learn:
        all_parameters.append(scale_parameters)
        lr_list.append(learning_rate_s)
        optim_list.append(optimizer_s)
        extra_args_list.append({"weight_decay": 0.0})

    assert lr_list[0] is not None
    print(optim_list)
    print(lr_list)

    while len(lr_list) > 1:
        if lr_list[-1] is None:
            lr_list.pop(-1)
            optim_list.pop(-1)
            extra_args_list.pop(-1)
            all_parameters[-2] += all_parameters[-1]
            all_parameters.pop(-1)
        else:
            break

    assert (
        len(lr_list) == len(optim_list) == len(extra_args_list) == len(all_parameters)
    )

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

    if experiment != "imagenet":
        schedulers = [
            torch.optim.lr_scheduler.LambdaLR(
                optimizers[0], lr_lambda=make_lr_scheduler(learning_rate, epochs)
            )
        ]
    elif "w" in learn or "b" in learn:
        if lr_schedule.startswith("multistep"):
            epochs = [int(s) for s in lr_schedule.split(":")[1:]]
            if not pretrained:
                schedulers = [
                    torch.optim.lr_scheduler.MultiStepLR(opt, epochs)
                    for opt in optimizers
                ]
            else:
                schedulers = [
                    torch.optim.lr_scheduler.MultiStepLR(opt, epochs)
                    for opt in optimizers
                ]
        elif lr_schedule.startswith("cosine"):
            eta_min = float(lr_schedule.split(":")[1])
            schedulers = [
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizers[0], epochs, eta_min=eta_min
                )
            ] + [
                torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
                for opt in optimizers[1:]
            ]

    return optimizers, schedulers


def apply_cle(rn, model):
    rn.apply(replace_relu6_quantized)
    if model == "mobilenetv2":
        equalize_residual_block(rn.features[0], *rn.features[1].conv, min_s=0)
        for i in range(2, 18):
            equalize_residual_block(*rn.features[i].conv, min_s=0)
    elif model == "resnet18":

        def _equalize_block(block):
            if isinstance(block, QuantizedBlock):
                equalize_sequence(block.features, method="max", runs=1)

        rn.apply(_equalize_block)
    elif model == "mobilenetv1":
        for block in rn.model[1:14]:
            equalize_pair(*block)
    else:
        raise ValueError("Equalization (step 3) not support for model {}".format(rn))
    # sqnr_step3, total_sqnr_step3 = get_sqnr_model(model)


class TrainingDoneException(Exception):
    pass


@click.command()
@click.option(
    "--experiment", type=click.Choice(["cifar10", "imagenet", "mnist"]), required=True
)
@click.option("--data-dir", type=click.Path(), required=True)
@click.option(
    "--model",
    type=click.Choice(
        [
            "vggquantized",
            "resnet18",
            "resnet56_quantized",
            "lenet5quantized",
            "mobilenetv2",
        ]
    ),
    required=True,
)
@click.option("--method", type=click.Choice(["bayesian_bits"]), default="bayesian_bits")
@click.option("--batch-size", type=int, default=64, required=False)
@click.option("--n-img", type=int, default=None, required=False)
@click.option("--smallval", is_flag=True, default=False)
@click.option("--gating-lambda", type=float, default=0.0)
@click.option("--learning-rate", type=float, default=1e-2)
@click.option("--learning-rate-q", type=float, default=None)
@click.option("--learning-rate-s", type=float, default=None)
@click.option(
    "--optimizer", type=click.Choice(["SGD", "Adam"]), default="SGD", required=False
)
@click.option(
    "--optimizer-q", type=click.Choice(["SGD", "Adam"]), default=None, required=False
)
@click.option(
    "--optimizer-s", type=click.Choice(["SGD", "Adam"]), default=None, required=False
)
@click.option("--gating-method", type=click.Choice(["l0", "fixed"]), default="l0")
@click.option("--gamma-4-init", type=float, default=6.0)
@click.option("--gamma-8-init", type=float, default=6.0)
@click.option("--gamma-16-init", type=float, default=6.0)
@click.option("--gamma-32-init", type=float, default=6.0)
@click.option("--weight-quant/--no-weight-quant", is_flag=True, default=True)
@click.option("--act-quant/--no-act-quant", is_flag=True, default=True)
@click.option("--quant-tricks", type=str, default="")
@click.option("--bias-corr", is_flag=True, default=False)
@click.option(
    "--learned-scale",
    type=click.Choice(["range", "scale"]),
    default=None,
    required=False,
)
@click.option("--epochs", type=int, required=False, default=25)
@click.option("--eval-every", type=int, required=False, default=5)
@click.option("--logdir", type=click.Path(exists=True), required=False, default=None)
@click.option("--make-subdir", is_flag=True, default=False)
@click.option("--debug", is_flag=True, default=False)
@click.option("--clip-input", is_flag=True, default=False)
@click.option("--train_all", is_flag=True, default=False)
@click.option("--checkpointing", is_flag=True, default=False)
@click.option("--pretrained", is_flag=True, default=False)
@click.option("--n-workers", type=int, default=64)
@click.option("--eval-only", is_flag=True, default=False)
@click.option("--include-pruning", is_flag=True, default=False)
@click.option("--prune-only", is_flag=True, default=False)
@click.option("--reg-type", type=click.Choice(["const", "bop"]), default="const")
@click.option(
    "--learn",
    type=str,
    default="wgs",
    help="Parameters to learn. Default: wgs. w=weights, g=gates, s=scales, b=biases.\n"
    "w and b are mutually exclusive.",
)
@click.option("--save-checkpoints", is_flag=True, default=False)
@click.option("--weight-decay", type=float, default=0.0)
@click.option("--naive-25-baseline", type=str, default=None)
@click.option("--with-cle", is_flag=True, default=False)
@click.option("--lr-schedule", type=str, default="MultiStepLR:10:20:40")
@click.option("--static-fold-bn", is_flag=True, default=False)
@click.option("--fixed-gate-finetune-epochs", type=int, default=0)
@click.option("--fixed-8bit", is_flag=True, default=False)
@click.option("--fixed-48bit", is_flag=True, default=False)
@click.option("--log-every", type=int, default=None)
def experiment(
    experiment,
    data_dir,
    model,
    method,
    n_img,
    smallval,
    gating_lambda,
    learning_rate,
    learning_rate_q,
    learning_rate_s,
    optimizer,
    optimizer_q,
    optimizer_s,
    gating_method,
    gamma_4_init,
    gamma_8_init,
    gamma_16_init,
    gamma_32_init,
    weight_quant,
    act_quant,
    quant_tricks,
    bias_corr,
    learned_scale,
    epochs,
    eval_every,
    logdir,
    make_subdir,
    debug,
    clip_input,
    batch_size,
    train_all,
    checkpointing,
    pretrained,
    n_workers,
    eval_only,
    include_pruning,
    prune_only,
    reg_type,
    weight_decay,
    learn,
    save_checkpoints,
    naive_25_baseline,
    with_cle,
    lr_schedule,
    static_fold_bn,
    fixed_gate_finetune_epochs,
    fixed_8bit,
    fixed_48bit,
    log_every,
):

    assert not (fixed_8bit and fixed_48bit), "Can't have it both ways"

    all_quantized = weight_quant and act_quant
    stopafter = None
    if eval_only:
        epochs = 0
        logdir = None

    if debug:
        logdir = None
        stopafter = 10

    if logdir is not None and make_subdir:
        timestamp = str(int(time()))
        logdir = os.path.join(logdir, timestamp)
        os.mkdir(logdir)

    logfile = os.path.join(logdir, "log") if logdir is not None else None
    thisexpfile = (logfile + "-thisexp") if logfile is not None else None
    if thisexpfile is not None:
        with open(thisexpfile, "w") as f:
            print("experiment", experiment, file=f)
            print("model", model, file=f)
            print("method", method, file=f)
            print("batch_size", batch_size, file=f)
            print("gating_method", gating_method, file=f)
            print("gating_lambda", gating_lambda, file=f)
            print("optimizer", optimizer, file=f)
            print("optimizer_q", optimizer_q, file=f)
            print("optimizer_s", optimizer_s, file=f)
            print("learning_rate", learning_rate, file=f)
            print("learning_rate_q", learning_rate_q, file=f)
            print("learning_rate_s", learning_rate_s, file=f)
            print("gamma_4_init", gamma_4_init, file=f)
            print("gamma_8_init", gamma_8_init, file=f)
            print("gamma_16_init", gamma_16_init, file=f)
            print("gamma_32_init", gamma_32_init, file=f)
            print("pretrained", pretrained, file=f)
            print("include_pruning", include_pruning, file=f)
            print("prune_only", prune_only, file=f)
            print("reg_type", reg_type, file=f)
            print("n_img", n_img, file=f)
            print("learn", learn, file=f)
            print("weight_decay", weight_decay, file=f)
            print("clip_input", clip_input, file=f)
            print("naive_25_baseline", naive_25_baseline, file=f)
            print("static_fold_bn", static_fold_bn, file=f)
            print("fixed_8bit", fixed_8bit, file=f)
            print("fixed_48bit", fixed_48bit, file=f)
            print("fixed_gate_finetune_epochs", fixed_gate_finetune_epochs, file=f)

    t0 = time()
    nbits = 8 * (1 + int("bayesian_bits" in method))
    qparams = {
        "method": method,
        "n_bits": nbits,
        "n_bits_act": nbits,
        "per_channel_weights": False,
        "percentile": False,
        "gating_method": gating_method,
        "gamma_4_init": gamma_4_init,
        "gamma_8_init": gamma_8_init,
        "gamma_16_init": gamma_16_init,
        "gamma_32_init": gamma_32_init,
        "learned_scale": learned_scale,
        "clip_input": learned_scale,
        "checkpointing": checkpointing,
        "include_pruning": include_pruning,
        "prune_only": prune_only,
        "reg_type": reg_type,
        "fixed_8bit": fixed_8bit,
        "fixed_48bit": fixed_48bit,
    }

    dataloaders, rn = get_dataloaders_and_model(
        experiment,
        data_dir,
        model,
        batch_size,
        qparams,
        n_img,
        smallval,
        n_workers,
        pretrained=pretrained,
    )

    if with_cle and model == "mobilenetv2":
        apply_cle(rn, model)

    act_quant = act_quant and (fixed_8bit or fixed_48bit or not prune_only)

    set_rn_q(rn, weight_quant, act_quant)
    rn.set_full_precision_logits()

    print(weight_quant, act_quant)

    with torch.no_grad():
        print("pushing data through")
        for x, y in iter(dataloaders.train_loader):
            x = x.cuda()
            rn(x[0:2])
            break
        print("finished")
        rn.train()

    if static_fold_bn:
        rn.freeze_batch_norm()

    per_layer_macs, quantizer_groups = get_groups_and_macs(rn, experiment, model)
    relevant_quantizers = get_relevant_quantizers_from_groups(quantizer_groups)

    if naive_25_baseline is not None:
        assert gating_method == "fixed"
        return run_naive_25_baseline(
            dataloaders,
            model,
            rn,
            naive_25_baseline,
            logfile,
            per_layer_macs,
            quantizer_groups,
        )

    print("Logging to:", logfile)
    if reg_type == "bop":
        assign_macs(
            per_layer_macs, quantizer_groups, prune_only=prune_only, model=model
        )
    print_and_log(rn, logfile=logfile)

    if all_quantized:
        pretty_print_quantization(
            relevant_quantizers, method, logfile=logfile, prune_only=prune_only
        )
        print_bitops(
            per_layer_macs,
            quantizer_groups,
            model=model,
            rn=rn,
            logfile=logfile,
            prune_only=prune_only,
        )

    optimizers, schedulers = get_optimizers(
        experiment,
        model,
        pretrained,
        epochs,
        learning_rate,
        learning_rate_q,
        learning_rate_s,
        optimizer,
        optimizer_q,
        optimizer_s,
        rn,
        all_quantized,
        learn=learn,
        lr_schedule=lr_schedule,
    )

    print_and_log(optimizers, logfile=logfile)
    print_and_log(schedulers, logfile=logfile)

    rn = torch.nn.DataParallel(rn)
    epoch = 0
    try:
        for epoch in range(1, epochs + 1):
            train_epoch(
                rn,
                dataloaders.train_loader,
                gating_lambda,
                optimizers,
                epoch,
                epochs,
                logfile=logfile,
                stopafter=stopafter,
                weight_decay=weight_decay,
                log_every=log_every,
            )

            if epoch % eval_every == 0:
                # if all_quantized:
                pretty_print_quantization(
                    relevant_quantizers, method, logfile=logfile, prune_only=prune_only
                )
                print_bitops(
                    per_layer_macs,
                    quantizer_groups,
                    model=model,
                    rn=rn,
                    logfile=logfile,
                    prune_only=prune_only,
                )

                validate(rn, dataloaders.val_loader, epoch=epoch, logfile=logfile)
                print_and_log(logfile=logfile)

            if schedulers is not None:
                for scheduler in schedulers:
                    scheduler.step()

            if save_checkpoints:
                save_checkpoint(rn, optimizers, epoch, logdir, logfile)

        if fixed_gate_finetune_epochs == 0:
            raise TrainingDoneException

        if not save_checkpoints:
            save_checkpoint(rn, optimizers, "FINAL_PRE_FT", logdir, logfile)

        print_and_log()
        print_and_log(
            "-------------------------------------------------------------------------"
        )
        print_and_log("  Fixed gate finetuning epochs")
        print_and_log(
            "-------------------------------------------------------------------------"
        )
        print_and_log()

        assert "w" in learn and "g" in learn and "s" in learn
        del optimizers

        rn = rn.module

        replace_bb_quantizers(rn)
        static_fold_bn_convs(rn)
        rn = rn.cuda()

        set_rn_q(rn, weight_quant, act_quant)
        rn.set_full_precision_logits()

        rn = torch.nn.DataParallel(rn)

        validate(rn, dataloaders.val_loader, epoch=epoch, logfile=logfile)
        optimizers = get_finetune_optimizers(experiment, 1e-4, 1e-4, "SGD", "Adam", rn)
        finetune_schedulers = get_finetune_schedulers(
            optimizers, "cosine:0", fixed_gate_finetune_epochs, dataloaders.train_loader
        )

        for epoch in range(1, fixed_gate_finetune_epochs + 1):
            train_finetune_epoch(
                rn,
                dataloaders.train_loader,
                optimizers,
                epoch,
                fixed_gate_finetune_epochs,
                logfile=logfile,
                schedulers=finetune_schedulers,
            )
            validate(rn, dataloaders.val_loader, epoch=epoch, logfile=logfile)
            print_and_log(logfile=logfile)

            if save_checkpoints:
                save_checkpoint(rn, optimizers, "FT-{}".format(epoch), logdir, logfile)

    except KeyboardInterrupt:
        print_and_log(" CTRL + C Pressed. Stopping", logfile=logfile)
        print_and_log(
            ">>>> Experiment terminated during epoch", epoch, logfile=thisexpfile
        )
        valtop1 = validate(rn, dataloaders.val_loader, epoch=epochs, logfile=logfile)
        print_and_log("final_valtop1", valtop1, logfile=thisexpfile)
    except TrainingDoneException:
        pass
    except Exception as e:
        print_and_log(" Exception:", e, logfile=logfile)
        print_and_log(traceback.format_exc(), logfile=logfile)
        print_and_log(
            ">>>> Experiment terminated during epoch",
            epoch,
            "with exception:",
            e,
            logfile=thisexpfile,
        )
        valtop1 = validate(rn, dataloaders.val_loader, epoch=epochs, logfile=logfile)
        print_and_log("final_valtop1", valtop1, logfile=thisexpfile)

    if eval_only:
        valtop1 = validate(rn, dataloaders.val_loader, epoch=epochs, logfile=logfile)
        print_and_log("final_valtop1", valtop1, logfile=thisexpfile)

    if not eval_only and not save_checkpoints:
        save_checkpoint(rn, optimizers, "FINAL", logdir, logfile)

    done_str = (
        "================================================================================\n"
        " DONE with the following experiment:\n"
        "   experiment:             " + str(experiment) + "\n"
        "   model:                  " + str(model) + "\n"
        "   method:                 " + str(method) + "\n"
        "   batch size:             " + str(batch_size) + "\n"
        "   gating method:          " + str(gating_method) + "\n"
        "   gating lambda:          " + str(gating_lambda) + "\n"
        "   learning rate:          " + str(learning_rate) + "\n"
        "   learning rate q:        " + str(learning_rate_q) + "\n"
        "   learning rate s:        " + str(learning_rate_s) + "\n"
        "   optimizer:              " + str(optimizer) + "\n"
        "   optimizer_q:            " + str(optimizer_q) + "\n"
        "   optimizer_s:            " + str(optimizer_s) + "\n"
        "   include_pruning:        " + str(include_pruning) + "\n"
        "   gamma_4_init:           " + str(gamma_4_init) + "\n"
        "   gamma_8_init:           " + str(gamma_8_init) + "\n"
        "   gamma_16_init:          " + str(gamma_16_init) + "\n"
        "   gamma_32_init:          " + str(gamma_32_init) + "\n"
        "   pretrained:             " + str(pretrained) + "\n"
        "   reg_type:               " + str(reg_type) + "\n"
        "   learn:                  " + str(learn) + "\n"
        "   weight_decay:           " + str(weight_decay) + "\n"
        "   static_fold_bn:         " + str(static_fold_bn) + "\n"
        "   prune_only:             " + str(prune_only) + "\n"
        "   fixed_8bit:             " + str(fixed_8bit) + "\n"
        "   fixed_48bit:            " + str(fixed_48bit) + "\n"
        "fixed_gate_finetune_epochs " + str(fixed_gate_finetune_epochs) + "\n"
        "Time taken: {}s".format(time() - t0) + "\n"
        "================================================================================"
    )

    print_and_log(done_str, logfile=logfile)
    print_and_log("total_time {:.4f}s".format(time() - t0), logfile=logfile)
    print_and_log("last_epoch", epoch, logfile=thisexpfile)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    experiment()
