# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import click
import os
from time import time
import torch
import warnings
import traceback

from benchmarks.image_net.fixed_gate_finetune_utils import (
    replace_bb_quantizers,
    static_fold_bn_convs,
    train_finetune_epoch,
    get_finetune_schedulers,
)
from benchmarks.image_net.fixed_gate_finetune_utils import get_finetune_optimizers
from benchmarks.image_net.gated_mixed_precision import (
    set_rn_q,
    get_dataloaders_and_model,
    get_groups_and_macs,
    get_relevant_quantizers_from_groups,
    print_bitops,
    validate,
    save_checkpoint,
)
from benchmarks.image_net.mixed_precision_utils import print_and_log


@click.command()
@click.option("--data-dir", type=click.Path(), required=True)
@click.option(
    "--model",
    type=click.Choice(["vggquantized", "resnet18", "lenet5quantized", "mobilenetv2"]),
    required=True,
)
@click.option("--model-file", type=click.Path(exists=True), required=True)
@click.option("--batch-size", type=int, default=64, required=False)
@click.option("--learning-rate", type=float, default=1e-2)
@click.option("--learning-rate-s", type=float, default=None)
@click.option(
    "--optimizer", type=click.Choice(["SGD", "Adam"]), default="SGD", required=False
)
@click.option(
    "--optimizer-s", type=click.Choice(["SGD", "Adam"]), default=None, required=False
)
@click.option("--epochs", type=int, required=False, default=25)
@click.option("--logdir", type=click.Path(exists=True), required=False, default=None)
@click.option("--make-subdir", is_flag=True, default=False)
@click.option("--n-workers", type=int, default=64)
@click.option("--lr-schedule", type=str, default="MultiStepLR:10:20:40")
@click.option("--weight-decay", type=float, default=0.0)
@click.option("--n-feed", type=int, default=0)
@click.option("--fixed-8bit", is_flag=True, default=False)
@click.option("--fixed-48bit", is_flag=True, default=False)
@click.option("--prune-only", is_flag=True, default=False)
def experiment(
    model,
    model_file,
    data_dir,
    learning_rate,
    learning_rate_s,
    optimizer,
    optimizer_s,
    n_workers,
    epochs,
    logdir,
    make_subdir,
    fixed_8bit,
    fixed_48bit,
    prune_only,
    batch_size,
    lr_schedule,
    n_feed,
    weight_decay=0,
):

    if logdir is not None and make_subdir:
        timestamp = str(int(time()))
        logdir = os.path.join(logdir, timestamp)
        os.mkdir(logdir)

    learned_scale = "range"

    logfile = os.path.join(logdir, "log") if logdir is not None else None
    thisexpfile = (logfile + "-thisexp") if logfile is not None else None
    if thisexpfile is not None:
        with open(thisexpfile, "w") as f:
            print("experiment", "imagenet", file=f)
            print("model", model, file=f)
            print("method", "bb-qaft", file=f)
            print("batch_size", batch_size, file=f)
            print("gating_method", "N/A", file=f)
            print("gating_lambda", "N/A", file=f)
            print("optimizer", optimizer, file=f)
            print("optimizer_q", "N/A", file=f)
            print("optimizer_s", optimizer_s, file=f)
            print("learning_rate", learning_rate, file=f)
            print("learning_rate_q", "NaN", file=f)
            print("learning_rate_s", learning_rate_s, file=f)
            print("gamma_4_init", "NaN", file=f)
            print("gamma_8_init", "NaN", file=f)
            print("gamma_16_init", "NaN", file=f)
            print("gamma_32_init", "NaN", file=f)
            print("pretrained", "N/A", file=f)
            print("include_pruning", "N/A", file=f)
            print("reg_type", "N/A", file=f)
            print("n_img", "N/A", file=f)
            print("learn", "N/A", file=f)
            print("weight_decay", weight_decay, file=f)
            print("clip_input", "N/A", file=f)
            print("naive_25_baseline", False, file=f)
            print("fixed_8bit", fixed_8bit, file=f)
            print("fixed_48bit", fixed_48bit, file=f)
            print("prune_only", prune_only, file=f)

    d = {
        k.replace("module.", ""): v
        for k, v in torch.load(model_file)["model_state"].items()
    }
    include_pruning = any([".gamma_2" in k for k in d.keys()])

    t0 = time()
    nbits = 16
    qparams = {
        "method": "bayesian_bits",  # for model loading
        "n_bits": nbits,
        "n_bits_act": nbits,
        "per_channel_weights": False,
        "percentile": False,
        "gating_method": "l0",
        "gamma_4_init": 6,
        "gamma_8_init": 6,
        "gamma_16_init": 6,
        "gamma_32_init": 6,
        "learned_scale": learned_scale,
        "clip_input": learned_scale,
        "checkpointing": True,
        "include_pruning": include_pruning,
        "reg_type": "bop",
        "fixed_8bit": fixed_8bit,
        "fixed_48bit": fixed_48bit,
        "prune_only": prune_only,
    }

    dataloaders, rn = get_dataloaders_and_model(
        "imagenet",
        data_dir,
        model,
        batch_size,
        qparams,
        None,
        False,
        n_workers,
        pretrained=True,
    )

    act_quant = fixed_8bit or fixed_48bit or not prune_only
    set_rn_q(rn, True, act_quant)
    rn.set_full_precision_logits()

    with torch.no_grad():
        print("pushing data through")
        for x, y in iter(dataloaders.train_loader):
            rn(x.cuda()[:8])
            break
        print("finished")
        rn.train()

    rn.load_state_dict(d)
    rn.quantized()
    rn.set_full_precision_logits()

    # print('Pre-transformation validation')
    # validate(rn, dataloaders.val_loader, epoch=-1, logfile=logfile)

    replace_bb_quantizers(rn)

    if n_feed > 0:
        with torch.no_grad():
            print("Feeding data for BatchNorm")
            for idx, (x, _) in enumerate(dataloaders.train_loader):
                t = time()
                rn(x.cuda())
                print("{} / {}; {:.4f}".format(idx + 1, n_feed, time() - t), end="\r")
                if idx + 1 == n_feed:
                    break
            print("finished     ")

    rn.quantized()
    rn.set_full_precision_logits()

    # print('Pre-BN folding validation')
    # validate(rn, dataloaders.val_loader, epoch=-1, logfile=logfile)

    static_fold_bn_convs(rn)
    rn = rn.cuda()
    rn.set_full_precision_logits()

    per_layer_macs, quantizer_groups = get_groups_and_macs(rn, "imagenet", model)
    # relevant_quantizers = get_relevant_quantizers_from_groups(quantizer_groups)

    # rn = modify_and_load_model(rn, model_file)
    # print_and_log(rn, logfile=logfile)

    # pretty_print_quantization(relevant_quantizers, None, logfile=logfile)
    print_bitops(per_layer_macs, quantizer_groups, model=model, rn=rn, logfile=logfile)

    optimizers = get_finetune_optimizers(
        "imagenet", learning_rate, learning_rate_s, optimizer, optimizer_s, rn
    )
    schedulers = get_finetune_schedulers(
        optimizers, lr_schedule, epochs, dataloaders.train_loader
    )

    print_and_log(optimizers, logfile=logfile)
    print_and_log(schedulers, logfile=logfile)

    rn = torch.nn.DataParallel(rn)
    epoch = 0
    print("Post-transformation validation")
    validate(rn, dataloaders.val_loader, epoch=epoch, logfile=logfile)
    try:
        for epoch in range(1, epochs + 1):
            train_finetune_epoch(
                rn,
                dataloaders.train_loader,
                optimizers,
                epoch,
                epochs,
                logfile=logfile,
                weight_decay=weight_decay,
                schedulers=schedulers,
            )

            validate(rn, dataloaders.val_loader, epoch=epoch, logfile=logfile)
            print_and_log(logfile=logfile)

            # if schedulers is not None:
            #     for scheduler in schedulers:
            #         scheduler.step()

            # save_checkpoint(rn, optimizers, epoch, logdir, logfile)

    except KeyboardInterrupt:
        print_and_log(" CTRL + C Pressed. Stopping", logfile=logfile)
        print_and_log(
            ">>>> Experiment terminated during epoch", epoch, logfile=thisexpfile
        )
        valtop1 = validate(rn, dataloaders.val_loader, epoch=epochs, logfile=logfile)
        print_and_log("final_valtop1", valtop1, logfile=thisexpfile)
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

    save_checkpoint(rn, optimizers, "FINAL", logdir, logfile)

    done_str = (
        "================================================================================\n"
        " DONE with the following experiment:\n"
        "   experiment:      " + str("imagenet") + "\n"
        "   model:           " + str(model) + "\n"
        "   batch size:      " + str(batch_size) + "\n"
        "   learning rate:   " + str(learning_rate) + "\n"
        "   learning rate s: " + str(learning_rate_s) + "\n"
        "   optimizer:       " + str(optimizer) + "\n"
        "   optimizer_s:     " + str(optimizer_s) + "\n"
        "   weight_decay:    " + str(weight_decay) + "\n"
        "   fixed_8bit:      " + str(fixed_8bit) + "\n"
        "   fixed_48bit:     " + str(fixed_48bit) + "\n"
        "   prune_only:      " + str(prune_only) + "\n"
        "Time taken: {}s".format(time() - t0) + "\n"
        "================================================================================"
    )

    print_and_log(done_str, logfile=logfile)
    print_and_log("total_time {:.4f}s".format(time() - t0), logfile=logfile)
    print_and_log("last_epoch", epoch, logfile=thisexpfile)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    experiment()
