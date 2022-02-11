# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
import torch

from anonymized_compression_package.quantization.straight_through import (
    QuantizationHijacker,
)
from anonymized_compression_package.quantization.quantized_folded_bn import (
    BatchNormQScheme,
)
from anonymized_compression_package.utils import to_numpy


def replace_relu6_quantized(layer):
    if isinstance(layer, QuantizationHijacker) and isinstance(
        layer.activation_function, torch.nn.ReLU6
    ):
        layer.activation_function = torch.nn.ReLU()


def equalize_residual_block(prev_layer, sep_layer, next_layer, method="max", min_s=0):
    if method == "range":
        # Equalize layers based on range (first for sep in the middle)!
        range_sep = get_weight_range(sep_layer, axis=(1, 2, 3))
        range_prev = get_weight_range(prev_layer, axis=(1, 2, 3))
        range_next = get_weight_range(next_layer, axis=(0, 2, 3))

        scale_factor = torch.pow(range_sep * range_prev * range_next, 1.0 / 3)
        scale_prev = range_prev / scale_factor
        scale_next = scale_factor / range_next
    elif method == "max":
        # Equalize layers based on max
        range_sep = get_weight_max(sep_layer, axis=(1, 2, 3))
        range_prev = get_weight_max(prev_layer, axis=(1, 2, 3))
        range_next = get_weight_max(next_layer, axis=(0, 2, 3))

        scale_factor = torch.pow(range_sep * range_prev * range_next, 1.0 / 3)
        scale_prev = range_prev / scale_factor
        scale_next = scale_factor / range_next

        if min_s:
            below_min = scale_prev < min_s
            # Set it to min
            scale_prev[below_min] = min_s

            # Rescale rest again
            scale_rest = torch.pow(range_sep * min_s * range_next, 1.0 / 2)
            scale_next[below_min] = (scale_rest / range_next)[below_min]

            print(
                "CLE[max]: corrected {}/{} scalings to min_s ({})".format(
                    below_min.sum(), len(scale_prev), min_s
                )
            )

            below_min_next = scale_next < min_s
            if below_min_next.sum():
                print("WARNING in next some are below min_s")
                scale_next[below_min_next] = min_s

    elif method == "sep":
        # Get maximum precision in the separable layer
        scale_factor = torch.pow(get_weight_max(sep_layer, axis=(1, 2, 3)), 0.5)

        scale_prev = 1.0 / scale_factor
        scale_next = scale_factor
    elif method == "sep_max":
        # Get maximum precision in the separable layer, equal others based on max
        range_sep = get_weight_max(sep_layer, axis=(1, 2, 3))
        range_prev = get_weight_max(prev_layer, axis=(1, 2, 3))
        range_next = get_weight_max(next_layer, axis=(0, 2, 3))

        scale_factor = torch.pow(range_sep * range_prev * range_next, 1.0 / 2)
        scale_prev = range_prev / scale_factor
        scale_next = scale_factor / range_next
    else:
        raise ValueError("Unknown equalize method {}".format(method))

    # Scale next layer
    zero_channels = remove_zero_channels_separable(prev_layer, sep_layer, next_layer)
    scale_next[zero_channels] = 1.0
    scale_next_layer(sep_layer, next_layer, scale_next)

    # Scale previous layer
    scale_prev[zero_channels] = 1.0
    scale_next_layer(prev_layer, sep_layer, scale_prev)


def equalize_pair(layer1, layer2, method="max"):
    if method == "range":
        range_l1 = get_weight_range(layer1, axis=(1, 2, 3))
        range_l2 = get_weight_range(layer2, axis=(0, 2, 3))

        scale_factor = range_l1 / torch.pow(range_l1 * range_l2, 1.0 / 2)
    elif method == "max":
        max_l1 = get_weight_max(layer1, axis=(1, 2, 3))
        max_l2 = get_weight_max(layer2, axis=(0, 2, 3))

        scale_factor = max_l1 / torch.pow(max_l1 * max_l2, 1.0 / 2)
    else:
        raise ValueError("Unknown equalize method {}".format(method))

    zero_channels = remove_zero_channels_pair(layer1, layer2)
    scale_factor[zero_channels] = 1.0

    scale_next_layer(layer1, layer2, scale_factor)


def equalize_sequence(sequence, method="max", runs=5):
    if len(sequence) == 2:
        return equalize_pair(*sequence, method=method)

    for _ in range(runs):
        for i in range(0, len(sequence) - 1):
            equalize_pair(*sequence[i : i + 2], method="max")


def get_weight_range(layer, axis):
    weight_folded = np.array(layer.get_weight_bias()[0].data)
    return torch.tensor(weight_folded.max(axis=axis) - weight_folded.min(axis=axis)).to(
        device=layer.weight.device
    )


def get_weight_max(layer, axis):
    weight_folded = to_numpy(layer.get_weight_bias()[0].data)
    return torch.tensor(np.abs(weight_folded).max(axis=axis)).to(
        device=layer.weight.device
    )


def scale_next_layer(layer, next_layer, scale_factor):
    layer.gamma.data /= scale_factor
    layer.beta.data /= scale_factor
    if next_layer.weight.shape[1] == scale_factor.shape[0]:
        # Normal conv
        next_layer.weight.data *= scale_factor[None, :, None, None]
    elif next_layer.weight.shape[1] == 1:
        # Separable conv
        next_layer.weight.data *= scale_factor[:, None, None, None]
    else:
        raise ValueError("Dimensions are not matching.")

    # Clear cached parameters
    layer.cached_params = None
    next_layer.cached_params = None


def remove_zero_channels_separable(prev_layer, sep_layer, next_layer):
    zero_channels = remove_zero_channels_pair(sep_layer, next_layer)

    # Sep layer output channels are shared with prev layer as well, so we can remove them
    prev_layer.gamma.data[zero_channels] = 0.0
    prev_layer.beta.data[zero_channels] = 0.0

    prev_layer.cached_params = None
    return zero_channels


def remove_zero_channels_pair(layer1, layer2):
    zero_channels = layer1.gamma.abs() < 1.0e-20

    # Calculate bias to absorb from sep layer
    absorb_bias = torch.zeros_like(layer1.beta)
    absorb_bias[zero_channels] = layer1.beta[zero_channels]
    absorb_bias[absorb_bias < 0] = 0
    weight_matrix = layer2.weight.sum(3).sum(2)
    bias_correction = weight_matrix.mv(absorb_bias)

    # Remove all weights and absorb bias
    layer1.beta.data[zero_channels] = 0.0
    layer1.gamma.data[zero_channels] = 0.0
    layer2.running_mean.data -= bias_correction
    layer2.weight.data[:, zero_channels] = 0.0

    layer1.cached_params = None
    layer2.cached_params = None
    return zero_channels
