# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.


import torch
from torch import nn

import numpy as np

from anonymized_compression_package.quantization.straight_through import (
    QuantizationHijacker,
)
import torch.nn.functional as F


class BatchNormQScheme(QuantizationHijacker):
    """ Implements the BatchNorm recommendations/best practices from Ragu's Google paper on
    Quantization.

    To avoid weight jitter as a result of batch normalization the paper proposes the following
    quantization method during training:

    General approach
    ----------------
    $ BN(\textbf{y}) = \gamma \left( \frac{\textbf{y} - \mu_B}{\sigma_B} \right) + \beta $

    $ BN_{inf}(\textbf{y}) = \gamma \left( \frac{\textbf{y} - \mu}{\sigma} \right) + \beta $

    where $\mu$ and $\sigma$ for inference are estimated from a running mean of
    $\mu_B$ and $\sigma_B$, and $\gamma$ and $\beta$ are learnable parameters.

    For a linear layer, where $\textbf{y} = \textbf{W}\textbf{x}$ the batchnorm parameters can be
    folded into the weight matrix for inference as follows:

    $ BN_{inf}(y) = \frac{\gamma\textbf{W}}{\sigma}\textbf{x} + \beta - \frac{\mu}{\sigma}$

    To avoid jitter due to quantization and a difference between the moving average and per-batch
    mean and variance the paper always quantizes the first term in the equation above, then corrects
    post-quantization with the batch mean and stddev during training.

    Training happens in two stages: before freeze, where normal train time batch norm is used, and
    after freeze, where batch norm is applied using the (frozen) running mean and variance. Weights
    are still updated after the running mean and variance mean

    Before running mean/variance freeze:
    $ BN(\textbf{y}) = Q\left( \frac{\gamma\textbf{W}}{\sigma} \right)\textbf{x}
    \frac{\sigma}{\sigma_B} + \beta - \frac{\gamma\mu_B}{\sigma_B} $

    After running mean/variance freeze:
    $ BN(\textbf{y}) = Q\left( \frac{\gamma\textbf{W}}{\sigma} \right)\textbf{x} +
    \beta - \frac{\gamma\mu}{\sigma} $

    NB: in this method, the running mean and variance seem to be not be quantized.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("bias", None)  # Bias will be learned by BN params
        super(BatchNormQScheme, self).__init__(*args, **kwargs, bias=False)

        bn_dim = self.get_bn_dim()
        self.register_buffer("running_mean", torch.zeros(bn_dim))
        self.register_buffer("running_var", torch.ones(bn_dim))

        self.gamma = nn.Parameter(torch.ones(bn_dim))
        self.beta = nn.Parameter(torch.zeros(bn_dim))
        self.epsilon = kwargs.get("eps", 1e-5)

        self._batch_norm_frozen = self._batch_norm_was_frozen = False
        self.momentum = kwargs.pop("momentum", 0.1)

        self.mean_corr = kwargs.get("mean_corr") or 0.0
        self.alpha = kwargs.get("alpha")

    @property
    def sigma(self):
        return torch.sqrt(self.running_var + self.epsilon)

    def get_bn_dim(self):
        if isinstance(self, nn.Linear):
            return self.out_features
        elif isinstance(self, (nn.Conv1d, nn.Conv2d)):
            return self.out_channels
        else:
            msg = "Unsupported type used. Must be instance of nn.Linear, nn.Conv1d or nn.Conv2d"
            raise NotImplementedError(msg)

    def freeze_batch_norm(self):
        self._batch_norm_frozen = True

    def unfreeze_batch_norm(self):
        self._batch_norm_frozen = False

    def forward(self, x):
        # General batch norm folding trick:
        # BN_inf(y) = gamma.(W.y - mu)/sigma + beta -->
        # W_bn  = gamma.W / sigma; bias_bn = gamma.mu/sigma;
        # BN_inf(y) = W_bn y + sigma
        # sigma and gamma are vectors whose values are broadcast over the rows of W
        # In forward the following steps are taken:
        # 1. Store original parameters
        # 2. Compute batch mean & variance with unquantized weights
        # 3. Fold EMA mean & variance into weight
        # 4. Quantize weights. This means quantization will happen with the weights
        #    similar/identical to those that are used during inference -> no jitter :D
        # 5. Run forward using the quantized weights.
        # 6. If BN not frozen: correct for batch mean & variance
        # 7. Quantize activations

        # ensure that superclass doesn't quantize activations since we need to use the unquantized
        # activations later on
        weight, bias = self.get_params()
        res = self.run_forward(x, weight, bias)

        # compute batch norm bias. Adjust for current batch mean/var if batch_norm not frozen
        res = self.adjust_output(x, res)

        # quantize activations if the caller so desires
        res = self.quantize_activations(res)

        return res

    def named_parameters(self, memo=None, prefix=""):
        r"""Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself

        Yields:
            (string, Parameter): Tuple containing the name and parameter

        Example::

            >>> for name, param in self.named_parameters():
            >>>    if name in ['bias']:
            >>>        print(param.size())

        """
        yield "{}weight".format(prefix + "." if prefix else ""), self.weight
        if not self._batch_norm_frozen:
            yield "{}gamma".format(prefix + "." if prefix else ""), self.gamma
            yield "{}beta".format(prefix + "." if prefix else ""), self.beta

    def get_weight_bias(self):
        # fold batch norm into weights. transpose to allow broadcasting
        # weight linear: Out_dim, In_dim
        # weight conv1d (2d): C_out, C_in, K(, K)
        # i.e. can use the same transpose + broadcasting logic for both
        weight_folded = (
            self.weight.transpose(0, -1) * self.gamma / self.sigma
        ).transpose(0, -1)

        if self.alpha is not None:
            wf = weight_folded
            if isinstance(self.alpha, nn.Parameter):
                weight_folded = 0.5 * (
                    (wf + self.alpha).abs() - (wf - self.alpha).abs()
                )
            else:
                weight_folded = torch.clamp(wf, -self.alpha, self.alpha)

        return weight_folded, None

    def adjust_output(self, x, res):
        shape = [res.shape[1]] + [1] * (len(res.shape) - 2)
        if not self.training or self._batch_norm_frozen:
            # not updating EMA mean & variance
            bias = (
                self.beta - self.gamma * self.running_mean / self.sigma + self.mean_corr
            )
            res += bias.view(*shape)

        else:
            # compute output using original (non-batch norm folded) weights, use these to compute
            # batch mean & variance, update EMA mean & variance. If no quantization is used this
            # corresponds to a slightly-more-expensive-to-compute regular Batch Norm
            bn_output = self.run_forward(x, self.weight, None)
            batch_mean, batch_var = self.get_batch_stats(bn_output)

            m = np.prod(res.shape) / shape[0]
            m = float((m - 1) / m)

            bv = m * batch_var + self.epsilon

            sigma_b = torch.sqrt(bv)
            # apply correction for current batch mean & var wrt folding of long term weights.
            scale = (self.sigma / sigma_b).view(*shape)
            bias = (self.beta - self.gamma * batch_mean / sigma_b).view(*shape)
            res = res * scale + bias

            self.update_running_avgs(batch_mean, batch_var)

        return res

    def get_batch_stats(self, x):
        if self._batch_norm_frozen:
            return None, None

        m = np.prod(x.shape) / x.shape[1]

        # this somewhat convoluted code removes the need for an expensive reshape operation.
        # complication arises from the fact that one can only take mean/var over 1 dim at a time
        axes = [0] + list(range(2, len(x.shape)))
        batch_mean = x
        for ax in reversed(axes):
            batch_mean = batch_mean.mean(ax)

        shape = [x.shape[1]] + [1] * (len(x.shape) - 2)
        batch_var = (x - batch_mean.view(shape)).pow(2)
        for ax in reversed(axes):
            batch_var = batch_var.mean(ax)

        batch_var *= m / (m - 1)

        return batch_mean, batch_var

    def update_running_avgs(self, batch_mean, batch_var):
        self.running_mean.copy_(
            self.momentum * batch_mean.detach()
            + (1 - self.momentum) * self.running_mean
        )
        self.running_var.copy_(
            self.momentum * batch_var.detach() + (1 - self.momentum) * self.running_var
        )
