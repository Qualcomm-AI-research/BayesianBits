# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.


from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck, resnet18, resnet50

from anonymized_compression_package.quantization.autoquant_utils import (
    quantize_model,
    QuantLinear,
)
from anonymized_compression_package.quantization.base_classes import QuantizedModel
from anonymized_compression_package.quantization.straight_through import (
    QuantizationHijacker,
)


default_quant_params = {
    "method": "asymmetric",
    "n_bits": 8,
    "per_channel_weights": False,
}


class QuantizedBlock(QuantizationHijacker):
    def __init__(self, block, **quant_params):
        super(QuantizedBlock, self).__init__(**quant_params)

        if isinstance(block, Bottleneck):
            features = nn.Sequential(
                block.conv1,
                block.bn1,
                block.relu,
                block.conv2,
                block.bn2,
                block.relu,
                block.conv3,
                block.bn3,
            )
        elif isinstance(block, BasicBlock):
            features = nn.Sequential(
                block.conv1, block.bn1, block.relu, block.conv2, block.bn2
            )

        self.features = quantize_model(features, **quant_params)
        self.downsample = (
            quantize_model(block.downsample, **quant_params)
            if block.downsample
            else None
        )

        self.relu = block.relu

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.features(x)

        out += residual
        out = self.relu(out)

        return self.quantize_activations(out)


class QuantizedResNet(QuantizedModel):
    def __init__(self, resnet, **quant_params):
        super(QuantizedResNet, self).__init__()
        specials = {BasicBlock: QuantizedBlock, Bottleneck: QuantizedBlock}
        features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.features = quantize_model(features, specials=specials, **quant_params)

        self.avgpool = resnet.avgpool
        self.fc = quantize_model(resnet.fc, **quant_params)

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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_quantized(pretrained=False, qparams=None, **kwargs):
    """Constructs a quantized ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """
    model = resnet18(pretrained, **kwargs)

    quant_params = default_quant_params.copy()
    if qparams:
        quant_params.update(qparams)
        print("New quantization parameters:", quant_params)

    quant_model = QuantizedResNet(model, **quant_params)
    return quant_model
