"""
AlexNet on CIFAR-100.

Seems like there are several different networks (A-E). Smallest one (A) has 11
different layers.

Their input size was 224x224. Ours is 32x32.
Spatial resolution is preserved in conv layers, ie output is also 224x224.
Conv2D - 3x3 - 64 channels
maxpool
Conv2D - 3x3 - 128 channels
maxpool
Conv2D - 3x3 - 256
Conv2D - 3x3 - 256
maxpool
Conv2D - 3x3 - 512
Conv2D - 512
maxpool
Conv2D - 512
Conv2D - 512
maxpool
FC - out features: 4096
ReLU
FC - out features: 4096
ReLU
FC - out features: 1000 (one for each class, so we need 100)
softmax

Actually wrong, there are some ReLUs after conv layres, and there are some dropout,
and some local response normalization. But I'm not sure how to implement those.
"""
import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

cifar_path = (pathlib.Path(__file__) / "../../dataset").resolve()
dataset = torchvision.datasets.CIFAR100(
    cifar_path,
    # ToTensor scales [0,255] to [0.0, 1.0]
    transform=torchvision.transforms.ToTensor(),
)

dataloader = torch.utils.data.DataLoader(dataset)


class AlexNet(nn.Module):
    """
    AlexNet on CIFAR-100.

    Seems like there are several different networks (A-E). Smallest one (A) has 11
    different layers.

    Their input size was 224x224. Ours is 32x32.
    Spatial resolution is preserved in conv layers, ie output is also 224x224.
    Conv2D - 3x3 - 64 channels
    relu
    maxpool
    Conv2D - 3x3 - 128 channels
    relu
    maxpool
    Conv2D - 3x3 - 256
    relu
    Conv2D - 3x3 - 256
    relu
    maxpool
    Conv2D - 3x3 - 512
    relu
    Conv2D - 512
    relu
    maxpool
    Conv2D - 512
    relu
    Conv2D - 512
    relu
    maxpool
    FC - out features: 4096
    ReLU
    FC - out features: 4096
    ReLU
    FC - out features: 1000 (one for each class, so we need 100)
    softmax
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # (b, 3, 32, 32) -> (b, 64, 32, 32
            nn.Conv2d(3, 64, (3, 3), stride=1, padding="same"),
            nn.ReLU(),
            # (b, 64, 32, 32) -> (b, 64, 16, 16)
            nn.MaxPool2d((2, 2), stride=2),
            # (b, 64, 16, 16) -> (b, 128, 16, 16)
            nn.Conv2d(64, 128, (3, 3), stride=1, padding="same"),
            nn.ReLU(),
            # (b, 128, 16, 16) -> (b, 128, 8, 8)
            nn.MaxPool2d((2, 2), stride=2),
            # (b, 128, 8, 8) -> (b, 256, 8, 8)
            nn.Conv2d(128, 256, (3, 3), stride=1, padding="same"),
            nn.ReLU(),
            # (b, 256, 8, 8) -> (b, 256, 8, 8)
            nn.Conv2d(256, 256, (3, 3), stride=1, padding="same"),
            nn.ReLU(),
            # (b, 256, 8, 8) -> (b, 256, 4, 4
            nn.MaxPool2d((2, 2), stride=2),
            # (b, 256, 4, 4) -> (b, 512, 4, 4)
            nn.Conv2d(256, 512, (3, 3), stride=1, padding="same"),
            nn.ReLU(),
            # (b, 512, 4, 4) -> (b, 512, 4, 4)
            nn.Conv2d(512, 512, (3, 3), stride=1, padding="same"),
            nn.ReLU(),
            # (b, 512, 4, 4) -> (b, 512, 2, 2)
            nn.MaxPool2d((2, 2), stride=2),
            # (b, 512, 2, 2) -> (b, 2048)
            nn.Flatten(),
            # (b, 2048) -> (b, 4096)
            nn.Linear(2048, 4096),
            nn.ReLU(),
            # (b, 4096) -> (b, 4096)
            nn.Linear(4096, 4096),
            nn.ReLU(),
            # (b, 4096) -> (b, 100)
            nn.Linear(4096, 100),
            # (b, 100) -> (b, 100)
            nn.Softmax(dim=1),
        )

        # Initialization procedure from 3.1: weights are sampled from N(0, 0.01)
        # and biases are 0
        for layer in self.layers:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.layers(x)


# TODO: Add optimization/training loop
# TODO: Train on GPU
# TODO: Save weights

# TODO: Training is quite complex, different LR, augmentations, ...

# Training loop
