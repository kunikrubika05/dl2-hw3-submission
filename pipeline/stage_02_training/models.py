"""VGG13 + общий декодер: конкатенация для U-Net, сложение для LinkNet."""

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


class VGG13Encoder(nn.Module):
    def __init__(
        self,
        num_blocks: int = 4,
        weights="DEFAULT",
    ) -> None:
        super().__init__()
        weights = getattr(weights, "name", weights)
        if weights not in (None, "DEFAULT", "IMAGENET1K_V1"):
            raise ValueError("weights must be None or VGG13 DEFAULT/IMAGENET1K_V1")
        self.num_blocks = num_blocks
        if isinstance(num_blocks, bool) or not isinstance(num_blocks, int) or not 1 <= num_blocks <= 4:
            raise ValueError("num_blocks must be an integer from 1 to 4")
        channels = (3, 64, 128, 256, 512)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels[i + 1], channels[i + 1], 3, padding=1),
                nn.ReLU(),
            )
            for i in range(num_blocks)
        ])
        if weights is not None:
            # Official torchvision VGG13 IMAGENET1K_V1; do not construct its unused classifier.
            weights = torch.hub.load_state_dict_from_url(
                "https://download.pytorch.org/models/vgg13-19584684.pth",
                map_location="cpu", progress=False, check_hash=True, weights_only=True,
            )
            for i, block in enumerate(self.blocks):
                block.load_state_dict({
                    f"{layer}.{name}": weights[f"features.{5 * i + layer}.{name}"]
                    for layer in (0, 2) for name in ("weight", "bias")
                })

    def forward(
        self,
        x: torch.Tensor,
    ) -> list[torch.Tensor]:
        activations = []
        for i, block in enumerate(self.blocks):
            if i:
                x = F.max_pool2d(x, 2)
            x = block(x)
            activations.append(x)
        return activations


class DecoderBlock(nn.Module):
    def __init__(
        self,
        out_channels: int,
        merge: Literal["concat", "add"] = "concat",
    ) -> None:
        super().__init__()
        if merge not in ("concat", "add"):
            raise ValueError("merge must be 'concat' or 'add'")
        self.merge = merge
        self.upconv = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        self.conv1 = nn.Conv2d(out_channels * (2 if merge == "concat" else 1), out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(
        self,
        down: torch.Tensor,
        left: torch.Tensor,
    ) -> torch.Tensor:
        x = self.upconv(F.interpolate(down, size=left.shape[-2:], mode="nearest"))
        x = torch.cat((x, left), dim=1) if self.merge == "concat" else x + left
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Decoder(nn.Module):
    def __init__(
        self,
        num_filters: int,
        num_blocks: int,
        merge: Literal["concat", "add"] = "concat",
    ) -> None:
        super().__init__()
        if merge not in ("concat", "add"):
            raise ValueError("merge must be 'concat' or 'add'")
        if isinstance(num_blocks, bool) or not isinstance(num_blocks, int) or num_blocks < 0:
            raise ValueError("num_blocks must be a nonnegative integer")
        self.blocks = nn.ModuleList([
            DecoderBlock(num_filters * 2 ** i, merge) for i in reversed(range(num_blocks))
        ])

    def forward(
        self,
        acts: list[torch.Tensor],
    ) -> torch.Tensor:
        x = acts[-1]
        for block, left in zip(self.blocks, acts[-2::-1]):
            x = block(x, left)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        num_blocks: int = 4,
        pretrained: bool = True,
        *,
        merge: Literal["concat", "add"] = "concat",
    ) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer")
        self.encoder = VGG13Encoder(num_blocks, weights="DEFAULT" if pretrained else None)
        self.decoder = Decoder(64, num_blocks - 1, merge)
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.final(self.decoder(self.encoder(x)))


class LinkNet(UNet):
    def __init__(
        self,
        num_classes: int = 1,
        num_blocks: int = 4,
        pretrained: bool = True,
    ) -> None:
        super().__init__(num_classes, num_blocks, pretrained, merge="add")
