# Parts of this script are taken from: https://github.com/bfortuner/pytorch_tiramisu.
#
# The source repository is under MIT License.
# Authors from original repository: Brendan Fortuner
#
# For more details and references checkout the repository and the readme of our repository.
#
# Author of this edited script: Anonymous

from turtle import forward

import torch
import torch.nn as nn

from .layers import *


def FCDenseNetSSN103(n_classes, rank=10):
    return FCDenseNetSSN(
        in_channels=3,
        down_blocks=(4, 5, 7, 10, 12),
        up_blocks=(12, 10, 7, 5, 4),
        bottleneck_layers=15,
        growth_rate=16,
        out_chans_first_conv=48,
        n_classes=n_classes,
        rank=rank,
    )


class sample_module(nn.Module):
    def __init__(self, in_channels, n_classes, rank, mean_weight) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.rank = rank

        self.mean = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.mean.weight = mean_weight

        self.log_diag = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.cov_fact = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_classes * rank,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.last_mean = None

    def forward(self, x):
        input_shape = x.shape

        mean = self.mean(x)
        cov_diag = self.log_diag(x).exp() + 10e-10
        cov_fac = self.cov_fact(x)

        mean = torch.permute(mean, dims=[0, 2, 3, 1])
        cov_fac = torch.permute(cov_fac, dims=[0, 2, 3, 1])
        cov_diag = torch.permute(cov_diag, dims=[0, 2, 3, 1])

        self.last_mean = mean.detach().cpu()
        self.last_cov_diag = cov_diag.detach().cpu()
        self.last_cov_fac = cov_fac.detach().cpu()

        mean = torch.reshape(mean, shape=[input_shape[0], -1])
        cov_fac = torch.reshape(cov_fac, shape=[input_shape[0], -1, self.rank])
        cov_diag = torch.reshape(cov_diag, shape=[input_shape[0], -1])

        if self.training:
            num_samples = 25
        else:
            num_samples = 25

        samples = torch.distributions.LowRankMultivariateNormal(
            loc=mean, cov_factor=cov_fac, cov_diag=cov_diag
        ).rsample((num_samples,))

        samples = torch.reshape(
            samples,
            [
                num_samples,
                input_shape[0],
                input_shape[2],
                input_shape[3],
                self.n_classes,
            ],
        )

        samples = torch.permute(samples, dims=[1, 0, 4, 2, 3])

        return samples


class FCDenseNetSSN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5),
        bottleneck_layers=5,
        growth_rate=16,
        out_chans_first_conv=48,
        n_classes=12,
        rank=10,
    ):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []
        self.n_classes = n_classes
        self.rank = rank
        ## First Convolution ##

        self.add_module(
            "firstconv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_chans_first_conv,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i])
            )
            cur_channels_count += growth_rate * down_blocks[i]
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module(
            "bottleneck",
            Bottleneck(
                cur_channels_count,
                growth_rate,
                bottleneck_layers))
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(
                TransitionUp(prev_block_channels, prev_block_channels)
            )
            cur_channels_count = prev_block_channels + \
                skip_connection_channel_counts[i]

            self.denseBlocksUp.append(
                DenseBlock(
                    cur_channels_count,
                    growth_rate,
                    up_blocks[i],
                    upsample=True))
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##

        self.transUpBlocks.append(
            TransitionUp(prev_block_channels, prev_block_channels)
        )
        cur_channels_count = prev_block_channels + \
            skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1], upsample=False))
        cur_channels_count += growth_rate * up_blocks[-1]

        ## Softmax ##

        self.mean = nn.Conv2d(
            in_channels=cur_channels_count,
            out_channels=n_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.log_diag = nn.Conv2d(
            in_channels=cur_channels_count,
            out_channels=n_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.cov_fact = nn.Conv2d(
            in_channels=cur_channels_count,
            out_channels=n_classes * self.rank,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, num_samples=None):

        input_shape = x.shape

        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        mean = self.mean(out)
        cov_diag = torch.clip(self.log_diag(out), -20, 20).exp() + 1e-5
        cov_fac = torch.clip(self.cov_fact(out), -20, 20)

        mean = torch.permute(mean, dims=[0, 2, 3, 1])
        cov_fac = torch.permute(cov_fac, dims=[0, 2, 3, 1])
        cov_diag = torch.permute(cov_diag, dims=[0, 2, 3, 1])

        self.last_mean = mean.detach().cpu()
        self.last_cov_diag = cov_diag.detach().cpu()
        self.last_cov_fac = cov_fac.detach().cpu()

        mean = torch.reshape(mean, shape=[input_shape[0], -1])
        cov_fac = torch.reshape(cov_fac, shape=[input_shape[0], -1, self.rank])
        cov_diag = torch.reshape(cov_diag, shape=[input_shape[0], -1])

        if num_samples is None:
            if self.training:
                num_samples = 20
            else:
                num_samples = 1

        samples = torch.distributions.LowRankMultivariateNormal(
            loc=mean, cov_factor=cov_fac, cov_diag=cov_diag
        ).rsample((num_samples,))

        samples = torch.reshape(
            samples,
            [
                num_samples,
                input_shape[0],
                input_shape[2],
                input_shape[3],
                self.n_classes,
            ],
        )

        samples = torch.permute(samples, dims=[1, 0, 4, 2, 3])

        return samples
