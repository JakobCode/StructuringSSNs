# All scripts in the folder ./Data/LIDC/data are taken from the Apache-2.0 lisensed repository
# https://github.com/MiguelMonteiro/PHiSeg-code
#
# Publication:
#
# Baumgartner, Christian F., et al.
# "Phiseg: Capturing uncertainty in medical image segmentation."
# International Conference on Medical Image Computing and Computer-Assisted Intervention.
# Springer, Cham, 2019.
#
#
# Code Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import numpy as np
import torch
from importlib_metadata import requires
from sklearn.semi_supervised import LabelPropagation
from torch import device
from torch import nn as nn
from torch import sparse_coo_tensor


class StochasticSegmentationNetwork(nn.Module):
    def __init__(
            self,
            resolution_levels,
            n_classes,
            rank,
            img_size,
            n0,
            norm=nn.BatchNorm2d):
        super().__init__()

        self.channel_dim = 1
        self.rank = rank
        self.n_classes = n_classes
        self.image_shape = img_size
        # encoder
        self.enc_list = []
        # decoder
        self.dec_list = []
        # recomb
        self.recomb_list = []

        self.resolution_levels = resolution_levels
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        conv_unit = nn.Conv2d

        add_bias = False if norm == nn.BatchNorm2d else True

        num_channels = [n0, 2 * n0, 4 * n0, 6 * n0, 6 * n0, 6 * n0, 6 * n0]

        for ii in range(resolution_levels):

            self.enc_list.append([])

            if ii != 0:
                self.enc_list[ii].append(nn.AvgPool2d(kernel_size=2, stride=2))

            self.enc_list[ii].append(
                conv_unit(
                    in_channels=num_channels[ii - 1] if ii > 0 else img_size[-3],
                    out_channels=num_channels[ii],
                    kernel_size=3,
                    bias=add_bias,
                    padding="same",
                )
            )
            if norm is not None:
                self.enc_list[ii].append(norm(num_channels[ii]))
            self.enc_list[ii].append(nn.ReLU())

            self.enc_list[ii].append(
                conv_unit(
                    in_channels=num_channels[ii],
                    out_channels=num_channels[ii],
                    kernel_size=3,
                    bias=add_bias,
                    padding="same",
                )
            )
            if norm is not None:
                self.enc_list[ii].append(norm(num_channels[ii]))
            self.enc_list[ii].append(nn.ReLU())

            self.enc_list[ii].append(
                conv_unit(
                    in_channels=num_channels[ii],
                    out_channels=num_channels[ii],
                    kernel_size=3,
                    bias=add_bias,
                    padding="same",
                )
            )
            if norm is not None:
                self.enc_list[ii].append(norm(num_channels[ii]))
            self.enc_list[ii].append(nn.ReLU())

        self.recomb_list.append(
            conv_unit(
                in_channels=num_channels[1],
                out_channels=num_channels[1],
                kernel_size=(1, 1),
                bias=add_bias,
            )
        )
        if norm is not None:
            self.recomb_list.append(norm(num_channels[1]))
        self.recomb_list.append(nn.ReLU())

        self.recomb_list.append(
            conv_unit(
                in_channels=num_channels[1],
                out_channels=num_channels[1],
                kernel_size=(1, 1),
                bias=add_bias,
                padding="same",
            )
        )
        if norm is not None:
            self.recomb_list.append(norm(num_channels[1]))
        self.recomb_list.append(nn.ReLU())

        self.recomb_list.append(
            conv_unit(
                in_channels=num_channels[1],
                out_channels=num_channels[1],
                kernel_size=(1, 1),
                bias=add_bias,
                padding="same",
            )
        )
        if norm is not None:
            self.recomb_list.append(norm(num_channels[1]))
        self.recomb_list.append(nn.ReLU())

        for jj in range(resolution_levels - 1):

            ii = resolution_levels - jj - 1  # used to index the encoder again

            self.dec_list.append([])

            p = min(resolution_levels - 2, ii)

            self.dec_list[jj].append(
                conv_unit(
                    in_channels=num_channels[ii - 1]
                    + num_channels[min(ii + 1, resolution_levels - 1)],
                    out_channels=num_channels[ii],
                    kernel_size=3,
                    bias=add_bias,
                    padding="same",
                )
            )
            if norm is not None:
                self.dec_list[jj].append(norm(num_channels[ii]))
            self.dec_list[jj].append(nn.ReLU())

            self.dec_list[jj].append(
                conv_unit(
                    in_channels=num_channels[ii],
                    out_channels=num_channels[ii],
                    kernel_size=3,
                    bias=add_bias,
                    padding="same",
                )
            )
            if norm is not None:
                self.dec_list[jj].append(norm(num_channels[ii]))
            self.dec_list[jj].append(nn.ReLU())

            self.dec_list[jj].append(
                conv_unit(
                    in_channels=num_channels[ii],
                    out_channels=num_channels[ii],
                    kernel_size=3,
                    bias=add_bias,
                    padding="same",
                )
            )
            if norm is not None:
                self.dec_list[jj].append(norm(num_channels[ii]))
            self.dec_list[jj].append(nn.ReLU())

        self.enc_list = nn.ModuleList(
            [nn.Sequential(*self.enc_list[i]) for i in range(len(self.enc_list))]
        )
        self.dec_list = nn.ModuleList(
            [nn.Sequential(*self.dec_list[i]) for i in range(len(self.dec_list))]
        )
        self.recomb_list = nn.Sequential(*self.recomb_list)

        self.mean = nn.Conv2d(
            in_channels=num_channels[1],
            out_channels=n_classes,
            kernel_size=(
                1,
                1))
        self.log_cov_diag = nn.Conv2d(
            in_channels=num_channels[1],
            out_channels=n_classes,
            kernel_size=(
                1,
                1))
        self.cov_factor = nn.Conv2d(
            in_channels=num_channels[1],
            out_channels=n_classes * rank,
            kernel_size=(1, 1),
        )

        self.last_mean = None
        self.last_cov_factor = None
        self.last_cov_diag = None
        # self.last_log_cov_diag = None

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.dist = torch.distributions.Normal(
            torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
        )

    def crop_and_concat(self, inputs, axis=-1):
        """
        Layer for cropping and stacking feature maps of different size along a different axis.
        Currently, the first feature map in the inputs list defines the output size.
        The feature maps can have different numbers of channels.
        :param inputs: A list of input tensors of the same dimensionality but can have different sizes
        :param axis: Axis along which to concatentate the inputs
        :return: The concatentated feature map tensor
        """

        output_size = inputs[0].shape[1:]
        # output_size = tf.shape(inputs[0])[1:]
        concat_inputs = [inputs[0]]

        for ii in range(1, len(inputs)):

            larger_size = inputs[ii].shape[1:]
            # larger_size = tf.shape(inputs[ii])

            # Don't subtract over batch_size because it may be None
            start_crop = np.subtract(larger_size, output_size) // 2

            if len(output_size) == 4:  # nets3D images
                cropped_tensor = inputs[ii][
                    :,
                    :,
                    start_crop[1]: (output_size[1] + start_crop[1]),
                    start_crop[2]: (output_size[2] + start_crop[2]),
                    start_crop[3]: (output_size[3] + start_crop[3]),
                ]
            elif len(output_size) == 3:  # nets2D images
                # cropped_tensor = torch.slice(inputs[ii],
                #                          (0, start_crop[0], start_crop[1], 0),
                #                          (-1, output_size[0], output_size[1], -1))
                cropped_tensor = inputs[ii][
                    :,
                    :,
                    start_crop[1]: (output_size[1] + start_crop[1]),
                    start_crop[2]: (output_size[2] + start_crop[2]),
                ]
            else:
                raise ValueError(
                    "Unexpected number of dimensions on tensor: %d" %
                    len(output_size))

            concat_inputs.append(cropped_tensor)

        return torch.cat(concat_inputs, dim=axis)

    def forward(self, x):

        encoded = []
        for i in range(self.resolution_levels):
            if i == 0:
                encoded.append(self.enc_list[i](x))
            else:
                encoded.append(self.enc_list[i](encoded[-1]))

        decoded = encoded[-1]
        for j in range(self.resolution_levels - 1):
            i = self.resolution_levels - j - 1

            decoded = self.upsample(decoded)

            decoded = self.crop_and_concat(
                [decoded, encoded[i - 1]], axis=self.channel_dim
            )
            decoded = self.dec_list[j](decoded)

        recomb = self.recomb_list(decoded)

        eps = 1e-5
        flat_size = self.image_shape[-1] * \
            self.image_shape[-2] * self.n_classes

        mean = self.mean(recomb)
        log_diag = self.log_cov_diag(recomb)

        cov_factor = self.cov_factor(recomb)

        # 2 x 128 x 128 --> 128 *128 * 2
        mean = torch.reshape(torch.permute(
            mean, [0, 2, 3, 1]), [-1, flat_size, 1])

        # 2 x 128 x 128
        log_diag = torch.reshape(torch.permute(
            log_diag, [0, 2, 3, 1]), [-1, flat_size])

        # 2*rank x 128 x 128
        cov_factor = torch.reshape(
            torch.permute(cov_factor, [0, 2, 3, 1]), [-1, flat_size, self.rank]
        )

        cov_diag = log_diag.exp() + eps

        if self.training:
            num_samples = 25
            z = self.dist.rsample(
                (num_samples, cov_factor.shape[0], self.rank, 1))
            e = self.dist.rsample(
                (num_samples, cov_factor.shape[0], cov_diag.shape[-1], 1)
            )
        else:
            num_samples = 100
            z = self.dist.rsample(
                (num_samples, cov_factor.shape[0], self.rank, 1))
            e = self.dist.rsample(
                (num_samples, cov_factor.shape[0], cov_diag.shape[-1], 1)
            )

        samples = (
            mean.unsqueeze(0)
            + torch.matmul(cov_factor.unsqueeze(0), z)
            + e * cov_diag.unsqueeze(0).unsqueeze(-1)
        )

        self.last_mean = mean.detach().cpu().numpy()
        self.last_cov_factor = cov_factor.detach().cpu().numpy()
        self.last_cov_diag = cov_diag.detach().cpu().numpy()

        if self.training:
            return samples.reshape(num_samples, -1, 128, 128, 2).permute(
                [1, 0, 4, 2, 3]
            )
        else:
            return samples.reshape(num_samples, -1, 128, 128, 2).permute(
                [1, 0, 4, 2, 3]
            )
