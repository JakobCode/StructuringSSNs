# The network is implemented following this blog post: https://amaarora.github.io/2020/09/13/unet.html

import torch
import torch.nn as nn
import torchvision


class StochasticSegmentationNetwork(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        rank,
        enc_chs=(64, 128, 256, 512, 1024),
        dec_chs=(1024, 512, 256, 128, 64),
    ):
        super().__init__()

        enc_chs = (n_channels,) + enc_chs

        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)

        self.num_classes = n_classes
        self.rank = rank

        self.mean = nn.Conv2d(64, n_classes, 1)
        self.cov_fact = nn.Conv2d(64, n_classes * rank, 1)
        self.diag = nn.Conv2d(64, n_classes, 1)

        self.last_mean = None
        self.last_cov_diag = None
        self.last_cov_fac = None

    def forward(self, x, num_samples=20):
        [batch_size, _, w, h] = x.shape

        # x = self.input(x)

        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])

        # B, C, W, H --> B, W, H, C ( * R)
        mean = self.mean(out)
        mean = mean.permute((0, 2, 3, 1))

        cov_factor = self.cov_fact(out).permute((0, 2, 3, 1))
        log_diag = self.diag(out).permute((0, 2, 3, 1))
        diag = log_diag.exp()

        # B, W, H, C --> B, W * H * C
        mean = torch.flatten(mean, start_dim=1)
        diag = torch.flatten(diag, start_dim=1) + 1e-20
        cov_factor = torch.reshape(
            cov_factor, (batch_size, w * h * self.num_classes, self.rank)
        )

        self.last_mean = mean.detach().cpu().numpy()
        self.last_cov_diag = diag.detach().cpu().numpy()
        self.last_log_cov_diag = log_diag.detach().cpu().numpy()

        self.last_cov_fac = cov_factor.detach().cpu().numpy()

        dist = torch.distributions.LowRankMultivariateNormal(
            loc=mean, cov_factor=cov_factor, cov_diag=diag
        )

        samples = (
            dist.rsample((num_samples,))
            .reshape([num_samples, batch_size, w, h, self.num_classes])
            .permute((1, 0, 4, 2, 3))
        )

        return samples


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(
            chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs
