import torch
from torch import nn
import model.common as common
import torch.nn.functional as F
from moco.builder_DGSRN import MoCo
from model.CBAM import CBAMLayer


import torch
from torch import nn


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)



def make_model(args):
    return BlindSR(args)


def conv2d(in_ch, out_ch, kernel_size=3, padding=None, stride1=1, bias=True):
    if padding is None:
        padding = kernel_size // 2
    return nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride1, padding=padding, bias=bias)


class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in // reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        """
        att = self.conv_du(x[1][:, :, None, None])

        return x[0] * att


class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_se = nn.Sequential(
            nn.Conv2d(channels_in*2, channels_in*2 // reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in*2 // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.conv = common.default_conv(channels_in, channels_out, 1)
        self.con = conv2d(channels_in, channels_out, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        """
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        """
        b, c, h, w = x[0].size()

        # branch 1
        out = self.avg_pool(x[0]).squeeze(-1).squeeze(-1)
        out = torch.cat((out, x[1]), dim=1)
        out = self.conv_se(out[:, :, None, None])
        out = x[0]*out
        # branch 2
        out = x[0]+out
        return out


class DABlock(nn.Module):
    def __init__(self, conv=common.default_conv, n_feat=64, kernel_size=3, reduction=8):
        super(DABlock, self).__init__()

        self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        """
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        """

        out = self.relu(self.da_conv1(x))
        out = self.relu(self.conv1(out))
        out = self.relu(self.da_conv2([out, x[1]]))
        out = self.conv2(out) + x[0]

        return out


class DAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
        super(DAG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            DABlock(conv, n_feat, kernel_size, reduction) \
            for _ in range(n_blocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        """
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        """
        res = x[0]
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1]])
        res = self.body[-1](res)
        res = res + x[0]

        return res


class DGSRN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DGSRN, self).__init__()

        self.n_groups = 5
        n_blocks = 5
        n_feats = 64
        kernel_size = 3
        reduction = 8
        scale = int(args.scale[0])

        # RGB mean for DIV2K
        rgb_mean = (0.4690, 0.4490, 0.4036)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(255.0, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(255.0, rgb_mean, rgb_std, 1)

        # head module
        modules_head = [conv(3, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # compress
        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )

        # body
        modules_body = [
            DAG(common.default_conv, n_feats, kernel_size, reduction, n_blocks) \
            for _ in range(self.n_groups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail
        # modules_tail = [common.Upsampler(conv, scale, n_feats, act=False),
        #                 conv(n_feats, 3, kernel_size)]
        modules_tail = [conv(n_feats, 3*scale*scale, kernel_size), nn.PixelShuffle(scale),
                        conv(3, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, k_v):
        k_v = self.compress(k_v)

        # sub mean
        x = self.sub_mean(x)

        # head
        x = self.head(x)

        # body
        res = x
        for i in range(self.n_groups):
            res = self.body[i]([res, k_v])
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        # add mean
        x = self.add_mean(x)

        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            # 加入CBAMLayer模块
            CBAMLayer(64),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out


class BlindSR(nn.Module):
    def __init__(self, args):
        super(BlindSR, self).__init__()

        # Generator
        self.G = DGSRN(args)

        # Encoder
        self.E = MoCo(base_encoder=Encoder)

    def forward(self, x):
        if self.training:
            x_query, x_key, kernel = x

            # degradation-aware representation learning
            fea, logits, labels = self.E(x_query, x_key, kernel)

            # degradation-aware SR
            sr = self.G(x_query, fea)

            return sr, logits, labels
        else:
            # degradation-aware representation learning
            x, y, kernel = x
            fea = self.E(x, x, kernel)

            # degradation-aware SR
            sr = self.G(x, fea)

            return sr

