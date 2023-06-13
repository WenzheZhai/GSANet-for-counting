import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import math


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.out_layer = nn.Sequential(
            nn.Conv2d(512, 218, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(218, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.dct = DCTBlock()

        self.spacial_branch_1 = nn.Sequential(conv_block(512, 4))
        self.spacial_branch_2 = nn.Sequential(conv_block(1024, 8))

        # self.backend = nn.Sequential(
        #     nn.Conv2d(512, 256, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, 1),
        #     nn.ReLU(inplace=True),
        # )

        # 权重初始化
        self._init_params()

        self.inplanes = 64  # 第一层卷积输出的通道数

        # self.pertubration = None
        # self.pertubration = DistributionUncertainty
        # self.uncertainty = 0.5

        res50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        self.frontend = nn.Sequential(res50.conv1, res50.bn1, res50.relu, res50.maxpool)

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.pertubration0 = (
        #     self.pertubration(dim=64, p=self.uncertainty)
        #     if self.pertubration
        #     else nn.Identity()
        # )
        # self.pertubration1 = (
        #     self.pertubration(dim=64, p=self.uncertainty)
        #     if self.pertubration
        #     else nn.Identity()
        # )
        # self.pertubration2 = (
        #     self.pertubration(dim=64, p=self.uncertainty)
        #     if self.pertubration
        #     else nn.Identity()
        # )
        # self.pertubration3 = (
        #     self.pertubration(dim=128, p=self.uncertainty)
        #     if self.pertubration
        #     else nn.Identity()
        # )
        # self.pertubration4 = (
        #     self.pertubration(dim=256, p=self.uncertainty)
        #     if self.pertubration
        #     else nn.Identity()
        # )
        # self.pertubration5 = (
        #     self.pertubration(dim=512, p=self.uncertainty)
        #     if self.pertubration
        #     else nn.Identity()
        # )

        # self.layer1 = res50.layer1
        # self.layer2 = res50.layer2
        # self.layer3 = res50.layer3

        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer1.load_state_dict(res50.layer1.state_dict())

        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer2.load_state_dict(res50.layer2.state_dict())

        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer3.load_state_dict(res50.layer3.state_dict())

        # self.layer3.load_state_dict(res50.layer3.state_dict())

        # self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        # stride=2 feature map size reduce by half
        # the size of feature map reduce by half. the number of channel down to quarter.
        # mutiply by 4

    def forward(self, x):
        x = self.frontend(x)  # 64 1/2
        # x = self.conv1(x)
        # # x = self.pertubration0(x)

        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # # x = self.pertubration1(x)

        x = self.layer1(x)  # 256 1/4
        # x = self.pertubration2(x)

        mid_x = self.layer2(x)  # 512 1/8
        # x = self.pertubration3(x)

        spatial_x1 = self.spacial_branch_1(mid_x)  # 128 1/8

        x = self.layer3(mid_x)  # 1024 1/16
        # x = self.pertubration4(x)

        spatial_x2 = self.spacial_branch_2(x)  # 128 1/16
        spatial_x2 = F.interpolate(spatial_x2, scale_factor=2)  # 128 1/8

        dct_x = self.dct(x)  # 1024 1/16
        dct_x = F.interpolate(dct_x, scale_factor=2)  # 256 1/8

        x = torch.cat((spatial_x1, spatial_x2, dct_x), 1)  # 512 1/8
        w = nn.Softmax(dim=1)(x)
        mid_x = mid_x * w + mid_x  # 512 1/8

        x = self.out_layer(mid_x)  # 1 1/8
        x = F.interpolate(x, scale_factor=8)  # 1 1
        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        """生成多个连续的残差块

        Args:
            block (object): 残差块类型 BasicBlock/Bottleneck
            planes (int): 残差块第一个卷积层的卷积核的个数
            blocks (int): 残差块个数 3/4/6/3
            stride (int, optional): 残差块中卷积步长. Defaults to 1.

        Returns:
            object: 连续残差块
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # 类属性
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, dim, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0
        self.dim = dim

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(
            x.shape[0], x.shape[1], 1, 1
        )
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(
            x.shape[0], x.shape[1], 1, 1
        )

        return x


class DCT3D(nn.Module):
    def __init__(self):
        super(DCT3D, self).__init__()

    def forward(self, input):
        B, C, M, N = input.shape

        A_col = torch.cos(
            torch.mm(
                torch.arange(N).float().unsqueeze(0).t(),
                (torch.arange(N).float() + 0.5).unsqueeze(0),
            )
            * math.pi
            / N
        ) * torch.sqrt(torch.tensor(2.0) / N)

        A_col[0, :] = A_col[0, :] / torch.sqrt(torch.tensor(2.0))

        A_col = A_col.type(torch.cuda.FloatTensor)

        A_row = torch.cos(
            torch.mm(
                torch.arange(M).float().unsqueeze(0).t(),
                (torch.arange(M).float() + 0.5).unsqueeze(0),
            )
            * math.pi
            / M
        ) * torch.sqrt(torch.tensor(2.0) / M)

        A_row[0, :] = A_row[0, :] / torch.sqrt(torch.tensor(2.0))
        A_row = A_row.type(torch.cuda.FloatTensor)
        A_cnl = torch.cos(
            torch.mm(
                torch.arange(C).float().unsqueeze(0).t(),
                (torch.arange(C).float() + 0.5).unsqueeze(0),
            )
            * math.pi
            / C
        ) * torch.sqrt(torch.tensor(2.0) / C)
        A_cnl[0, :] = A_cnl[0, :] / torch.sqrt(torch.tensor(2.0))
        A_cnl = A_cnl.type(torch.cuda.FloatTensor)

        col = input.view(B * C * M, N).transpose(0, 1)
        D_col = A_col.mm(col)

        row = D_col.view(N, B * C, M).transpose(0, 2).contiguous().view(M, B * C * N)
        D_row = A_row.mm(row)

        cnl = D_row.view(M, B, C, N).transpose(0, 2).contiguous().view(C, B * M * N)
        D_cnl = A_cnl.mm(cnl)

        size = D_cnl.shape[0] * D_cnl.shape[1]
        x = D_cnl.view(size)

        freqs = [
            B * C * M * N / 4,
            B * C * M * N / 16,
        ]

        for i in freqs:
            # mark = abs(x).topk(i)[0][-1]
            mark = abs(x).topk(int(i))[0][-1]
            mask = abs(x).ge(mark)
            temp = mask.type(torch.cuda.FloatTensor) * x
            D_cnl = temp.view(D_cnl.shape)

            ID_cnl = A_cnl.t().mm(D_cnl)  # ID_cnl == cnl
            ID_cnl_ = (
                ID_cnl.view(C, B, M, N).transpose(0, 2).contiguous().view(M, B * C * N)
            )  # temp == D_row
            ID_row = A_row.t().mm(ID_cnl_)  # ID_row == row
            ID_row_ = (
                ID_row.view(M, B * C, N).transpose(0, 2).contiguous().view(N, B * C * M)
            )
            ID_col = A_col.t().mm(ID_row_)  # ID_col == col
            ID_col_ = ID_col.transpose(0, 1).view(B, C, M, N)
            if i == freqs[0]:
                output = ID_col_.unsqueeze(0)

            else:
                output = torch.cat((output, ID_col_.unsqueeze(0)), 0)

        return output


class DCTBlock(nn.Module):
    def __init__(self):
        super(DCTBlock, self).__init__()

        self.dct = DCT3D()

        self.branch_1 = nn.Sequential(conv_block(1024, 8))
        self.branch_2 = nn.Sequential(conv_block(1024, 8))

        # self.weights = nn.Softmax(dim=1)

    def forward(self, x):
        d = self.dct(x)

        d_1 = self.branch_1(d[0])
        d_2 = self.branch_2(d[1])

        d = torch.cat((d_1, d_2), 1)
        # w = self.weights(d)
        # x = x * w + x
        return d


def conv_block(in_channels=512, branches=2):
    out_channels = int(in_channels / branches)
    # the number of channel must be int
    # int / int = float

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
        ),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(inplace=True),
    )
