import torch.nn as nn
import torch
from torchvision import models
import math
import torch.nn.functional as F

import math
import torch
import torch.nn as nn
from torchvision.models import ResNet

# Res50

from misc.layer import Conv2d, FC

from misc.utils import *


class DCT3D(nn.Module):
    def __init__(self, freqs):
        super(DCT3D, self).__init__()
        self.freqs = freqs

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
            B * C * M * N / 64,
            B * C * M * N / 256,
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


def conv_block(inp, oup, dilation):
    return nn.Sequential(
        nn.Conv2d(512, 128, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=1),
        nn.ReLU(inplace=True),
    )


class DCTBlock(nn.Module):
    def __init__(self, inp, oup, freqs):
        super(DCTBlock, self).__init__()

        self.dct = DCT3D(freqs)

        self.branch_1 = nn.Sequential(conv_block(inp, oup, 2))
        self.branch_2 = nn.Sequential(conv_block(inp, oup, 2))
        self.branch_3 = nn.Sequential(conv_block(inp, oup, 2))
        self.branch_4 = nn.Sequential(conv_block(inp, oup, 2))
        self.weights = nn.Softmax(dim=1)

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        d = self.dct(x)
        d_1 = self.branch_1(d[0])
        d_2 = self.branch_2(d[1])
        d_3 = self.branch_3(d[2])
        d_4 = self.branch_4(d[3])
        d = torch.cat((d_1, d_2, d_3, d_4), 1)
        w = self.weights(d)
        x = x * w + x
        return x


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(
                in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate
            )
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class TestNet(nn.Module):
    def __init__(self, load_weights=False):
        super(TestNet, self).__init__()
        self.seen = 0
####################################        
        self.de_pred = nn.Sequential(
            Conv2d(1024, 512, 1, same_padding=True, NL="relu"),
            # Conv2d(128, 1, 1, same_padding=True, NL="relu"),
        )

        initialize_weights(self.modules())

        # res = models.resnet50(pretrained=pretrained)
        res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # pre_wts = torch.load(model_path)
        # res.load_state_dict(pre_wts)
        self.frontend = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2
        )
        self.own_reslayer_3 = make_res_layer(FcaBottleneck, 256, 6, stride=1)
        self.own_reslayer_3.load_state_dict(res.layer3.state_dict(), False)

        
#############################        
        # self.frontend_feat = [
        #     64,
        #     64,
        #     128,
        #     128,
        #     "M",
        #     256,
        #     256,
        #     256,
        #     "M",
        #     512,
        #     512,
        #     512,
        #     "M",
        # ]
        self.backend_feat1 = [512]
        self.backend_feat2 = [512, 512, 512, 256, 128, 64]
        
        # self.frontend = make_layers(self.frontend_feat)
        
        self.backend1 = []
        inp = 512
        freqs = [1, 16, 32, 64]
        for oup in self.backend_feat1:
            self.backend1.append(DCTBlock(inp, oup, freqs))
            inp = oup
        self.backend1 = nn.Sequential(*self.backend1)
        self.backend2 = make_layers(self.backend_feat2, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        # if not load_weights:
        #     mod = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        #     self._initialize_weights()
        #     # for i in xrange(len(self.frontend.state_dict().items())):
        #     # print(len(self.frontend.state_dict().items())) -> 20
        #     for i in range(len(self.frontend.state_dict().items())):
        #         list(self.frontend.state_dict().items())[i][1].data[:] = list(
        #             mod.state_dict().items()
        #         )[i][1].data[:]

    def forward(self, x):
        
############################        
        x = self.frontend(x)
        x = self.own_reslayer_3(x)
        x = self.de_pred(x)
############################        
        x = self.backend1(x)
        x = self.backend2(x)
        x = self.output_layer(x)
        x = F.interpolate(x, scale_factor=8)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class FcaBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        *,
        reduction=16,
    ):
        global _mapper_x, _mapper_y
        super(FcaBottleneck, self).__init__()
        # assert fea_h is not None
        # assert fea_w is not None
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.att = MultiSpectralAttentionLayer(
            planes * 4,
            c2wh[planes],
            c2wh[planes],
            reduction=reduction,
            freq_sel_method="top16",
        )

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
        out = self.att(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FcaBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        *,
        reduction=16,
    ):
        global _mapper_x, _mapper_y
        super(FcaBasicBlock, self).__init__()
        # assert fea_h is not None
        # assert fea_w is not None
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
        self.planes = planes
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.att = MultiSpectralAttentionLayer(
            planes,
            c2wh[planes],
            c2wh[planes],
            reduction=reduction,
            freq_sel_method="top16",
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.att(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



def get_freq_indices(method):
    assert method in [
        "top1",
        "top2",
        "top4",
        "top8",
        "top16",
        "top32",
        "bot1",
        "bot2",
        "bot4",
        "bot8",
        "bot16",
        "bot32",
        "low1",
        "low2",
        "low4",
        "low8",
        "low16",
        "low32",
    ]
    num_freq = int(method[3:])
    if "top" in method:
        all_top_indices_x = [
            0,
            0,
            6,
            0,
            0,
            1,
            1,
            4,
            5,
            1,
            3,
            0,
            0,
            0,
            3,
            2,
            4,
            6,
            3,
            5,
            5,
            2,
            6,
            5,
            5,
            3,
            3,
            4,
            2,
            2,
            6,
            1,
        ]
        all_top_indices_y = [
            0,
            1,
            0,
            5,
            2,
            0,
            2,
            0,
            0,
            6,
            0,
            4,
            6,
            3,
            5,
            2,
            6,
            3,
            3,
            3,
            5,
            1,
            1,
            2,
            4,
            2,
            1,
            1,
            3,
            0,
            5,
            3,
        ]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif "low" in method:
        all_low_indices_x = [
            0,
            0,
            1,
            1,
            0,
            2,
            2,
            1,
            2,
            0,
            3,
            4,
            0,
            1,
            3,
            0,
            1,
            2,
            3,
            4,
            5,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            1,
            2,
            3,
            4,
        ]
        all_low_indices_y = [
            0,
            1,
            0,
            1,
            2,
            0,
            1,
            2,
            2,
            3,
            0,
            0,
            4,
            3,
            1,
            5,
            4,
            3,
            2,
            1,
            0,
            6,
            5,
            4,
            3,
            2,
            1,
            0,
            6,
            5,
            4,
            3,
        ]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif "bot" in method:
        all_bot_indices_x = [
            6,
            1,
            3,
            3,
            2,
            4,
            1,
            2,
            4,
            4,
            5,
            1,
            4,
            6,
            2,
            5,
            6,
            1,
            6,
            2,
            2,
            4,
            3,
            3,
            5,
            5,
            6,
            2,
            5,
            5,
            3,
            6,
        ]
        all_bot_indices_y = [
            6,
            4,
            4,
            6,
            6,
            3,
            1,
            4,
            4,
            5,
            6,
            5,
            2,
            2,
            5,
            1,
            4,
            3,
            5,
            0,
            3,
            1,
            1,
            2,
            4,
            2,
            1,
            1,
            5,
            3,
            3,
            3,
        ]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method="top16"):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(
            dct_h, dct_w, mapper_x, mapper_y, channel
        )
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(
                x, (self.dct_h, self.dct_w)
            )
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer(
            "weight", self.get_dct_filter(height, width, mapper_x, mapper_y, channel)
        )

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, "x must been 4 dimensions, but got " + str(
            len(x.shape)
        )
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[
                        i * c_part : (i + 1) * c_part, t_x, t_y
                    ] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y
                    )

        return dct_filter



def make_res_layer(block, planes, blocks, stride=1):
    downsample = None
    inplanes = 512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)

