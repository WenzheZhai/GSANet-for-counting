import torch.nn as nn
import torch
from torchvision import models
import math
import torch.nn.functional as F


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


class FFPNet(nn.Module):
    def __init__(self, load_weights=False):
        super(FFPNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [
            64,
            64,
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            "M",
        ]
        self.backend_feat1 = [512]
        self.backend_feat2 = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend1 = []
        inp = 512
        freqs = [1, 16, 32, 64]
        for oup in self.backend_feat1:
            self.backend1.append(DCTBlock(inp, oup, freqs))
            inp = oup
        self.backend1 = nn.Sequential(*self.backend1)
        self.backend2 = make_layers(self.backend_feat2, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self._initialize_weights()
            # for i in xrange(len(self.frontend.state_dict().items())):
            # print(len(self.frontend.state_dict().items())) -> 20
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(
                    mod.state_dict().items()
                )[i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)
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


# contiguous() 深拷贝
