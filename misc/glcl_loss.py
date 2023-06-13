import torch
import torch.nn as nn
import torch.nn.functional as F


class GLCL(nn.Module):
    def __init__(self):
        super(GLCL, self).__init__()

    def forward(self, input, target):
        T = 170
        # T = 0.0026

        # (batch_size, channel, _, _) = input.size()
        batch_size, channel, _, _ = input.shape
        target = target.unsqueeze(1)
        # print(input.shape, target.shape)
        # print(input.size(), target.size())
        # density map dimension 1
        # torch.Size([8, 1, 576, 768]) torch.Size([8, 576, 768]) 
        # torch.Size([8, 1, 576, 768]) torch.Size([8, 1, 576, 768])

        h_x = (
            torch.tensor(
                [[1.0 / 3, 0, -1.0 / 3], [1.0 / 3, 0, -1.0 / 3], [1.0 / 3, 0, -1.0 / 3]]
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # 1 1 3 3 dimension value equal 1 expand
        h_x = h_x.expand(batch_size, channel, 3, 3).contiguous()
        # print(h_x.shape, h_x.shape)
        # contiguous() deep copy
        # h_y = h_x.transpose(2, 3)
        h_y = h_x.transpose(2, 3)
        # print(h_y.shape, h_y.shape)
        
        
        # if input.is_cuda:
        #     h_x = h_x.cuda(input.get_device())
        #     h_y = h_y.cuda(input.get_device())
        h_x = h_x.cuda()
        h_y = h_y.cuda()
            
        # feat1 = F.avg_pool2d(input, kernel_size=2, stride=2)
        # feat2 = F.avg_pool2d(target, kernel_size=2, stride=2)
        feat1 = input # 8 1 
        feat2 = target

        feat1_x = F.conv2d(feat1, h_x, padding=1)
        feat1_y = F.conv2d(feat1, h_y, padding=1)
        grad1 = torch.sqrt(feat1_x.pow(2) + feat1_y.pow(2))

        feat2_x = F.conv2d(feat2, h_x, padding=1)
        feat2_y = F.conv2d(feat2, h_y, padding=1)
        grad2 = torch.sqrt(feat2_x.pow(2) + feat2_y.pow(2))

        gms = (2 * grad1 * grad2 + T) / (grad1.pow(2) + grad2.pow(2) + T)
        err = abs(input - target)
        alpha = 1.313
        beta = 1.0
        err = alpha * torch.log(torch.cosh(err) + beta * (1 - gms))

        return err.mean()
