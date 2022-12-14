import torch
from torch import nn
from torch.nn.parameter import Parameter


class ECA_Net(nn.Module):

    def __init__(self, k_size=3):
        super(ECA_Net, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        print('=> Using ECA attention')

    def forward(self, x):
        # 将[n,c,t]转为[n,t,c]对c维度进行池化
        y = x.transpose(-1, -2).contiguous()

        # 池化后变成[n,t,1]
        y = self.avg_pool(y)

        # 将[n,t,1]改为[n,1,t]后进行ECA计算
        y = self.conv(y.transpose(-1, -2).contiguous())
        # Multi-scale information fusion
        y = self.sigmoid(y)

        #return x * y.expand_as(x)
        return x * y + x
