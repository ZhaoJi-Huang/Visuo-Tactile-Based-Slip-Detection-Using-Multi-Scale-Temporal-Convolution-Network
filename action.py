import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb


class Action(nn.Module):
    def __init__(self, net, n_segment=3, shift_div=8):
        super(Action, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.in_channels = self.net.in_channels
        self.out_channels = self.net.out_channels
        self.kernel_size = self.net.kernel_size
        self.stride = self.net.stride
        self.padding = self.net.padding
        self.reduced_channels = self.in_channels//16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fold = self.in_channels // shift_div

        # shifting
        self.action_shift = nn.Conv1d(
                                    self.in_channels, self.in_channels,
                                    kernel_size=3, padding=1, groups=self.in_channels,
                                    bias=False)      
        self.action_shift.weight.requires_grad = True
        self.action_shift.weight.data.zero_()
        self.action_shift.weight.data[:self.fold, 0, 2] = 1 # shift left
        self.action_shift.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right  


        if 2*self.fold < self.in_channels:
            self.action_shift.weight.data[2 * self.fold:, 0, 1] = 1 # fixed


        # # spatial temporal excitation
        self.action_p1_conv1 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), 
                                    stride=(1, 1 ,1), bias=False, padding=(1, 1, 1))       


        # # channel excitation
        self.action_p2_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.action_p2_conv1 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=3, stride=1, bias=False, padding=1, 
                                       groups=1)
        self.action_p2_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
    


        # motion excitation
        self.pad = (0,0,0,0,0,0,0,1)
        self.action_p3_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.action_p3_bn1 = nn.BatchNorm2d(self.reduced_channels)
        self.action_p3_conv1 = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 3), 
                                    stride=(1 ,1), bias=False, padding=(1, 1), groups=self.reduced_channels)
        self.action_p3_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        print('=> Using ACTION')


    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment

        x_shift = x.view(n_batch, self.n_segment, c, h, w)  # [N,T,C,H,W]
        x_shift = x_shift.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, n_segment)
        x_shift = x_shift.contiguous().view(n_batch*h*w, c, self.n_segment)  # [NHW,C,T]
        x_shift = self.action_shift(x_shift)  # (n_batch*h*w, c, n_segment)，这一步的shift作用暂时不清楚
        x_shift = x_shift.view(n_batch, h, w, c, self.n_segment)  # [N,H,W,C,T]
        x_shift = x_shift.permute([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
        x_shift = x_shift.contiguous().view(nt, c, h, w)  # [NT,C,H,W]


        # 3D convolution: c*T*h*w, spatial temporal excitation
        nt, c, h, w = x_shift.size()
        # 转换t和C的维度位置，transpose(2,1)
        x_p1 = x_shift.view(n_batch, self.n_segment, c, h, w).transpose(2,1).contiguous()  # [N,C,T,H,W]
        # 计算维度C的均值，
        x_p1 = x_p1.mean(1, keepdim=True)  # [N,1,T,H,W]
        x_p1 = self.action_p1_conv1(x_p1)  # 3D Conv,[N,1,T,H,W]
        x_p1 = x_p1.transpose(2,1).contiguous().view(nt, 1, h, w)  # [NT,1,H,W]
        x_p1 = self.sigmoid(x_p1)
        x_p1 = x_shift * x_p1 + x_shift


        # 2D convolution: c*T*1*1, channel excitation
        x_p2 = self.avg_pool(x_shift)  # 平均池化,[NT,C,1,1]
        x_p2 = self.action_p2_squeeze(x_p2)  # 用1*1的2D Conv来降维，[NT,C/16,1,1]
        nt, c, h, w = x_p2.size()
        # squeeze(-1)去掉最后一个维度，[N,C/16,T]
        x_p2 = x_p2.view(n_batch, self.n_segment, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2,1).contiguous()
        x_p2 = self.action_p2_conv1(x_p2)  # 1D Conv,[N,C/16,T]
        x_p2 = self.relu(x_p2)
        x_p2 = x_p2.transpose(2,1).contiguous().view(-1, c, 1, 1)  # [NT,C/16,1,1]
        x_p2 = self.action_p2_expand(x_p2)  # 用1*1的2D Conv来升维，[NT,C,1,1]
        x_p2 = self.sigmoid(x_p2)
        x_p2 = x_shift * x_p2 + x_shift
        

        # # 2D convolution: motion excitation
        x3 = self.action_p3_squeeze(x_shift)  # 用1*1的2D Conv来降维，[NT,C/16,H,W]
        #print(x3.shape)  # [64, 4, 56, 56]
        x3 = self.action_p3_bn1(x3)  # BN
        nt, c, h, w = x3.size()
        x3_plus0, _ = x3.view(n_batch, self.n_segment, c, h, w).split([self.n_segment-1, 1], dim=1)  # 取前T-1个时间段的特征，[N,T-1,C/16,H,W]
        #print(x3_plus0.shape)  # [8, 7, 4, 56, 56]
        x3_plus1 = self.action_p3_conv1(x3)  # 3*3的2D Conv，[NT,C/16,H,W]
        #print(x3_plus1.shape)  # [64, 4, 56, 56]

        _ , x3_plus1 = x3_plus1.view(n_batch, self.n_segment, c, h, w).split([1, self.n_segment-1], dim=1)  # 取第一个后面的T-1个时间段的特征，[N,T-1,C/16,H,W]
        #print(x3_plus1.shape)  # [8, 7, 4, 56, 56]
        x_p3 = x3_plus1 - x3_plus0
        #print(x_p3.shape)
        x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)  # [N,T,C/16,H,W]
        #print(x_p3.shape)  # [8, 8, 4, 56, 56]
        x_p3 = self.avg_pool(x_p3.view(nt, c, h, w))  # 平均池化,[NT,C/16,1,1]
        #print(x_p3.shape)
        x_p3 = self.action_p3_expand(x_p3)  # 用1*1的2D Conv来升维，[NT,C,1,1]
        #print(x_p3.shape)
        x_p3 = self.sigmoid(x_p3)
        x_p3 = x_shift * x_p3 + x_shift

        out = self.net(x_p1 + x_p2 + x_p3)
        return out





class TemporalPool(nn.Module):
    def __init__(self, net, n_segment):
        super(TemporalPool, self).__init__()
        self.net = net
        self.n_segment = n_segment

    def forward(self, x):
        x = self.temporal_pool(x, n_segment=self.n_segment)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, n_segment):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        return x


def make_temporal_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))


    # pdb.set_trace()
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        if place == 'block':
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i].conv1 = Action(b.conv1, n_segment=this_segment, shift_div = n_div)
                return nn.Sequential(*(blocks))

            pdb.set_trace()
            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = Action(b.conv1, n_segment=this_segment, shift_div = n_div)
                        # pdb.set_trace()
                return nn.Sequential(*blocks)

            # pdb.set_trace()
            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
             
    else:
        raise NotImplementedError(place)


def make_temporal_pool(net, n_segment):
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError






