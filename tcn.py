import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from ECA_Net import ECA_Net
from self_attention import SelfAttention, MultiHeadSelfAttention


class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "If symmetric chomp, chomp size needs to be even"
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:  # 非因果卷积
            return x[:, :, self.chomp_size//2:-self.chomp_size//2].contiguous()
        else:  # 因果卷积
            return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding, False)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding, False)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        #print(x.shape)  # [8, 64, 5]
        out = self.net(x)  # input (N, C, L) L表示时间序列，这里是将28*28的图拆成784*1，每个时间序列的特征为1，共784个时间序列
        #print(out.shape)  # [8, 64, 5]
        res = x if self.downsample is None else self.downsample(x)
        #print(res.shape)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            k = kernel_size[i]
            layers += [TemporalBlock(in_channels, out_channels, k, stride=1, dilation=dilation_size,
                                     padding=(k-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# -----------------------------ECA+TCN----------------------------------------
class ECA_TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(ECA_TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding, False)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding, False)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.eca = ECA_Net()
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.eca(x)
        #print(x.shape)  # [8, 64, 8]
        out = self.net(out)  # input (N, C, L) L表示时间序列，这里是将28*28的图拆成784*1，每个时间序列的特征为1，共784个时间序列
        #print(out.shape)  # [8, 64, 8]
        res = x if self.downsample is None else self.downsample(x)
        #print(res.shape)
        return self.relu(out + res)

class ECA_TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.2):
        super(ECA_TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            k = kernel_size[i]
            layers += [ECA_TemporalBlock(in_channels, out_channels, k, stride=1, dilation=dilation_size,
                                     padding=(k - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# -----------------------------ECA+MS_TCN----------------------------------------
class ECA_MSTCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.2):
        super(ECA_MSTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = [(s - 1) * dilation_size for s in kernel_size]
            layers += [ECA_MultiscaleTemporalConvNetBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ECA_MultiscaleTemporalConvNetBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_sizes, stride, dilation, padding, dropout=0.2):
        super(ECA_MultiscaleTemporalConvNetBlock, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)
        self.n_outputs_branch = n_outputs // self.num_kernels
        assert n_outputs % self.num_kernels == 0, "Number of output channels needs to be divisible by number of kernels"
        for k_idx, k in enumerate(self.kernel_sizes):
            cbcr = SingleBlock(n_inputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx])
            setattr(self, 'cbcr0_{}'.format(k_idx), cbcr)  # setattr(object,name,value)设置属性值，用来存放单个卷积层
        self.dropout0 = nn.Dropout(dropout)

        for k_idx, k in enumerate(self.kernel_sizes):
            cbcr = SingleBlock(n_outputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx])
            setattr(self, 'cbcr1_{}'.format(k_idx), cbcr)
        self.dropout1 = nn.Dropout(dropout)

        self.eca = ECA_Net()
        # downsample?
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        # final relu
        self.relu = nn.ReLU()

    def forward(self, x):

        # first multi-branch set of convolutions
        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, 'cbcr0_{}'.format(k_idx))  # 将卷积层拿出来准备做卷积运算
            xx = self.eca(x)
            xx = branch_convs(xx)
            outputs.append(xx)  # [8,32,5]
        out0 = torch.cat(outputs, 1)  # 将同一层的两个卷积(k=3,k=5)结果进行拼接，恢复到64维特征
        # print(out0.shape)  # [8,64,5]
        out0 = self.dropout0(out0)

        # second multi-branch set of convolutions
        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, 'cbcr1_{}'.format(k_idx))
            outputs.append(branch_convs(out0))
        out1 = torch.cat(outputs, 1)
        out1 = self.dropout1(out1)

        # downsample?
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out1 + res)

# -----------------------------SelfAttention+TCN----------------------------------------
class SelfAttention_TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(SelfAttention_TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding, False)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding, False)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.selfAttention = SelfAttention(n_inputs, n_inputs, n_inputs)
        #self.selfAttention = MultiHeadSelfAttention(n_inputs, n_inputs, n_inputs)  # 多头注意力，默认8个头
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.selfAttention(x)
        #print(x.shape)  # [8, 64, 8]
        out = self.net(out)  # input (N, C, L) L表示时间序列，这里是将28*28的图拆成784*1，每个时间序列的特征为1，共784个时间序列
        #print(out.shape)  # [8, 64, 8]
        res = x if self.downsample is None else self.downsample(x)
        #print(res.shape)
        return self.relu(out + res)


class SelfAttention_TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.2):
        super(SelfAttention_TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            k = kernel_size[i]
            layers += [SelfAttention_TemporalBlock(in_channels, out_channels, k, stride=1, dilation=dilation_size,
                                     padding=(k - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MultiscaleTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.2):
        super(MultiscaleTemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = [(s - 1) * dilation_size for s in kernel_size]
            layers += [MultiscaleTemporalConvNetBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MultiscaleTemporalConvNetBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_sizes, stride, dilation, padding, dropout=0.2):
        super(MultiscaleTemporalConvNetBlock, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)
        self.n_outputs_branch = n_outputs // self.num_kernels
        assert n_outputs % self.num_kernels == 0, "Number of output channels needs to be divisible by number of kernels"
        for k_idx, k in enumerate(self.kernel_sizes):
            cbcr = SingleBlock(n_inputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx])
            setattr(self, 'cbcr0_{}'.format(k_idx), cbcr)  # setattr(object,name,value)设置属性值，用来存放单个卷积层
        self.dropout0 = nn.Dropout(dropout)

        for k_idx, k in enumerate(self.kernel_sizes):
            cbcr = SingleBlock(n_outputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx])
            setattr(self, 'cbcr1_{}'.format(k_idx), cbcr)
        self.dropout1 = nn.Dropout(dropout)

        # downsample?
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        # final relu
        self.relu = nn.ReLU()

    def forward(self, x):

        # first multi-branch set of convolutions
        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, 'cbcr0_{}'.format(k_idx))  # 将卷积层拿出来准备做卷积运算
            outputs.append(branch_convs(x))  # [8,32,5]
        out0 = torch.cat(outputs, 1)  # 将同一层的两个卷积(k=3,k=5)结果进行拼接，恢复到64维特征
        # print(out0.shape)  # [8,64,5]
        out0 = self.dropout0(out0)

        # second multi-branch set of convolutions
        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, 'cbcr1_{}'.format(k_idx))
            outputs.append(branch_convs(out0))
        out1 = torch.cat(outputs, 1)
        out1 = self.dropout1(out1)

        # downsample?
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out1 + res)

class SingleBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(SingleBlock, self).__init__()
        self.conv = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp = Chomp1d(padding, False)
        self.relu = nn.ReLU()
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv(x)
        out = self.chomp(out)
        out = self.relu(out)
        return out

# 单个卷积结构，dwpw的作用不知道是什么
class ConvBatchChompRelu(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, relu_type, dwpw=False):
        super(ConvBatchChompRelu, self).__init__()
        self.dwpw = dwpw
        if dwpw:
            self.conv = nn.Sequential(
                # -- dw
                nn.Conv1d(n_inputs, n_inputs, kernel_size, stride=stride,
                          padding=padding, dilation=dilation, groups=n_inputs, bias=False),
                nn.BatchNorm1d(n_inputs),
                Chomp1d(padding, True),
                nn.PReLU(num_parameters=n_inputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),
                # -- pw
                nn.Conv1d(n_inputs, n_outputs, 1, 1, 0, bias=False),
                nn.BatchNorm1d(n_outputs),
                nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                  stride=stride, padding=padding, dilation=dilation)
            self.batchnorm = nn.BatchNorm1d(n_outputs)
            self.chomp = Chomp1d(padding, True)
            self.non_lin = nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU()

    def forward(self, x):
        if self.dwpw:
            return self.conv(x)
        else:
            out = self.conv(x)
            out = self.batchnorm(out)
            out = self.chomp(out)
            return self.non_lin(out)


# --------- MULTI-BRANCH VERSION ---------------
# 单个block
class MultibranchTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_sizes, stride, dilation, padding, dropout=0.2,
                 relu_type='relu', dwpw=False):
        super(MultibranchTemporalBlock, self).__init__()

        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)
        self.n_outputs_branch = n_outputs // self.num_kernels
        assert n_outputs % self.num_kernels == 0, "Number of output channels needs to be divisible by number of kernels"

        for k_idx, k in enumerate(self.kernel_sizes):
            cbcr = ConvBatchChompRelu(n_inputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type,
                                      dwpw=dwpw)
            setattr(self, 'cbcr0_{}'.format(k_idx), cbcr)  # setattr(object,name,value)设置属性值，用来存放单个卷积层
        self.dropout0 = nn.Dropout(dropout)

        for k_idx, k in enumerate(self.kernel_sizes):
            cbcr = ConvBatchChompRelu(n_outputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type,
                                      dwpw=dwpw)
            setattr(self, 'cbcr1_{}'.format(k_idx), cbcr)
        self.dropout1 = nn.Dropout(dropout)

        # downsample?
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if (n_inputs // self.num_kernels) != n_outputs else None

        # final relu
        if relu_type == 'relu':
            self.relu_final = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu_final = nn.PReLU(num_parameters=n_outputs)

    def forward(self, x):

        # first multi-branch set of convolutions
        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, 'cbcr0_{}'.format(k_idx))  # 将卷积层拿出来准备做卷积运算
            outputs.append(branch_convs(x))  # [8,32,5]
        out0 = torch.cat(outputs, 1)  # 将同一层的两个卷积(k=3,k=5)结果进行拼接，恢复到64维特征
        # print(out0.shape)  # [8,64,5]
        out0 = self.dropout0(out0)

        # second multi-branch set of convolutions
        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, 'cbcr1_{}'.format(k_idx))
            outputs.append(branch_convs(out0))
        out1 = torch.cat(outputs, 1)
        out1 = self.dropout1(out1)

        # downsample?
        res = x if self.downsample is None else self.downsample(x)

        return self.relu_final(out1 + res)


class MultibranchTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.2, relu_type='relu', dwpw=False):
        super(MultibranchTemporalConvNet, self).__init__()

        ksizes = kernel_size
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            padding = [(s - 1) * dilation_size for s in ksizes]
            layers.append(MultibranchTemporalBlock(in_channels, out_channels, ksizes,
                                                   stride=1, dilation=dilation_size, padding=padding, dropout=dropout,
                                                   relu_type=relu_type,
                                                   dwpw=dwpw))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    # --------------------------------
