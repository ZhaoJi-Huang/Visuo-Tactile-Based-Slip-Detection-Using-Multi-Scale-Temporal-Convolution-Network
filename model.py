# -*- coding: utf-8 -*-
import torch
import torchvision as tv
from torch import nn
from tcn import TemporalConvNet, MultibranchTemporalConvNet, MultiscaleTemporalConvNet, ECA_TCN, SelfAttention_TCN, ECA_MSTCN
import os
import numpy as np
from torchvision.models import vgg19_bn, vgg16_bn, inception_v3, alexnet


# ---------------------------------resnet18-------------------------------------
class resnet18(nn.Module):
    def __init__(self, pretrained=True):
        super(resnet18, self).__init__()
        self.resnet = tv.models.resnet18(pretrained=pretrained)
        # for name, value in self.resnet.named_parameters():
        #     if (name != 'fc.weight') and (name != 'fc.bias'):
        #         value.requires_grad = False

        for name, value in self.resnet.named_parameters():
            # if name.find("bn") != -1:
            #    value.requires_grad = False
            # if name.find("fc") != -1:
            #    value.requires_grad = False
            value.requires_grad = False

    def forward(self, input):
        x = self.resnet.conv1(input)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        #x = self.resnet.avgpool(x)
        x = x.flatten(start_dim=1)
        return x

# ---------------------------------resnet34------------------------------------
class resnet34(nn.Module):
    def __init__(self, pretrained=True):
        super(resnet34, self).__init__()
        self.resnet = tv.models.resnet34(pretrained=pretrained)
        # for name, value in self.resnet.named_parameters():
        #     if (name != 'fc.weight') and (name != 'fc.bias'):
        #         value.requires_grad = False

        for name, value in self.resnet.named_parameters():
            # if name.find("bn") != -1:
            #    value.requires_grad = False
            # if name.find("fc") != -1:
            #    value.requires_grad = False
            value.requires_grad = False

    def forward(self, input):
        x = self.resnet.conv1(input)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        #x = self.resnet.avgpool(x)
        x = x.flatten(start_dim=1)
        return x

# ---------------------------------resnet50-------------------------------------
class resnet50(nn.Module):
    def __init__(self, pretrained=True):
        super(resnet50, self).__init__()
        self.resnet = tv.models.resnet50(pretrained=pretrained)
        # for name, value in self.resnet.named_parameters():
        #     if (name != 'fc.weight') and (name != 'fc.bias'):
        #         value.requires_grad = False

        for name, value in self.resnet.named_parameters():
            # if name.find("bn") != -1:
            #    value.requires_grad = False
            # if name.find("fc") != -1:
            #    value.requires_grad = False
            value.requires_grad = False

    def forward(self, input):
        x = self.resnet.conv1(input)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        #x = self.resnet.avgpool(x)
        x = x.flatten(start_dim=1)
        return x

# ---------------------------------------vgg16--------------------------------------------
class vgg16_network(nn.Module):
    def __init__(self):
        super(vgg16_network, self).__init__()
        # Define CNN to extract features.
        self.vgg = tv.models.vgg16(pretrained=True)
        # self.vgg.classifier = nn.Sequential()
        # To delete fc8
        # self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

        for name, value in self.vgg.named_parameters():
            value.requires_grad = False

        # self.vgg.classifier = torch.nn.Sequential(
        #     nn.Linear(512*7*7, 1024),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(1024, 64),
        #     nn.ReLU()
        # )


    def forward(self, input):
        x = self.vgg.features(input)  # [512,7,7]
        #x = self.vgg.avgpool(x)  # [512,7,7]
        x = torch.flatten(x, 1)

        return x

# ---------------------------------------inceptionv3--------------------------------------------
class inceptionv3(nn.Module):
    def __init__(self):
        super(inceptionv3, self).__init__()
        # Define CNN to extract features.
        self.inception = tv.models.inception_v3(pretrained=True)

        for name, value in self.inception.named_parameters():
            value.requires_grad = False

    def forward(self, input):
        # N x 3 x 299 x 299
        x = self.inception.Conv2d_1a_3x3(input)
        # N x 32 x 149 x 149
        x = self.inception.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6e(x)
        # N x 768 x 17 x 17
        #aux_defined = self.inception.training and self.inception.aux_logits
        # if aux_defined:
        #     aux = self.inception.AuxLogits(x)
        # else:
        #     aux = None
        # N x 768 x 17 x 17
        x = self.inception.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception.Mixed_7b(x)
        # N x 2048 x 8 x 8  input(224,224)时N x 2048 x 5 x 5
        x = self.inception.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        #x = self.inception.avgpool(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)

        return x

# ----------------------------------------TCN-----------------------------------------------------
class MS_TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(MS_TCN, self).__init__()
        self.tcn = MultiscaleTemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        out = self.tcn(inputs)  # input should have dimension (N, C, L)
        return out

class MSTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(MSTCN, self).__init__()
        self.tcn = MultibranchTemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        # print(y1.shape)  # [8, 64, 13]
        # print(y1[:, :, -1].shape)  # [8, 64]
        o = self.linear(y1[:, :, -1])
        return o

class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        # input_size=1,num_channels = [25 25 25 25 25 25 25 25],kernel_size=7,dropout=0.05
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        # self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        out = self.tcn(inputs)  # input should have dimension (N, C, L)
        # print(y1.shape)  # [8, 64, 13]
        # print(y1[:, :, -1].shape)  # [8, 64]
        # o = self.linear(y1[:, :, -1])
        return out

class TCN_ECA(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCN_ECA, self).__init__()
        self.tcn = ECA_TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        # self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        out = self.tcn(inputs)  # input should have dimension (N, C, L)
        # print(y1.shape)  # [8, 64, 13]
        # print(y1[:, :, -1].shape)  # [8, 64]
        # o = self.linear(y1[:, :, -1])
        return out

class MSTCN_ECA(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(MSTCN_ECA, self).__init__()
        self.tcn = ECA_MSTCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        out = self.tcn(inputs)  # input should have dimension (N, C, L)
        return out

class TCN_SelfAttention(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCN_SelfAttention, self).__init__()
        self.tcn = SelfAttention_TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        out = self.tcn(inputs)  # input should have dimension (N, C, L)
        return out

# ---------------------------------visual-only，LSTM-------------------------------------
class resnet_lstm(nn.Module):
    def __init__(self, num_layers=2, num_classes=2):
        super(resnet_lstm, self).__init__()
        self.cnn = tv.models.resnet18(pretrained=True)
        self.cnn.eval()
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=num_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(512*7*7, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, inputs):
        cnn_output_list = list()
        for t in range(inputs.size(1)):
            #print(inputs[:, t, :, :, :].shape)  # [8, 3, 224, 224]
            cnn_output = resnet_lstm.extractFeature(self, inputs[:, t, :, :, :])
            #print(cnn_output.shape)  # [8, 25088]

            cnn_output = self.dropout(cnn_output)
            cnn_output = self.relu(self.fc1(cnn_output))
            #print(cnn_output.shape)  # [8, 1024]
            cnn_output = self.dropout(cnn_output)
            cnn_output = self.relu(self.fc2(cnn_output))
            #print(cnn_output.shape)  # [8, 64]
            cnn_output_list.append(cnn_output)

        x = torch.stack(tuple(cnn_output_list), dim=1)
        #print(x.shape)  # [8, 13, 64]
        out, hidden = self.lstm(x)
        #print(out.shape)  # [8, 13, 64]
        x = out[:, -1, :]
        #print(x.shape)  # [8, 64]
        x = self.relu(x)
        x = self.fc3(x)
        #print(x.shape)  # [8, 2]
        return x

    def extractFeature(self, input):
        # print('feature extracting start')
        n = self.cnn
        with torch.no_grad():
            x = n.conv1(input)
            x = n.bn1(x)
            x = n.relu(x)
            x = n.maxpool(x)
            x = n.layer1(x)
            x = n.layer2(x)
            x = n.layer3(x)
            x = n.layer4(x)
            x = x.flatten(start_dim=1)

        return x

# -------------------------------tactile-only，LSTM------------------------------
class cnn_tactile(nn.Module):
    def __init__(self):
        super(cnn_tactile, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # [8,4,4]
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # [8,2,2]

        #self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1))  # [16,2,2]
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # [16,2,2]
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # [16,1,1]

        #self.conv3 = nn.Conv2d(16, 32, kernel_size=(2, 2), stride=(1, 1))  # [32,1,1]
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1))  # [32,1,1]
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))  # [32,1,1]

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))  # [8, 8, 4, 4]
        x = self.pool1(x)  # [8, 8, 2, 2]
        # print(x.shape)  # [8, 8, 4, 4]

        x = self.relu(self.conv2(x))  # [8, 16, 2, 2]
        x = self.pool2(x)  # [8, 16, 1, 1]
        # print(x.shape)  # [8, 16, 2, 2]

        x = self.relu(self.conv3(x))  # [8, 32, 1, 1]
        x = self.pool3(x)  # [8, 32, 1, 1]
        # print(x.shape)  # [8, 32, 1, 1]

        x = x.flatten(start_dim=1)
        # print(x.shape)  # [8, 32]

        return x

class cnn_tactile_lstm(nn.Module):
    def __init__(self, num_layers=2, num_classes=2):
        super(cnn_tactile_lstm, self).__init__()
        self.cnn = cnn_tactile()
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=num_layers,
                            batch_first=True)  # 32,64
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(32, 64)
        self.fc1 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()


    def forward(self, inputs):
        cnn_output_list = list()
        for t in range(inputs.size(1)):
            # print(inputs[:, t, :, :, :].shape)  # [8, 3, 4, 4]
            cnn_output = self.cnn(inputs[:, t, :, :, :])
            cnn_output = self.dropout(cnn_output)
            cnn_output = self.relu(self.fc(cnn_output))
            # print(cnn_output.shape)  # [8, 64]
            cnn_output_list.append(cnn_output)

        x = torch.stack(tuple(cnn_output_list), dim=1)
        # print(x.shape)  # [8, 13, 64]
        out, hidden = self.lstm(x)
        # print(out.shape)  # [8, 13, 64]
        x = out[:, -1, :]
        # print(x.shape)  # [8, 64]
        x = self.relu(x)
        x = self.fc1(x)
        # print(x.shape)  # [8, 2]
        return x

# -------------------------------visual-onlt，TCN-----------------------------------------
class v_resnet_tcn(nn.Module):
    def __init__(self, input_size=64, num_channels=[64, 64, 64], kernel_size=[3, 3, 3], dropout=0.2, num_classes=2):
        super(v_resnet_tcn, self).__init__()
        #self.cnn = resnet18()
        self.cnn = resnet34()
        #self.cnn = resnet50()
        #self.cnn = vgg16_network()
        #self.cnn = self.cnn = inceptionv3()

        num_channels = [64, 64]
        kernel_size = [5, 5]
        #self.tcn = TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.tcn = MS_TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        #self.tcn = TCN_ECA(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        #self.tcn = TCN_SelfAttention(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)

        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 64)
        #self.fc1 = nn.Linear(2048 * 7 * 7, 1024)  # (512*7*7, 1024)
        #self.fc2 = nn.Linear(1024, 64)
        #self.fc = nn.Linear(512 * 7 * 7, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, inputs):
        cnn_output_list = list()
        for t in range(inputs.size(1)):
            cnn_output = self.cnn(inputs[:, t, :, :, :])

            cnn_output = self.dropout(cnn_output)
            cnn_output = self.relu(self.fc1(cnn_output))
            # print(cnn_output.shape)  # [4, 1024]
            cnn_output = self.dropout(cnn_output)
            cnn_output = self.relu(self.fc2(cnn_output))
            # print(cnn_output.shape)  # [4, 64]
            #cnn_output = self.dropout(cnn_output)
            #cnn_output = self.relu(self.fc(cnn_output))
            cnn_output_list.append(cnn_output)

        x = torch.stack(tuple(cnn_output_list), dim=2)  # [8, 64, 13]
        out = self.tcn(x)  # [8, 64, 13]
        out = out[:, :, -1]  # [8, 64]
        out = self.relu(out)
        out = self.fc3(out)  # [8, 2]
        return out

# -------------------------------tactile-only，TCN---------------------------------------------
class t_cnn_tcn(nn.Module):
    def __init__(self, input_size=64, num_channels=[64, 64, 64], kernel_size=[3, 3, 3], dropout=0.2, num_classes=2):
        super(t_cnn_tcn, self).__init__()
        self.cnn = cnn_tactile()
        #self.cnn = resnet18()
        num_channels = [64, 64]
        kernel_size = [5, 5]
        input_size=64
        #self.tcn = TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.tcn = MS_TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        #self.tcn = TCN_ECA(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        #self.tcn = TCN_SelfAttention(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)

        self.fc = nn.Linear(32, 64)
        self.fc1 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, inputs):
        cnn_output_list = list()
        for t in range(inputs.size(1)):
            #print(inputs[:, t, :, :, :].shape)  # [8, 3, 4, 4]
            cnn_output = self.cnn(inputs[:, t, :, :, :])
            #print(cnn_output.shape)  # [8, 32]
            cnn_output = self.dropout(cnn_output)
            cnn_output = self.relu(self.fc(cnn_output))
            #print(cnn_output.shape)  # [8, 64]
            cnn_output_list.append(cnn_output)

        x = torch.stack(tuple(cnn_output_list), dim=2)  # [4, 64, 13]
        out = self.tcn(x)  # [8, 64, 13]
        out = out[:, :, -1]  # [8, 64]
        out = self.relu(out)
        out = self.fc1(out)  # [8, 2]
        return out


# ---------------------------------visual-tactile fusion，LSTM----------------------------------------------
class vt_resnet_cnn_lstm(nn.Module):
    def __init__(self, num_layers=2, num_classes=2):
        super(vt_resnet_cnn_lstm, self).__init__()
        #self.cnn1 = resnet18()
        self.cnn1 = resnet34()
        self.cnn2 = cnn_tactile()
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=num_layers,
                            batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(512*7*7, 1024)  # (512*7*7, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, num_classes)  # 128
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, v_inputs, t_inputs):
        cnn_output_list = list()
        for t in range(v_inputs.size(1)):
            v_cnn_output = self.cnn1(v_inputs[:, t, :, :, :])  # [8, 25088]
            t_cnn_output = self.cnn2(t_inputs[:, t, :, :, :])  # [8, 32]

            v_cnn_output = self.dropout(v_cnn_output)
            v_cnn_output = self.relu(self.fc1(v_cnn_output))  # [8, 1024]
            v_cnn_output = self.dropout(v_cnn_output)
            v_cnn_output = self.relu(self.fc2(v_cnn_output))  # [8, 64]

            t_cnn_output = self.dropout(t_cnn_output)
            t_cnn_output = self.relu(self.fc3(t_cnn_output))  # [8, 64]

            cnn_output = torch.cat((v_cnn_output, t_cnn_output), 1)  # 64+64=128，[8, 128]
            cnn_output_list.append(cnn_output)

        x = torch.stack(tuple(cnn_output_list), dim=1)  # [8, 13, 128]
        out, hidden = self.lstm(x)  # [8, 13, 64]
        x = out[:, -1, :]  # [8, 64]
        x = self.relu(x)
        x = self.fc4(x)  # [8, 2]
        return x

# ---------------------------------visual-tactile fusion，TCN----------------------------------------------
class vt_resnet_tcn(nn.Module):
    def __init__(self, num_classes=2):
        super(vt_resnet_tcn, self).__init__()
        self.cnn1 = resnet18()
        self.cnn2 = cnn_tactile()
        input_size = 128
        num_channels = [64, 64, 64]
        kernel_size = [3, 3, 3]
        dropout = 0.2
        self.tcn = TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc1 = nn.Linear(512*7*7, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, num_classes)  # 128
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, v_inputs, t_inputs):
        cnn_output_list = list()
        for t in range(v_inputs.size(1)):
            # print(inputs[:, t, :, :, :].shape)
            v_cnn_output = self.cnn1(v_inputs[:, t, :, :, :])
            t_cnn_output = self.cnn2(t_inputs[:, t, :, :, :])

            v_cnn_output = self.dropout(v_cnn_output)
            v_cnn_output = self.relu(self.fc1(v_cnn_output))
            # print(cnn_output.shape)
            v_cnn_output = self.dropout(v_cnn_output)
            v_cnn_output = self.relu(self.fc2(v_cnn_output))
            #print(v_cnn_output.shape)

            t_cnn_output = self.dropout(t_cnn_output)
            t_cnn_output = self.relu(self.fc3(t_cnn_output))

            cnn_output = torch.cat((v_cnn_output, t_cnn_output), 1)  # 64+64=128
            cnn_output_list.append(cnn_output)

        x = torch.stack(tuple(cnn_output_list), dim=2)
        # print(x.shape)  # [8, 64, 13]
        out = self.tcn(x)
        out = out[:, :, -1]
        out = self.relu(out)
        out = self.fc4(out)
        #print(out.shape)  # [8,2]
        return out

# ------------------------------cnn_mstcn---------------------------------------------
class vt_resnet_tcn_tcn(nn.Module):
    def __init__(self, num_classes=2):
        super(vt_resnet_tcn_tcn, self).__init__()
        input_size = 64
        num_channels = [64, 64]
        kernel_size = [5, 5]
        Mnum_channels = [64, 64, 64]
        Mkernel_size = [3, 3]
        dropout = 0.2
        self.v_resnet_tcn = resnet_tcn(input_size, num_channels, kernel_size, dropout)
        self.t_cnn_tcn = cnn_tcn(input_size, num_channels, kernel_size, dropout)

        #self.tcn = TCN(input_size * 2, Mnum_channels, kernel_size=Mkernel_size, dropout=dropout)
        self.tcn = MS_TCN(input_size * 2, Mnum_channels, kernel_size=Mkernel_size, dropout=dropout)
        #self.tcn = TCN_ECA(input_size * 2, Mnum_channels, kernel_size=Mkernel_size, dropout=dropout)
        #self.tcn = MSTCN_ECA(input_size * 2, Mnum_channels, kernel_size=Mkernel_size, dropout=dropout)
        #self.tcn = TCN_SelfAttention(input_size * 2, Mnum_channels, kernel_size=Mkernel_size, dropout=dropout)

        self.fc = nn.Linear(64, num_classes)
        #self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, v_inputs, t_inputs):
        v_out = self.v_resnet_tcn(v_inputs)
        t_out = self.t_cnn_tcn(t_inputs)
        # print(t_out.shape)  # [8, 64, 13]

        vt_output = torch.cat((v_out, t_out), 1)
        #vt_output = vt_output[:, :, -1]
        # print(vt_output.shape)  # [8, 128, 13]
        out = self.tcn(vt_output)
        out = out[:, :, -1]
        #out = self.relu(self.fc(vt_output))
        # print(out.shape)  # [8, 64]
        out = self.relu(out)
        out = self.fc(out)
        # print(out.shape)  # [8,2]
        return out

class resnet_tcn(nn.Module):
    def __init__(self, input_size=64, num_channels=[64, 64, 64], kernel_size=[3, 3, 3], dropout=0.2):
        super(resnet_tcn, self).__init__()
        #self.cnn = resnet18()
        self.cnn = resnet34()
        #self.cnn = resnet50()
        #self.cnn = vgg16_network()
        #self.cnn = inceptionv3()

        #self.tcn = TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.tcn = MS_TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        #self.tcn = TCN_ECA(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        #self.tcn = MSTCN_ECA(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        #self.tcn = TCN_SelfAttention(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)

        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 64)
        #self.fc1 = nn.Linear(2048 * 5 * 5, 1024)  # (2048 * 7 * 7, 1024)
        #self.fc2 = nn.Linear(1024, 64)
        #self.fc = nn.Linear(512 * 7 * 7, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, inputs):
        cnn_output_list = list()
        for t in range(inputs.size(1)):
            cnn_output = self.cnn(inputs[:, t, :, :, :])

            cnn_output = self.dropout(cnn_output)
            cnn_output = self.relu(self.fc1(cnn_output))
            # print(cnn_output.shape)
            cnn_output = self.dropout(cnn_output)
            cnn_output = self.relu(self.fc2(cnn_output))
            # print(cnn_output.shape)
            #cnn_output = self.dropout(cnn_output)
            #cnn_output = self.relu(self.fc(cnn_output))
            cnn_output_list.append(cnn_output)

        x = torch.stack(tuple(cnn_output_list), dim=2)  
        out = self.tcn(x)
        return out

class cnn_tcn(nn.Module):
    def __init__(self, input_size=64, num_channels=[64, 64, 64], kernel_size=[3, 3, 3], dropout=0.2):
        super(cnn_tcn, self).__init__()
        self.cnn = cnn_tactile()

        #self.tcn = TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.tcn = MS_TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        #self.tcn = TCN_ECA(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        #self.tcn = MSTCN_ECA(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        #self.tcn = TCN_SelfAttention(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)

        self.fc = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, inputs):
        cnn_output_list = list()
        for t in range(inputs.size(1)):
            # print(inputs[:, t, :, :, :].shape)  # [8, 3, 4, 4]
            cnn_output = self.cnn(inputs[:, t, :, :, :])
            cnn_output = self.dropout(cnn_output)
            cnn_output = self.relu(self.fc(cnn_output))
            # print(cnn_output.shape)
            cnn_output_list.append(cnn_output)

        x = torch.stack(tuple(cnn_output_list), dim=2)
        out = self.tcn(x)
        return out


