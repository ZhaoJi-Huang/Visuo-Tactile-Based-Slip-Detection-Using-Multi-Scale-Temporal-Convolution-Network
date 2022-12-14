# -*- coding: utf-8 -*-
import os
import torch
import re
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()


def load_data(path, label_files, num, init, length):
    datas = []
    pat = re.compile(r'object([0-9]+)_result')
    label_file = label_files[num]
    #print(label_file)
    idx = pat.search(label_file).group(1)
    fp = open(label_file, 'r')
    lines = fp.readlines()
    datas.extend([line.replace('\n', '') + ' ' + idx for line in lines])

    datasets = []
    for data in datas:
        train_data = data.split(' ')
        object_id = train_data[-1]  # '001','002'......
        start_index = train_data[-4]  # 5
        end_index = train_data[-3]  # 25
        id_2 = train_data[-2]  # 680_mm
        status = train_data[2]  # Label

        data_path = os.path.join(path, 'object' + object_id, id_2, 'image')
        rgb_img_paths = []
        for root, dirs, files in os.walk(data_path, topdown=True):
            files.sort(key=lambda x: int(x[-12:-4]))
            for file in files:
                if file.endswith('.jpg'):
                    rgb_img_paths.append(os.path.join(root, file))
                    # if start_index in file:
                    #     index = len(rgb_img_paths) - 1
                    # if end_index in file:
                    #     index_end = len(rgb_img_paths) - 1
        index = int(start_index)
        index_end = int(end_index)
        init = index_end - index - length + 2
        for i in range(init):
            rowTemp = []
            visual = []
            tactile = []
            rowTemp.append(object_id)
            rowTemp.append(status)
            for j in range(index, index+length):
                visual_path = rgb_img_paths[j]
                visual.append(visual_path)
                tactile_path = visual_path.replace('image', 'xela1')
                tactile.append(tactile_path)
            rowTemp.append(visual)
            rowTemp.append(tactile)

            datasets.append(rowTemp)
            index = index + 1

    return datasets

def train_test_dataset(path, length, flag):
    #print(length)
    label_files = []
    train_dataset = []
    test_dataset = []
    label_path = os.path.join(path, 'label_list')

    for root, dirs, files in os.walk(label_path, topdown=True):
        files.sort()
        for file in files:
            if file.endswith('.dat'):
                label_files.append(os.path.join(root, file))

    if flag == 'train':
        for i in range(10, 11):
            train_dataset = train_dataset + load_data(path, label_files, i, 7, length)
        return train_dataset
    elif flag == 'test':
        for i in range(0, 10):
            test_dataset = test_dataset + load_data(path, label_files, i, 7, length)
        return test_dataset


class MyDataset(Dataset):
    def __init__(self, image_paths, length, transform_v, transform_t, flag):
        self.image_paths = image_paths
        self.length = length
        self.transform_v = transform_v
        self.transform_t = transform_t

        self.label = []
        self.visual_sequence = []
        self.tactile_sequence = []
        self.classes = ['0', '1']
        self.flag = flag  # train
        self.dataset = train_test_dataset(self.image_paths, self.length, self.flag)

        le = LabelEncoder()
        le.fit(self.classes)

        # convert category -> 1-hot
        action_category = le.transform(self.classes).reshape(-1, 1)
        enc = OneHotEncoder()
        enc.fit(action_category)
        for item in self.dataset:
            self.label.append(str(item[1]))
            self.visual_sequence.append(item[2])
            self.tactile_sequence.append(item[3])
        self.label = labels2cat(le, self.label)

    def __getitem__(self, index):

        visuals = []
        tactiles = []
        for i in range(self.length):
            visualTemp = Image.open(self.visual_sequence[index][i])
            if self.transform_v:
                visualTemp = self.transform_v(visualTemp)  # [3, 224, 224]
                # print(visualTemp.shape)
            visuals.append(visualTemp.unsqueeze(0))

            tactileTemp = Image.open(self.tactile_sequence[index][i])
            if self.transform_t:
                tactileTemp = self.transform_t(tactileTemp)
                # print(tactileTemp.shape)  # [3, 4, 4]
            tactiles.append(tactileTemp.unsqueeze(0))

        x_v = torch.cat(visuals, dim=0)  # [13, 3, 224, 224]
        x_t = torch.cat(tactiles, dim=0)  # [13, 3, 224, 224]
        # print(x_v.shape,x_t.shape)

        y = torch.tensor(self.label[index], dtype=torch.long)
        # print(x_v.shape, x_t.shape, y)
        return x_v, x_t, y

    def __len__(self):
        return len(self.visual_sequence)


