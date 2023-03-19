# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2023/3/19 10:22
# author:DYP
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 是否使用GPU
config = {
    'seed': 5201314,
    'test_size': 0.3,  # 训练、测试比例
    'epochs': 10,
    'batch_size': 32,
    'learning_rate': 1e-5,
    # 'early_stop': 400,    # If model has not improved for this many consecutive epochs, stop training.
    'file_path': 'D:\\AAA_Postgraduate_Project_Demo\\pytorch_study\\data\\WISDM\\WISDM_ar_v1.1_raw.csv',
    'file_name': 'WISDM_ar_v1.1_raw.csv',
    'window_size': 80,  # 滑窗大小
    'step_size': 40,  # 步
    'input_height': 1,
    'input_width': 80,
    'feature': 3,
    'n_class': 6
}
action = {'Jogging': 0,
          'Walking': 1,
          'Upstairs': 2,
          'Downstairs': 3,
          'Sitting': 4,
          'Standing': 5
          }


# 加载数据集
def read_data(file_path):
    data = pd.read_csv(file_path, error_bad_lines=False)
    print(data.head())
    return data


# 滑动窗口机制函数
def segment_signal(data, size=config['window_size'], step_size=config['step_size']):
    segments = []
    labels = []
    for i in range(0, len(data) - size, step_size):
        x = data['x-axis'].values[i: i + size]
        y = data['y-axis'].values[i: i + size]
        z = data['z-axis'].values[i: i + size]
        segments.append([x, y, z])
        label = stats.mode(data['Activity'][i: i + size])[0][0]
        labels.append(action[label])

    return segments, labels


# 处理数据集
def process_data(data):
    segments, labels = segment_signal(data)
    segments = np.asarray(segments, dtype=np.float32)  # (27160, 3, 80)
    # segments = np.asarray(segments, dtype=np.float32).transpose(0, 2, 1)  # shape[27160,80,3]
    labels = np.asarray(labels, dtype=np.int32)
    # reshaped_segments = segments.reshape(len(segments), config['feature'], config['window_size'])
    # print(labels)
    print('reshaped_segments.shape:', segments.shape)

    x_train, x_test, y_train, y_test = train_test_split(segments, labels, test_size=0.3, random_state=10,
                                                        shuffle=True)
    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    print("x_train size: ", x_train.shape)
    print("x_test size: ", x_test.shape)
    print("y_train size: ", y_train.shape)
    print("y_test size: ", y_test.shape)
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_dataLoader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataLoader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    return train_dataLoader, test_dataLoader


# 读取数据集
dataset = read_data(os.path.join(config['file_path']))
# 数据集处理
train_dataLoader, test_dataLoader = process_data(dataset)
print(train_dataLoader)
print(test_dataLoader)
