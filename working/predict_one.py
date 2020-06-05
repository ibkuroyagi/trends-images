#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# in No.1 delet fnc and normalize each target data
# so, I changed fMRIdataset
#%%
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

import random
from tqdm.notebook import tqdm
import math
from functools import partial
import h5py
from datetime import datetime
import os
import time
import gc
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import nilearn as nl
import nilearn.plotting as nlplt
import nibabel as nib

# target_id {0:"age", 1:"domain1_var1", 2:"domain1_var2", 3:"domain2_var1", 4:"domain2_var2"}

parser = argparse.ArgumentParser()
parser.add_argument(
    "-fno", "--file_No", type=int, default=0,
    help="number of file (default=0)")
parser.add_argument(
    "-tid", "--target_id", type=int, default=0,
    help="target_id (default=0)")
parser.add_argument(
    "-fold", "--fold", type=int, default=0,
    help="fold (default=0)")
parser.add_argument(
    "-nw", "--num_workers", type=int, default=4,
    help="num_workers (default=4)")
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()
target_id = args.target_id
file_No = args.file_No
print(f"target_id:{target_id}, file_No:{file_No}, fold:{args.fold}")
print(f"verbose:{args.verbose} type:", type(args.verbose))
# %%
class config:
    epochs = 100
    batch_size = 16
    test_batch_size = 16
    learning_rate = 1e-3
    fMRI_mask_path = '../input/trends-assessment-prediction/fMRI_mask.nii'
    root_train_path = '../input/trends-assessment-prediction/fMRI_train'
    root_test_path = '../input/trends-assessment-prediction/fMRI_test'
    num_folds = 5
    seed = 2020
    verbose = args.verbose
    verbose_step = 1
    num_workers = args.num_workers
    test_num_workers = 4
    target = ["age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]
    weight = [0.3, 0.175, 0.175, 0.175, 0.175]
    # cross validationをするときはここでfoldを変更する
    fold = args.fold


# %%
print(config.target[target_id], config.weight[target_id])


# %%
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(config.seed)

# %%
def weighted_metric(y_true, y_pred):
    # weight = np.array([0.3, 0.175, 0.175, 0.175, 0.175])
    weight = np.array(config.weight[target_id])
    return np.sum(np.sum(np.abs(y_true - y_pred), axis=0) / np.sum(y_pred, axis=0) * weight)


# %%
class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# %%
class TReNDSLoss(nn.Module):
    def __init__(self, device):
        super(TReNDSLoss, self).__init__()
        # self.weights = torch.tensor([.3, .175, .175, .175, .175], dtype=torch.float32, device=device)
        self.weights = torch.tensor(config.weight[target_id], dtype=torch.float32, device=device)
    def __loss(self, output, target):
        nom = torch.sum(torch.abs(output - target), dim=0)
        denom = torch.sum(target, dim=0)
        return torch.sum(self.weights * nom / denom)

    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs):
        return self.__loss(output, target)


# %%
__all__ = [
    'resnet10', 
    'resnet18', 
    'resnet34', 
    'resnet50', 
    'resnet101',
    'resnet152', 
    'resnet200'
]

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)

def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))
    return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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

class ResNet3D(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 num_class = 5,
                 no_cuda=False):

        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet3D, self).__init__()

        # 3D conv net
        self.conv1 = nn.Conv3d(53, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        # self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 64*2, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 128*2, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 256*2, layers[3], shortcut_type, stride=1, dilation=4)

        self.fea_dim = 256*2 * block.expansion
        self.fc = nn.Sequential(nn.Linear(self.fea_dim, num_class, bias=True))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:

            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        emb_3d = x.view((-1, self.fea_dim))
        out = self.fc(emb_3d)
        return out


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet3D(BasicBlock, [1, 1, 1, 1],**kwargs)
    return model

def resnet3d_10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet3D(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet3D(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet3D(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet3D(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet3D(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet3D(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet3D(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


# %%
class TReNDSModel(nn.Module):
    def __init__(self):
        super(TReNDSModel, self).__init__()
        
        # modules = list(resnet50().children())[:-1]
        modules = list(resnet10().children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        self.m1 = nn.MaxPool3d(kernel_size=(3, 3, 3))
        self.f0 = nn.Flatten()
        self.l0 = nn.Linear(5500, 1024)
        # self.l0 = nn.Linear(17788, 1024) # resnet10 -> 4096, resnet50 -> 16384
        self.p0 = nn.PReLU()
        self.l1 = nn.Linear(1024, 256)
        self.p1 = nn.PReLU()
        self.l2 = nn.Linear(256, 1)
        
        
    def forward(self, inputs, fnc, loading):
        features = self.resnet(inputs)
        x = self.m1(features)
        flatten = self.f0(x) #shape=(batch, 16384) +(batch, 1378)) + (bathc, 26)
        x = torch.cat([flatten, fnc, loading], dim=1) #shape(batch, 16384+1378+26)
        x = self.l0(x)
        x = self.p0(x)
        x = self.l1(x)
        x = self.p1(x)
        out = self.l2(x)
        return out


# %%
def count_parameters(model):
    params = []
    for p in model.parameters():
        params.append(p.numel()) 
    return params

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# num_parameters=count_parameters(model)
# print(num_parameters)
# num_parameters=count_trainable_parameters(model)
# print(num_parameters)

# %%
class MRIMapDataset(Dataset):
    def __init__(self, df=None, fnc=None, loading=None, mode="train"):
        super(Dataset, self).__init__()
        self.mode = mode
        self.fnc = fnc.iloc[:, 1:-2].values
        self.loading = loading.iloc[:, 1:-2].values
        
        if mode == "train":
            # self.labels = df[['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']].values
            self.labels = df[config.target[target_id]].values
            self.list_IDs = df["Id"].values.astype(str)
        elif mode == "test":
            list1 = os.listdir(config.root_test_path)
            self.list_IDs = sorted(list1)

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, idx):
        if self.mode == "train":
            scan_id = self.list_IDs[idx]        
            subject_filename = config.root_train_path + '/' + scan_id + '.mat'
            subject_data = h5py.File(subject_filename, 'r')['SM_feature'][()]
            subject_data = np.moveaxis(subject_data, [0, 1, 2, 3], [3, 2, 1, 0])
            fnc = self.fnc[idx]
            loading = self.loading[idx]
            return {
                'scan_maps': torch.tensor(subject_data, dtype=torch.float),
                'fnc': torch.tensor(fnc, dtype=torch.float),
                'loading': torch.tensor(loading, dtype=torch.float),
                'targets': torch.tensor(self.labels[idx, ], dtype=torch.float)
            }
        elif self.mode == "test":
            scan_id = self.list_IDs[idx]        
            subject_filename = config.root_test_path + '/' + scan_id
            subject_data = h5py.File(subject_filename, 'r')['SM_feature'][()]
            subject_data = np.moveaxis(subject_data, [0, 1, 2, 3], [3, 2, 1, 0])
            fnc = self.fnc[idx]
            loading = self.loading[idx]
            return {
                'scan_maps': torch.tensor(subject_data, dtype=torch.float),
                'fnc': torch.tensor(fnc, dtype=torch.float),
                'loading': torch.tensor(loading, dtype=torch.float),
            }


# df = pd.read_csv('../input/trends-assessment-prediction/train_scores.csv', nrows=50)
df = pd.read_csv('../input/trends-assessment-prediction/train_scores.csv')
df_mean = df.mean()
df_std = df.std()

df['kfold'] = -1
df = df.fillna(df.mean())

kf = KFold(n_splits=config.num_folds, shuffle=True)
for fold, (trn_, val_) in enumerate(kf.split(df)):
    df.loc[val_, 'kfold'] = fold


# %%
train_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")
drop_cols = ["age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]
train_df.drop(drop_cols, axis=1, inplace=True)
train_df["is_train"] = True
train_df["kfold"] = df["kfold"].astype(int)

fnc = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")
fnc.fillna(fnc.mean(), inplace=True)
fnc = fnc.merge(train_df, on="Id", how="left")
test_fnc = fnc[fnc["is_train"] != True].copy()
fnc = fnc[fnc["is_train"] == True].copy()

loading = pd.read_csv("../input/trends-assessment-prediction/loading.csv")
loading.fillna(loading.mean(), inplace=True)
loading = loading.merge(train_df, on="Id", how="left")
test_loading = loading[loading["is_train"] != True].copy()
loading = loading[loading["is_train"] == True].copy()


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_model_path = "models/resnet10.pth"
log_path = "logs/log_resnet10.csv"
log_path = log_path[:-4] + f"{config.target[target_id]}_fold{config.fold}_No{file_No}.csv"
save_model_path = save_model_path[:-4] + f"{config.target[target_id]}_fold{config.fold}_No{file_No}.pth"
print(device)
print(save_model_path)
print(log_path)


# %%
log_df = pd.read_csv(log_path)

# %% [markdown]
# # submit test data

# %%
# resnet10でだいたい45分 num_worker=0のとき
test_dataset = MRIMapDataset(fnc=test_fnc, loading=test_loading, mode="test")
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    num_workers=config.num_workers
)
model = TReNDSModel()
model.load_state_dict(torch.load(save_model_path))  # '../input/trend3dcnn/checkpoint0.pth'
model.to(device)
model.eval()

test_preds = np.empty((0, 1))
with torch.no_grad():
    for step, data in enumerate(tqdm(test_dataloader)):
        scan_maps = data['scan_maps']
        fnc = data['fnc']
        loading = data['loading']   
        scan_maps = scan_maps.to(device, dtype=torch.float)
        fnc = fnc.to(device, dtype=torch.float)
        loading = loading.to(device, dtype=torch.float)
        outputs = model(scan_maps, fnc, loading)
        batch_size = scan_maps.size(0)
        outputs = outputs.detach().cpu().numpy()
        test_preds = np.concatenate([test_preds, outputs], 0)
        torch.cuda.empty_cache()
        gc.collect()


# %%
print(outputs.shape)
print(test_preds)


# %%
# test_df = pd.DataFrame(test_preds, columns=["age", "domain1_var1", "domain1_var2","domain2_var1", "domain2_var2"])
test_df = pd.DataFrame(test_preds, columns=[config.target[target_id]])
test_df.describe()
test_df.to_csv(f"output/test_{config.target[target_id]}fold{config.fold}_No{file_No}.csv", index=False)


# %%
# list1 = os.listdir(config.root_test_path)
# list2 = sorted(list1)
# test_df["Id"] = list2
# test_df["Id"] = test_df["Id"].map(lambda x: x[:-4])
# test_df.set_index("Id", drop=True, inplace=True)
# test_df


# # %%
# df_long = test_df.stack().reset_index()
# df_long.rename(columns={'level_1': 'target', 0: 'Predicted'}, inplace=True)
# df_long["Id"] = df_long["Id"] + "_" + df_long["target"]
# df_long.drop("target", axis=1, inplace=True)
# df_long.to_csv('submission_No{file_No}.csv', index=False)

