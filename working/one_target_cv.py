#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    "-nw", "--num_workers", type=int, default=4,
    help="num_workers (default=4)")
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()
target_id = args.target_id
file_No = args.file_No
print(f"target_id:{target_id}, file_No:{file_No}")
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

# %% [markdown]
# # Metrics

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
        nom = torch.sum(torch.abs(output-target), dim=0)
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



model = TReNDSModel()


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

# %% [markdown]
# ## Early Stopping

# %%
class EarlyStopping:
    def __init__(self, patience=5, checkpoint_path='checkpoint.pth', device="cpu"):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_score = None
        self.device = device

    def load_best_weights(self, model):
        model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))

    def __call__(self, score, model, mode="min"):
        # cpuでも使用できるようにするためにパラメータを一度cpuに変換してから保存し、再度deviceに直す
        if mode == "max":
            if self.best_score is None or (score > self.best_score):
                torch.save(model.to('cpu').state_dict(), self.checkpoint_path)
                model.to(self.device)
                self.best_score, self.counter = score, 0
                return 1
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    return 2
        elif mode == "min":
            if self.best_score is None or (score < self.best_score):
                torch.save(model.to('cpu').state_dict(), self.checkpoint_path)
                model.to(self.device)
                self.best_score, self.counter = score, 0
                return 1
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    return 2
        return 0

# %% [markdown]
# # GPU Fitter

# %%
class GPUFitter:
    def __init__(self, model, fold, device, config, save_model_path="checkpoint.pth", log_path="log.csv"):
        self.model = model
        self.device = device
        self.log_path = log_path[:-4] + f"{config.target[target_id]}_fold{fold}_No{file_No}.csv"
        self.save_model_path = save_model_path[:-4] + f"{config.target[target_id]}_fold{fold}_No{file_No}.pth"
        
        self.epoch = 0
        self.fold = fold
        self.config = config
                
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=3, 
            factor=0.3, 
            verbose=True
        )
        self.early_stopping = EarlyStopping(patience=10, device=device, checkpoint_path=self.save_model_path)
        self.criterion = TReNDSLoss(self.device)
                
        self.log(f'Fitter prepared for fold {self.fold}. Device is {self.device} target:{config.target[target_id]}')
        self.columns = ["loss", "score", "val_loss", "val_score", "lr"]
        self.log_df = pd.DataFrame(columns=self.columns)
        
    def fit(self, train_loader, valid_loader):
        for e in range(self.config.epochs):
            lr = self.optimizer.param_groups[0]['lr']
            if self.config.verbose:
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR:{lr:.5f}')
            
            t = time.time()
            loss, score = self.train_one_epoch(train_loader)
            
            val_loss, val_score = self.validation_one_epoch(valid_loader)
            print(f'Epoch: {self.epoch}, loss: {loss.avg:.5f}, score: {score.avg:.5f},'                  f'val_loss: {val_loss.avg:.5f}, val_score: {val_score.avg:.5f},'                  f'time:{(time.time() - t):.5f}, lr{lr:.7f}' )
            res = self.early_stopping(val_score.avg, self.model, mode="min")
            self.scheduler.step(val_loss.avg)
            tmp = pd.DataFrame([[loss.avg, score.avg, val_loss.avg, val_score.avg, lr]], columns=self.columns)
            self.log_df = pd.concat([self.log_df, tmp], axis=0)
            self.log_df.to_csv(self.log_path, index=False)
            if res == 2:
                print("Early Stopping")
                print(self.log_path)
                print(self.save_model_path)
                break
            self.epoch += 1
    
    def train_one_epoch(self, train_loader):
        self.model.train()
        losses = AverageMeter()
        scores = AverageMeter()
        t = time.time()
        
        for step, data in enumerate(train_loader):
            scan_maps = data['scan_maps']
            fnc = data['fnc']
            loading =data['loading']
            targets = data['targets']
            
            scan_maps = scan_maps.to(self.device, dtype=torch.float)
            fnc = fnc.to(self.device, dtype=torch.float)
            loading = loading.to(self.device, dtype=torch.float)
            targets = targets.to(self.device, dtype=torch.float)
            self.optimizer.zero_grad()
            
            outputs = self.model(scan_maps, fnc, loading)
            
            loss = self.criterion(outputs, targets)
            
            batch_size = scan_maps.size(0)
            losses.update(loss.detach().item(), batch_size)
            
            targets = targets.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            
            metric = weighted_metric(targets, outputs)
            scores.update(metric, batch_size)
            loss.backward()
            self.optimizer.step()
            if config.verbose:
                if step % self.config.verbose_step == 0:
                    self.log(
                        f'Train Step {step}, ' + \
                        f'fold {self.fold}, ' + \
                        f'loss: {losses.avg:.5f}, ' + \
                        f'competition metric: {scores.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}'
                    )
        
        self.model.eval()
        
        return losses, scores
    
    def validation_one_epoch(self, validation_loader):
        self.model.eval()
        
        losses = AverageMeter()
        scores = AverageMeter()
        t = time.time()
        
        with torch.no_grad():
            for step, data in enumerate(validation_loader):
                scan_maps = data['scan_maps']
                fnc = data['fnc']
                loading =data['loading']
                targets = data['targets']

                scan_maps = scan_maps.to(self.device, dtype=torch.float)
                fnc = fnc.to(self.device, dtype=torch.float)
                loading = loading.to(self.device, dtype=torch.float)
                targets = targets.to(self.device, dtype=torch.float)
                outputs = self.model(scan_maps, fnc, loading)

                loss = self.criterion(outputs, targets)
                batch_size = scan_maps.size(0)
                losses.update(loss.detach().item(), batch_size)
                
                targets = targets.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                metric = weighted_metric(targets, outputs)
                scores.update(metric, batch_size)
                if config.verbose:
                    if step % self.config.verbose_step == 0:
                        self.log(
                        f'Validation Step {step}, ' + \
                        f'fold {self.fold}, ' + \
                        f'loss: {losses.avg:.5f}, ' + \
                        f'competition metric: {scores.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}'
                        )
                
        return losses, scores
    
    def log(self, message):
        if self.config.verbose:
            print(message)

# %% [markdown]
# # Loading data
# %% [markdown]
# ## num_workerについて
# 
# 0だとメインプロセスのみがバッチをロードして、1以上だとサブプロセスが生えて代わりにロードしてくれるらしい  
# これを1以上にすると、Pythonのコードを実行してるメインプロセスとは別のワーカープロセスがメインプロセスと並列的にデータのロードを行ってメモリにキューして行ってくれるので、メインプロセスは、データのロード以外の仕事に集中できる。  
# ただし、ワーカープロセス数は増やせばいいってもんじゃなくて、メインプロセスの他の処理の忙しさとか、CPUコア数とかバッチサイズとかにも複雑に依存するので、実測値がデフォルトよりもよくなるかはわからん。  
# ワーカープロセスが過多だと、メインプロセスが次のバッチを必要とするまでにメモリが詰まったり、その分CPUが占領され流とか、メモリが足りなくなるとか  
# 
# #### 例
# すべてのデータを使うと (num_worker=8, batch_size=16)のときはメモリエラー  
# すべてのデータを使うと (num_worker=4, batch_size=16)のときはうまくいく  
# https://deeplizard.com/learn/video/kWVgvsejXsE これがnum_workerについて分かりやすい
# 

# %%
# nrowsで読み込むtrainデータ数を決定している. すべて使う場合はnrow消す
# df = pd.read_csv('../input/trends-assessment-prediction/train_scores.csv', nrows=50)

df = pd.read_csv('../input/trends-assessment-prediction/train_scores.csv')

df['kfold'] = -1
df = df.fillna(df.mean())

kf = KFold(n_splits=config.num_folds, shuffle=True)
for fold, (trn_, val_) in enumerate(kf.split(df)):
    df.loc[val_, 'kfold'] = fold



# %%
# df = pd.read_csv('../input/trends-assessment-prediction/train_scores.csv')
# list_IDs = df["Id"].values.astype(str)
# idx = 1
# scan_id = list_IDs[idx]
# subject_filename = config.root_train_path + '/' + str(scan_id) + '.mat'
# subject_data = h5py.File(subject_filename, 'r')['SM_feature'][()]
# subject_data = np.moveaxis(subject_data, [0, 1, 2, 3], [3, 2, 1, 0])
# print(subject_filename)
# print(subject_data.shape)



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

# %% [markdown]
# # Running on multiple folds

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_model_path = "models/resnet10.pth"
log_path = "logs/log_resnet10.csv"
print(device)
print(save_model_path)
print(log_path)


# %%
def run(fold):
    
    model = TReNDSModel()
    model.to(device)
    
    train_df = df[df['kfold'] != fold].reset_index(drop=True)
    valid_df = df[df['kfold'] == fold].reset_index(drop=True)
    train_fnc = fnc[fnc['kfold'] != fold].reset_index(drop=True)
    valid_fnc = fnc[fnc['kfold'] != fold].reset_index(drop=True)
    train_loading = loading[loading['kfold'] != fold].reset_index(drop=True)
    valid_loading = loading[loading['kfold'] != fold].reset_index(drop=True)
    
    train_dataset = MRIMapDataset(df=train_df, fnc=train_fnc, loading=train_loading, mode="train")
    valid_dataset = MRIMapDataset(df=valid_df, fnc=valid_fnc, loading=valid_loading, mode="train")
    
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    fitter = GPUFitter(model, fold, device, config, save_model_path=save_model_path, log_path=log_path)
    fitter.fit(train_data_loader, valid_data_loader)
    print('over')
    return fitter


# %%
for fold in range(1, 5):
    fitter = run(fold)
    log_df = pd.read_csv(fitter.log_path)
    # %%
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.title("loss")
    log_df.loss.plot()
    log_df.val_loss.plot()
    plt.legend()
    plt.tight_layout()
    plt.subplot(1,2,2)
    plt.title("score")
    log_df.score.plot()
    log_df.val_score.plot()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'pictures/{config.target[target_id]}_fold{fold}_No{file_No}.png')


    # %%
    # train_df = df[df['kfold'] != fold].reset_index(drop=True)
    # train_dataset = MRIMapDataset(df=train_df, mode="train")
    # train_dataset[0]["scan_maps"].shape

    # %%
    # resnet10でだいたい45分 num_worker=0のとき
    test_dataset = MRIMapDataset(fnc=test_fnc, loading=test_loading, mode="test")
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
    model = TReNDSModel()
    model.load_state_dict(torch.load(fitter.save_model_path))#'../input/trend3dcnn/checkpoint0.pth'
    model.to(device)
    model.eval()

    test_preds = np.empty((0, 1))
    with torch.no_grad():
        for step, data in enumerate(tqdm(test_dataloader)):
            scan_maps = data['scan_maps']
            fnc = data['fnc']
            loading =data['loading']       
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
    test_df.to_csv(f"output/test_{config.target[target_id]}fold{fold}_No{file_No}.csv", index=False)


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

