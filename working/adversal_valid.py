import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
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
from sklearn.model_selection import KFold, StratifiedKFold


parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--file_No", type=int, default=0,
    help="number of file (default=0)")
parser.add_argument(
    "-fold", "--fold", type=int, default=0,
    help="fold (default=0)")
# parser.add_argument(
#     "-nw", "--num_workers", type=int, default=4,
#     help="num_workers (default=4)")
# parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()
file_No = args.file_No
fold = args.fold
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
    verbose = True
    verbose_step = 1
    num_workers = 4
    test_num_workers = 4
    target = ["age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]
    weight = [0.3, 0.175, 0.175, 0.175, 0.175]
    # cross validationをするときはここでfoldを変更する
    fold = fold


print("fold", config.fold, "file_No", file_No)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(config.seed)

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

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weight = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
    return cb_loss
no_of_classes = 2
logits = F.softmax(torch.rand(10,no_of_classes)).float()#.view(-1)
labels = torch.empty(10).random_(2).to(torch.int64)
print(logits)
print(labels)
beta = 0.9999
gamma = 1.0
samples_per_cls = [400, 4000]
loss_type = "focal"
cb_loss = CB_loss(labels, logits, samples_per_cls, no_of_classes,loss_type, beta, gamma)
print(cb_loss)


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
        self.l2 = nn.Linear(256, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, fnc, loading):
        features = self.resnet(inputs)
        x = self.m1(features)
        flatten = self.f0(x)  # shape=(batch, 16384) +(batch, 1378)) + (bathc, 26)
        x = torch.cat([flatten, fnc, loading], dim=1)  # shape(batch, 16384+1378+26)
        x = self.l0(x)
        x = self.p0(x)
        x = self.l1(x)
        x = self.p1(x)
        out = self.l2(x)
        out = self.sigmoid(out)
        return out


class MRIMapDataset(Dataset):
    def __init__(self, df=None, fnc=None, loading=None, mode="train"):
        super(Dataset, self).__init__()
        self.mode = mode
        self.fnc = fnc.iloc[:, 1:-2].values
        self.loading = loading.iloc[:, 1:-2].values
        
        if mode == "train":
            # self.labels = df[['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']].values
            self.labels = df["is_site2"].values
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
            fnc = self.fnc[idx] / 600.0
            loading = self.loading[idx]
            return {
                'scan_maps': torch.tensor(subject_data, dtype=torch.float),
                'fnc': torch.tensor(fnc, dtype=torch.float),
                'loading': torch.tensor(loading, dtype=torch.float),
                'targets': torch.tensor(self.labels[idx, ], dtype=torch.int)
            }
        elif self.mode == "test":
            scan_id = self.list_IDs[idx]        
            subject_filename = config.root_test_path + '/' + scan_id
            subject_data = h5py.File(subject_filename, 'r')['SM_feature'][()]
            subject_data = np.moveaxis(subject_data, [0, 1, 2, 3], [3, 2, 1, 0])
            fnc = self.fnc[idx].values / 600.0
            loading = self.loading[idx].values
            return {
                'scan_maps': torch.tensor(subject_data, dtype=torch.float),
                'fnc': torch.tensor(fnc, dtype=torch.float),
                'loading': torch.tensor(loading, dtype=torch.float),
            }


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


class GPUFitter:
    def __init__(self, model, fold, device, config, save_model_path="checkpoint.pth", log_path="log.csv"):
        self.model = model
        self.device = device
        self.log_path = log_path[:-4] +f"_fold{fold}_No{file_No}.csv"
        self.save_model_path = save_model_path[:-4] +f"_fold{fold}_No{file_No}.pth"
        
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

        self.no_of_classes = 2
        self.beta = 0.9999
        self.gamma = 2.0
        self.samples_per_cls = [1176 * 4, 102 * 4]
        self.samples_per_cls_valid = [1176 * 4, 102 * 1]
        self.loss_type = "focal"
        self.criterion = CB_loss  # TReNDSLoss(self.device)                
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
            print(f'Epoch: {self.epoch}, loss: {loss.avg:.5f}, score: {score:.5f},'                  f'val_loss: {val_loss.avg:.5f}, val_score: {val_score:.5f},'                  f'time:{(time.time() - t):.5f}, lr{lr:.7f}' )
            res = self.early_stopping(val_score, self.model, mode="max")
            self.scheduler.step(val_loss.avg)
            tmp = pd.DataFrame([[loss.avg, score, val_loss.avg, val_score, lr]], columns=self.columns)
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
        t = time.time()
        _targets = []
        _outputs = []
        for step, data in enumerate(train_loader):
            scan_maps = data['scan_maps']
            fnc = data['fnc']
            loading = data['loading']
            targets = data['targets']
            
            scan_maps = scan_maps.to(self.device, dtype=torch.float)
            fnc = fnc.to(self.device, dtype=torch.float)
            loading = loading.to(self.device, dtype=torch.float)
            targets = targets.to(self.device, dtype=torch.int)
            self.optimizer.zero_grad()
            
            outputs = self.model(scan_maps, fnc, loading)
            loss = self.criterion(targets, outputs, self.samples_per_cls, self.no_of_classes, self.loss_type, self.beta, self.gamma)
            
            batch_size = scan_maps.size(0)
            losses.update(loss.detach().item(), batch_size)
            
            targets = targets.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            _targets += list(targets)
            _outputs += list(outputs)
            loss.backward()
            self.optimizer.step()
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    self.log(
                        f'Train Step {step}, ' + \
                        f'fold {self.fold}, ' + \
                        f'loss: {losses.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}'
                    )
        scores = f1_score(_targets, _outputs) 
        return losses, scores

    def validation_one_epoch(self, validation_loader):
        self.model.eval()
        
        losses = AverageMeter()
        t = time.time()
        _targets = []
        _outputs = []
        with torch.no_grad():
            for step, data in enumerate(validation_loader):
                scan_maps = data['scan_maps']
                fnc = data['fnc']
                loading = data['loading']
                targets = data['targets']

                scan_maps = scan_maps.to(self.device, dtype=torch.float)
                fnc = fnc.to(self.device, dtype=torch.float)
                loading = loading.to(self.device, dtype=torch.float)
                targets = targets.to(self.device, dtype=torch.float)
                outputs = self.model(scan_maps, fnc, loading)

                loss = self.criterion(targets, outputs, self.samples_per_cls_valid, self.no_of_classes, self.loss_type, self.beta, self.gamma)
                batch_size = scan_maps.size(0)
                losses.update(loss.detach().item(), batch_size)
                
                targets = targets.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                _targets += list(targets)
                _outputs += list(outputs)
                if self.config.verbose:
                    if step % self.config.verbose_step == 0:
                        self.log(
                        f'Validation Step {step}, ' + \
                        f'fold {self.fold}, ' + \
                        f'loss: {losses.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}'
                        )
        scores = f1_score(_targets, _outputs)
        return losses, scores

    def log(self, message):
        if self.config.verbose:
            print(message)

train_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")
drop_cols = ["age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]
train_df.drop(drop_cols, axis=1, inplace=True)
train_df["is_train"] = True

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

# devide test data site2 and unknow
site2_id = pd.read_csv("../input/trends-assessment-prediction/reveal_ID_site2.csv")
site2_id['is_site2'] = 1
loading["is_site2"] = 0
fnc["is_site2"] = 0
test_loading = pd.merge(test_loading, site2_id, how="left")
test_loading.loc[test_loading["is_site2"] != 1, ["is_site2"]] = "unknow"
test_fnc = pd.merge(test_fnc, site2_id, how="left")
test_fnc.loc[test_fnc["is_site2"] != 1, ["is_site2"]] = "unknow"

adversal_loading_df = pd.concat([loading, test_loading[test_loading["is_site2"] == 1]], axis=0).reset_index(drop=True).drop("is_train", axis=1)
adversal_fnc_df = pd.concat([fnc, test_fnc[test_fnc["is_site2"] == 1]], axis=0).reset_index(drop=True).drop("is_train", axis=1)


adversal_loading_df['kfold'] = -1
adversal_fnc_df['kfold'] = -1

kf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)
for fold, (trn_, val_) in enumerate(kf.split(adversal_loading_df[adversal_loading_df.columns[1:-1]], adversal_loading_df["is_site2"].astype(int))):
    adversal_loading_df.loc[val_, 'kfold'] = fold
    adversal_fnc_df.loc[val_, 'kfold'] = fold
adversal_target_df = adversal_loading_df[["Id", "is_site2", "kfold"]]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_model_path = "adversal_resnet10.pth"
log_path = "log_adversal_resnet10.csv"
print(device)
print(save_model_path)
print(log_path)


def run(fold):  
    model = TReNDSModel()
    model.to(device)
    adversal_train_loading_df = adversal_loading_df[adversal_loading_df['kfold'] != fold].reset_index(drop=True)
    adversal_valid_loading_df = adversal_loading_df[adversal_loading_df['kfold'] == fold].reset_index(drop=True)
    adversal_train_fnc_df = adversal_fnc_df[adversal_fnc_df['kfold'] != fold].reset_index(drop=True)
    adversal_valid_fnc_df = adversal_fnc_df[adversal_fnc_df['kfold'] == fold].reset_index(drop=True)
    adversal_target_train_df = adversal_target_df[adversal_target_df['kfold'] != fold].reset_index(drop=True)
    adversal_target_valid_df = adversal_target_df[adversal_target_df['kfold'] != fold].reset_index(drop=True)
    
    train_dataset = MRIMapDataset(df=adversal_target_train_df, fnc=adversal_train_fnc_df, loading=adversal_train_loading_df, mode="train")
    valid_dataset = MRIMapDataset(df=adversal_target_valid_df, fnc=adversal_valid_fnc_df, loading=adversal_valid_loading_df, mode="train")
    
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True
    )
    
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False
    )
    
    fitter = GPUFitter(model, fold, device, config, save_model_path=save_model_path, log_path=log_path)
    fitter.fit(train_data_loader, valid_data_loader)
    print('over')
    return fitter

# 
fitter = run(config.fold)


#
log_df = pd.read_csv(fitter.log_path)
ptcture_path = log_path[:-4] +f"_fold{config.fold}_No{file_No}.png"
plt.figure(figsize=(15,5))
plt.title("loss")
plt.subplot(1,2,1)
log_df.loss.plot()
log_df.val_loss.plot()
plt.subplot(1,2,2)
plt.title("score")
log_df.score.plot()
log_df.val_score.plot()
plt.savefig(f"adversal/{ptcture_path}")


adversal_valid_fnc_df = adversal_fnc_df[adversal_fnc_df['kfold'] == config.fold].reset_index(drop=True)
adversal_valid_loading_df = adversal_target_df[adversal_loading_df['kfold'] == config.fold].reset_index(drop=True)

valid_dataset = MRIMapDataset(fnc=adversal_valid_fnc_df, loading=adversal_valid_loading_df, mode="test")
valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    shuffle=False)
_test_loading = test_loading.loc[test_loading["is_site2"] != 1]
_test_fnc = test_fnc.loc[test_fnc["is_site2"] != 1]
test_dataset = MRIMapDataset(fnc=_test_fnc, loading=_test_loading, mode="test")
test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False)
model = TReNDSModel()
model.load_state_dict(torch.load(fitter.save_model_path))#'../input/trend3dcnn/checkpoint0.pth'
model.to(device)
model.eval()

test_preds = np.empty((0, 2))
valid_preds = np.empty((0, 2))
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
    test_df = pd.DataFrame(test_preds, columns=["site1", "site2"])
    test_df.to_csv(f"adversal/test_fold{config.fold}_No{file_No}.csv", index=False)
    for step, data in enumerate(tqdm(valid_data_loader)):
        scan_maps = data['scan_maps']
        fnc = data['fnc']
        loading =data['loading']       
        scan_maps = scan_maps.to(device, dtype=torch.float)
        fnc = fnc.to(device, dtype=torch.float)
        loading = loading.to(device, dtype=torch.float)
        
        outputs = model(scan_maps, fnc, loading)
        batch_size = scan_maps.size(0)
        outputs = outputs.detach().cpu().numpy()
        valid_preds = np.concatenate([valid_preds, outputs], 0)
        torch.cuda.empty_cache()
        gc.collect()
    valid_df = pd.DataFrame(valid_preds, columns=["site1", "site2"])
    valid_df.to_csv(f"adversal/valid_fold{config.fold}_No{file_No}.csv", index=False)

print(outputs.shape)
print(test_preds)
print(valid_preds.shape)