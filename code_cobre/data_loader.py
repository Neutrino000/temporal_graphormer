from __future__ import print_function, division
import math
import random

from src.utils.miscellaneous import mkdir
import os
import numpy as np
import torch
import scipy.io as io
import nibabel as nib
from torch.utils.data import DataLoader, Dataset
from utilities_func import z_tensor, aal_mask
import matplotlib.pyplot as plt



class Mydataset(Dataset):
    def __init__(self, args, train, txt_path, fold):
        self.args = args
        self.dataset_dir = self.args.dataset_dir +'/'
        self.temp_dir = self.dataset_dir + 'clip_dataset_revised/'
        datainfo = open(txt_path, 'r')
        imgs = []
        if self.args.sampling=='continue':
            for line in datainfo:
                line = line.strip('/n')
                imgs.append((line[:-5], line[-4], line[-2]))
        else:
            for line in datainfo:
                line = line.strip('/n')
                imgs.append((line[:-3], line[-2]))
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        sub_name = self.imgs[index][0]
        label = int(self.imgs[index][1])
        file_name = sub_name + '_' + self.imgs[index][2] +'.pt'
        dict_tempo = torch.load(self.temp_dir + file_name)
        fmri_mat = dict_tempo['fmri_mat'].to(self.args.device)
        fc_mat = dict_tempo['fc_mat'].to(self.args.device)
        mask = dict_tempo['mask'].to(self.args.device)
        roi_mat = aal_mask(self.args, fmri_mat)
        roi_mat = z_tensor(roi_mat, mask)
        return roi_mat, fc_mat, label, sub_name
