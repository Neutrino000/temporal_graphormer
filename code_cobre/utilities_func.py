import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import scipy.io as io
import math
from args_config import parse_args
from datetime import datetime
import argparse

def normalized_tensor(vector):
    # 计算向量的L2范数
    norm = torch.norm(vector, p=2)

    if norm == 0:
        # 避免零除错误，如果范数为零，返回原始向量
        return vector
    else:
        # 归一化向量
        normalized_vector = vector / norm
        return normalized_vector

def temporary_func_con(args, fmri_mat, sub_name, time_course_dir, num_seg):
    input_size = 8 * (math.floor(np.array(fmri_mat.shape[1:]).min()/8)+1)
    # fmri_mat_seg = fmri_mat[-((num_seg+1)*args.time_length_con+1):-(num_seg*args.time_length_con+1)].to(args.device)
    if num_seg==0:
        fmri_mat_seg = fmri_mat[-args.time_length_con:].to(args.device)
    else:
        fmri_mat_seg = fmri_mat[-((num_seg + 1) * args.time_length_con ):-(num_seg * args.time_length_con )].to(
            args.device)
    # if num_seg==args.num_segment_con-1:
    #     fmri_mat_seg = fmri_mat[-args.time_length_con:].to(args.device)
    # else:
    #     fmri_mat_seg = fmri_mat[-(args.num_segment_con-num_seg)*args.time_length_con:-(args.num_segment_con-num_seg-1)*args.time_length_con].to(args.device)
    if args.time_course=='dpabi' and args.datasets!='abide':
        sub_dir = time_course_dir + args.ifgroup + '_' + args.time_course_cate + '/' + sub_name + '.mat'
        if args.time_course_cate=='seed':
            roi_mat = io.loadmat(sub_dir)['SeedSeries'].transpose()
        else:
            roi_mat = io.loadmat(sub_dir)['ROISignals'].transpose()
    elif args.time_course=='dpabi' and args.datasets=='abide':
        sub_dir = time_course_dir + sub_name + '.1D'
        with open(sub_dir, 'r') as file:
            lines = file.readlines()
            line = lines[0].split('\t')
            num_time = len(lines) - 1
            num_roi = len(line)
            roi_mat = np.zeros([num_time, num_roi])
            for i in range(num_time):
                line = lines[i+1].split('\t')
                for j in range(num_roi):
                    roi_mat[i, j] = float(line[j])
            roi_mat = roi_mat[-args.time_points:, :].transpose()
    elif args.time_course=='fmri':
        aal_temp = np.array(nib.load(args.temp_dir).get_fdata())
        roi_mat = np.zeros([args.num_roi, fmri_mat_seg.shape[0]])
        for i in range(args.num_roi):
            mask = np.equal(aal_temp, float(i))
            masked_mat = np.multiply(np.array((fmri_mat_seg.cpu())), mask)
            num_voxel = np.sum(masked_mat!=0)/fmri_mat_seg.shape[0]
            mean_roi = np.sum(masked_mat, axis=(1, 2, 3))/num_voxel
            roi_mat[i] = mean_roi.reshape(fmri_mat_seg.shape[0])
    if args.time_course=='dpabi':
        if num_seg==args.num_segment_con-1:
            roi_mat_seg = roi_mat[:, -args.time_length_con:]
        else:
            roi_mat_seg = roi_mat[:, -(args.num_segment_con - num_seg) * args.time_length_con:-(
                        args.num_segment_con - num_seg - 1) * args.time_length_con]
    else:
        roi_mat_seg = roi_mat
    corr_mat = np.corrcoef(roi_mat_seg)
    corr_flat = corr_mat.flatten()
    corr_flat.sort()
    mask = np.where(corr_mat>corr_flat[int(args.ratio_bi*corr_flat.shape[0])], 1, 0)
    if args.binary:
        final_corr = mask
    else:
        final_corr = np.multiply(corr_mat, mask)
    if np.isnan(corr_mat).any():
        print(sub_name)
    fmri_mat_seg1 = torch.empty(torch.Size([args.time_length_con, input_size, input_size, input_size]))
    fmri_mat_seg1[:, 1:np.array(fmri_mat.shape).min()+1, :, 1:np.array(fmri_mat.shape).min()+1] = fmri_mat_seg[:, :, 5:input_size+5, :]
    return fmri_mat_seg1, final_corr


def temporary_func(args, fmri_mat, sub_name, time_course_dir):
    input_size = 8 * (math.floor(np.array(fmri_mat.shape[1:]).min()/8)+1)
    fmri_mat_seg = torch.empty(torch.Size([args.num_segment, input_size, input_size, input_size]))
    # fmri_mat_seg = torch.empty(torch.Size([args.num_segment]) + fmri_mat.shape[-3:])
    for i in range(args.num_segment):
        if args.segment_proc=='mean':
            fmri_mat_seg11 = fmri_mat[-(args.time_length+args.interval*(args.num_segment-i-1)):-(args.interval*(args.num_segment-i-1)), :, :, :]
            fmri_mat_seg[i, 1:np.array(fmri_mat.shape).min()+1, :, 1:np.array(fmri_mat.shape).min()+1] = torch.mean(fmri_mat_seg11, dim=0)[:, :, :, 5:input_size+5, :]
        else:
            fmri_mat_seg[i, 1:np.array(fmri_mat.shape).min()+1, :, 1:np.array(fmri_mat.shape).min()+1] = fmri_mat[-(int(args.time_length/2)+args.interval*(args.num_segment-i-1)), :, :, :][:, :, :, 5:input_size+5, :]
    time_corr = []
    if args.time_course=='dpabi' and args.datasets!='abide':
        sub_dir = time_course_dir + sub_name + '.mat'
        if args.time_course_cate=='seed':
            roi_mat = io.loadmat(sub_dir)['SeedSeries'].transpose()
        else:
            roi_mat = io.loadmat(sub_dir)['ROISignals'].transpose()
        for i in range(args.num_segment):
            if i==args.num_segment-1:
                corr_matrix = np.corrcoef(roi_mat[:, -(args.time_length+args.interval*(args.num_segment-i-1)):])
            else:
                corr_matrix = np.corrcoef(roi_mat[:, -(args.time_length+args.interval*(args.num_segment-i-1)):-(args.interval*(args.num_segment-i-1))])
            time_corr.append(corr_matrix)
    elif args.time_course=='dpabi' and args.datasets=='abide':
        sub_dir = time_course_dir + sub_name + '.1D'
        with open(sub_dir, 'r') as file:
            lines = file.readlines()
            line = lines[0].split('\t')
            num_time = len(lines) - 1
            num_roi = len(line)
            roi_mat = np.zeros([num_time, num_roi])
            for i in range(num_time):
                line = lines[i + 1].split('\t')
                for j in range(num_roi):
                    roi_mat[i, j] = float(line[j])
            roi_mat = roi_mat[-args.time_points:, :].transpose()
        file.close()
        for i in range(args.num_segment):
            if i==args.num_segment - 1:
                corr_matrix = np.corrcoef(roi_mat[:, -(args.time_length+args.interval*(args.num_segment-i-1)):])
            else:
                corr_matrix = np.corrcoef(roi_mat[:, -(args.time_length+args.interval*(args.num_segment-i-1)):-(args.interval*(args.num_segment-i-1))])
            time_corr.append(corr_matrix)
    elif args.time_course=='fmri':
        aal_temp = np.array(nib.load(args.temp_dir).get_fdata())
        roi_mat = np.zeros([args.num_roi, fmri_mat.shape[0]])
        for i in range(args.num_roi):
            mask = np.equal(aal_temp, float(i))
            masked_mat = np.multiply(np.array((fmri_mat.cpu())), mask)
            num_voxel = np.sum(masked_mat!=0)/fmri_mat.shape[0]
            mean_roi = np.sum(masked_mat, axis=(1, 2, 3))/num_voxel
            roi_mat[i] = mean_roi.reshape(fmri_mat.shape[0])
        for i in range(args.num_segment):
            corr_matrix = np.corrcoef(roi_mat[:, -(args.time_length + args.interval * (args.num_segment - i - 1)):-(
                        args.interval * (args.num_segment - i - 1))])
            time_corr.append(corr_matrix)
    time_corr = np.stack(time_corr, axis=0)
    time_corr = time_corr.astype(float)
    final_corr = np.ones_like(time_corr)
    for i in range(time_corr.shape[0]):
        corr_flat = time_corr[i].flatten()
        corr_flat.sort()
        corr = np.where(time_corr[i]>corr_flat[int(args.ratio_bi*corr_flat.shape[0])], 1, 0)
        if args.binary:
            final_corr[i] = corr
        else:
            final_corr[i] = np.multiply(time_corr[i], corr)
        if np.isnan(time_corr).any():
            print(sub_name)
    return fmri_mat_seg, final_corr


def z_tensor(vector, mask):
    for i in range(vector.shape[0]):
        for j in range(vector.shape[1]):
            mat = vector[i, j]
            # mask = (mat>0)
            mask = mask.bool()
            mean = mat[mask].mean()
            std = mat[mask].std()
            z_mat = torch.zeros_like(mat)
            z_mat[mask] = (mat[mask]-mean)/std
            vector[i, j] = z_mat
    return vector


def aal_mask(args, fmri_mat):
    aal_temp1 = torch.from_numpy(nib.load(args.temp_dir).get_fdata()).to(args.device)
    aal_temp = torch.empty(torch.Size(fmri_mat.shape[-3:])).to(args.device)
    aal_temp[1:np.array(aal_temp1.shape).min() + 1, :, 1:np.array(aal_temp1.shape).min() + 1] = aal_temp1[:, 5:fmri_mat.shape[-1] + 5, :]
    masked_mat = torch.empty(torch.Size([args.time_length_con, args.num_roi])+fmri_mat.shape[-3:]).to(args.device) #[30, 116, 64, 64, 64]
    for i in range(args.num_roi):
        mask = args.ratio_roi * torch.eq(aal_temp, float(i))+torch.ones_like(fmri_mat).to(args.device)
        masked_mat[:, i] = torch.mul(fmri_mat, mask)
    return masked_mat


def merged_args(args_list):
    merged_args_dict = {}
    for args in args_list:
        args_dict = vars(args)
        merged_args_dict.update(args_dict)
    merged_args = argparse.Namespace(**merged_args_dict)
    return merged_args

def args_to_text(args_obj):
    args_dict = vars(args_obj)
    text = ""
    for key, value in args_dict.items():
        text += f"{key}: {value}\n"
    return text

def save_args_to_txt(args_obj, filename):
    text = args_to_text(args_obj)
    with open(filename, 'w') as file:
        file.write(text)


class Ortho_cluster(nn.Module):
    def __init__(self, main_args, v_dimension):
        super(Ortho_cluster, self).__init__()
        self.args = main_args
        self.k_oc = self.args.k_oc
        self.e_para = torch.randn(self.k_oc, v_dimension)
        self.softmax = nn.Softmax(dim=1)
        if self.args.init_oc!='learnable':
            self.e_para.requires_grad = False
        self.initialize_weights()
        self.e_para = nn.Parameter(self.e_para)

    def initialize_weights(self):
        torch.nn.init.xavier_normal_(self.e_para)
        if self.args.init_oc=='ortho':
            basis = []
            for i in range(self.e_para.shape[0]):
                basis_vector = self.e_para[i] - sum(
                    torch.dot(self.e_para[i], existing_basis) * existing_basis for existing_basis in basis)
                if torch.norm(basis_vector) > 1e-10:  # 避免添加过小的向量
                    basis.append(basis_vector / torch.norm(basis_vector))
            self.e_para = torch.stack(basis)

    def forward(self, x):
        zg = []
        for i in range(x.shape[0]):
            p_fea = torch.matmul(x[i], self.e_para.t())
            p_softmax = self.softmax(p_fea)
            zg.append(torch.matmul(p_softmax.t(), x[i]))
        return torch.stack(zg)
