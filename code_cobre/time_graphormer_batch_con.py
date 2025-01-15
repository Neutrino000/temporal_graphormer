import torch
from generic_UNet import Generic_UNet
import torch.nn as nn
from initialization import InitWeights_He
from load_pretrained_weights import load_pretrained_weights
import numpy as np
from feature_extraction import FE_UNet
import os
import argparse
from bert.modeling_bert import BertConfig
from bert.modeling_graphormer import Graphormer
from args_config import parse_args, hippo_args, tumor_args, tg_args, fg_args
from utilities_func import Ortho_cluster


def symmetric_normalize_adjacency(adjacency_matrix):
    # 对称归一化
    degree = torch.sum(adjacency_matrix, dim=1)
    degree_sqrt_inv = torch.sqrt(1.0 / (degree + 1e-9))  # 避免除以零
    D_sqrt_inv = torch.diag(degree_sqrt_inv)
    normalized_adjacency = torch.matmul(torch.matmul(D_sqrt_inv, adjacency_matrix), D_sqrt_inv)
    return normalized_adjacency


def normalize_adjacency(adj):
    if adj.ndimension() == 2:
        adj = symmetric_normalize_adjacency(adj)
    else:
        for i in range(adj.shape[0]):
            adj[i] = symmetric_normalize_adjacency(adj[i])
    return adj

class unet_timegraphormer(nn.Module):
    def __init__(self):
        super(unet_timegraphormer, self).__init__()
        self.main_args = parse_args()
        if self.main_args.unet_cate=='tumor':
            self.unet_args = tumor_args()
        else:
            self.unet_args = hippo_args()
        self.conv_op = nn.Conv3d
        self.norm_op = nn.InstanceNorm3d
        self.norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        self.dropout_op = nn.Dropout3d
        self.dropout_op_kwargs = {'p': 0, 'inplace': True}
        self.net_nonlin = nn.LeakyReLU
        self.net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.net_numpool = len(self.unet_args.pool_kernel_size)
        self.net_numpool_ori = len(self.unet_args.pool_kernel_size_ori)
        self.unet = Generic_UNet(self.unet_args.num_inchannels_ori, self.unet_args.base_num_features, self.unet_args.num_classes, self.net_numpool_ori,
                                 self.unet_args.conv_per_stage, 2, self.conv_op, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                 self.dropout_op_kwargs, self.net_nonlin, self.net_nonlin_kwargs, False, False, lambda x: x,
                                 InitWeights_He(1e-2), self.unet_args.pool_kernel_size_ori, self.unet_args.conv_kernel_size_ori, False, True, True)
        load_pretrained_weights(self.unet, self.unet_args.path_model)
        self.fe_unet = FE_UNet(self.unet_args.num_inchannels, self.unet_args.base_num_features, self.unet_args.num_classes, self.net_numpool,
                               self.unet_args.conv_per_stage, 2, self.conv_op, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                               self.dropout_op_kwargs, self.net_nonlin, self.net_nonlin_kwargs, False, False, lambda x: x,
                               InitWeights_He(1e-2), self.unet_args.pool_kernel_size, self.unet_args.conv_kernel_size, False, True, True)
        for name, param in self.unet.named_parameters():
            for namefe, paramfe in self.fe_unet.named_parameters():
                if name == namefe and name != 'conv_blocks_context.0.blocks.0.conv.weight':
                    paramfe.data.copy_(param.data)
                if name == namefe and name == 'conv_blocks_context.0.blocks.0.conv.weight':
                    for i in range(int(paramfe.data.shape[1] / param.data.shape[1])):
                        paramfe.data[:, i * param.data.shape[1]:(i + 1) * param.data.shape[1]].copy_(param.data)
        self.pooling = nn.AvgPool3d(kernel_size=2, stride=2)
        del self.unet

    def forward(self, x):
        x = self.fe_unet(x)
        x = self.pooling(x)
        return x


class Transencoder(nn.Module):
    def __init__(self, main_args, ifencoding):
        super(Transencoder, self).__init__()
        self.args = tg_args()
        self.input_feat_dim = [int(item) for item in self.args.input_feat_dim.split(',')]
        self.hidden_feat_dim = [int(item) for item in self.args.hidden_feat_dim.split(',')]
        self.output_feat_dim = self.input_feat_dim[1:]+[self.args.inter_out_dim]
        which_blk_graph = [int(item) for item in self.args.which_gcn.split(',')]
        self.config_list = []
        for i in range(len(self.output_feat_dim)):
            config_class, model_class = BertConfig, Graphormer
            self.config = config_class.from_pretrained(self.args.config_name)
            self.config.output_attentions = False
            self.config.hidden_dropout_prob = main_args.drop_out
            self.config.img_feature_dim = self.input_feat_dim[i]
            self.config.output_feature_dim = self.output_feat_dim[i]
            self.config.ifencoding = ifencoding
            self.config.ifsagp = tg_args().ifsagp
            self.config.sagp_pool_ratio = tg_args().sagp_pool_ratio
            self.args.hidden_size = self.hidden_feat_dim[i]
            self.args.intermediate_size = int(self.args.hidden_size*self.args.interm_size_scale)

            if which_blk_graph[i]==1:
                self.config.graph_conv = True
                # logger.info('Add Graph Conv')
            else:
                self.config.graph_conv = False

            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

            for idx, param in enumerate(update_params):
                arg_param = getattr(self.args, param)
                config_param = getattr(self.config, param)
                if arg_param > 0 and arg_param != config_param:
                    # logger.info('Update config parameter {}: {} -> {}'.format(param, config_param, arg_param))
                    setattr(self.config, param, arg_param)

            assert self.config.hidden_size % self.config.num_attention_heads ==0
            self.config_list.append(self.config)
            # model = model_class(config=self.config)
            # logger.info('Init model from scratch')
            # self.trans_encoder.append(model)
        # self.trans_encoder = torch.nn.Sequential(*self.trans_encoder)
        self.trans_encoder0 = Graphormer(self.config_list[0])
        # self.trans_encoder1 = Graphormer(self.config_list[1])
        self.trans_encoder1 = Graphormer(self.config_list[1])
        # self.total_params = sum(p.numel() for p in self.trans_encoder.parameters())
        # logger.info('Graphormer encoders total parameters: {}'.format(self.total_params))

    def forward(self, x, adj):
        x, adj = self.trans_encoder0(x, adj)
        adj = normalize_adjacency(adj)
        # x, adj = self.trans_encoder1(x, adj)
        x, adj = self.trans_encoder1(x, adj)
        return x

class Final_transencoder(nn.Module):
    def __init__(self, main_args, ifencoding, in_shape):
        super(Final_transencoder, self).__init__()
        self.args = fg_args()
        self.input_feat_dim = [int(item) for item in self.args.input_feat_dim.split(',')]
        self.input_feat_dim[0] = in_shape
        self.hidden_feat_dim = [int(item) for item in self.args.hidden_feat_dim.split(',')]
        self.output_feat_dim = self.input_feat_dim[1:] + [self.args.inter_out_dim]

        which_blk_graph = [int(item) for item in self.args.which_gcn.split(',')]
        self.config_list = []
        for i in range(len(self.output_feat_dim)):
            config_class, model_class = BertConfig, Graphormer
            self.config = config_class.from_pretrained(self.args.config_name)

            self.config.output_attentions = False
            self.config.hidden_dropout_prob = main_args.drop_out
            self.config.img_feature_dim = self.input_feat_dim[i]
            self.config.output_feature_dim = self.output_feat_dim[i]
            self.config.ifencoding = ifencoding
            self.config.ifsagp = fg_args().ifsagp
            self.config.sagp_pool_ratio = fg_args().sagp_pool_ratio
            self.args.hidden_size = self.hidden_feat_dim[i]
            self.args.intermediate_size = int(self.args.hidden_size*self.args.interm_size_scale)

            if which_blk_graph[i]==1:
                self.config.graph_conv = True
                # logger.info('Add Graph Conv')
            else:
                self.config.graph_conv = False

            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

            for idx, param in enumerate(update_params):
                arg_param = getattr(self.args, param)
                config_param = getattr(self.config, param)
                if arg_param > 0 and arg_param != config_param:
                    # logger.info('Update config parameter {}: {} -> {}'.format(param, config_param, arg_param))
                    setattr(self.config, param, arg_param)

            assert self.config.hidden_size % self.config.num_attention_heads ==0
            self.config_list.append(self.config)
        self.trans_encoder0 = Graphormer(self.config_list[0])
        # self.trans_encoder1 = Graphormer(self.config_list[1])
        self.trans_encoder1 = Graphormer(self.config_list[1])

    def forward(self, x, adj):
        x = self.trans_encoder0(x, adj)
        # x = self.trans_encoder1(x, adj)
        x = self.trans_encoder1(x, adj)
        return x

class TimeGraphormer_con(nn.Module):
    def __init__(self, main_args):
        super(TimeGraphormer_con, self).__init__()
        self.tg_args = tg_args()
        self.fg_args = fg_args()
        self.args = main_args
        if self.args.datasets=='cobre' or self.args.datasets=='abide':
            self.cate = 2
        else:
            if self.args.which_class=='all':
                self.cate = 6
            else:
                self.cate = 2
        self.unet = unet_timegraphormer()
        if not self.args.unet_train:
            for param in self.unet.parameters():
                param.requires_grad = False
        self.encoder_list = nn.ModuleList()
        for i in range(self.args.num_encoder):
            self.encoder_list.append(Transencoder(self.args, False).to(self.args.device))
        self.oc = Ortho_cluster(self.args, self.tg_args.inter_out_dim)
        if self.tg_args.ifsagp:
            if self.args.if_readout:
                self.final_graphormer = Final_transencoder(self.args, True, self.args.k_oc * self.tg_args.inter_out_dim).to(self.args.device)
            else:
                self.new_shape = self.args.num_roi
                for ii in range(self.tg_args.num_graph):
                    self.new_shape = int(self.new_shape * self.tg_args.sagp_pool_ratio)
                # self.new_shape = int(self.tg_args.sagp_pool_ratio*int(self.tg_args.sagp_pool_ratio*self.args.num_roi))
                self.final_graphormer = Final_transencoder(self.args, True, self.new_shape * self.tg_args.inter_out_dim).to(self.args.device)
        else:
            if self.args.if_readout:
                self.final_graphormer = Final_transencoder(self.args, True, self.tg_args.inter_out_dim * self.tg_args.inter_out_dim).to(self.args.device)
            else:
                self.final_graphormer = Final_transencoder(self.args, True, self.args.num_roi * self.tg_args.inter_out_dim).to(self.args.device)
        self.linear1 = nn.Linear(self.args.time_length_con*self.fg_args.inter_out_dim, self.fg_args.linear_dim[0]).to(self.args.device)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(self.fg_args.linear_dim[0], self.fg_args.linear_dim[1]).to(self.args.device)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(self.fg_args.linear_dim[1], self.fg_args.linear_dim[2]).to(self.args.device)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(self.fg_args.linear_dim[2], self.cate).to(self.args.device)


    def forward(self, x, adj):
        batch_size = x.shape[0]
        adj = normalize_adjacency(adj)
        #if adj.ndimension() == 2:
        #    adj = symmetric_normalize_adjacency(adj)
        #else:
        #    for i in range(batch_size):
        #        adj[i] = symmetric_normalize_adjacency(adj[i])
        features = []
        # for i in range(batch_size):
        #     roi_feature = []
        #     for nn in range(6):
        #         roi_feature.append(self.unet(x[i, nn*5:(nn+1)*5]).view(5, 116, -1))
        #     roi_feature = torch.stack(roi_feature, dim=0)
        #     features.append(roi_feature.view(-1, *roi_feature.shape[2:]))
        for i in range(batch_size):
            # frame_feature = []
            # for frame in range(30):
            #     frame_feature.append(self.unet(x[i, frame]).view(116, -1))
            # frame_feature = torch.stack(frame_feature, dim=0)
            # features.append(frame_feature)
            #     frame_feature.append(torch.stack(roi_feature, dim=0))
            # frame_feature = torch.stack(frame_feature, dim=0)
            # print(frame_feature.shape)
                # aaa = self.unet(x[i, :, roi].unsqueeze(1)).view(self.args.time_length_con, -1)
                # print(aaa.shape)
            features.append(self.unet(x[i]).view(self.args.time_length_con, self.args.num_roi, -1))
        # features = features.view(features.shape[0], features.shape[1], -1)
        features = torch.stack(features, dim=0)
        features_list = []
        for i in range(self.args.time_length_con):
            if self.args.if_readout:
                features_list.append(self.oc(self.encoder_list[int(i//self.args.num_encoder)](features[:, i, :, :], adj)))
            else:
                features_list.append(self.encoder_list[int(i//self.args.num_encoder)](features[:, i, :, :], adj))
        features = torch.stack(features_list, dim=1).reshape(batch_size, self.args.time_length_con, -1)
        features = self.final_graphormer(features, None)
        features = features.view(features.shape[0], -1)
        features = self.linear1(features)
        features = self.relu1(features)
        features = self.linear2(features)
        features = self.relu2(features)
        features = self.linear3(features)
        features = self.relu3(features)
        features = self.linear4(features)
        return features

if __name__=='__main__':
    args = parse_args()
    tg = TimeGraphormer_con(args).to(args.device)
    # print(tg)
