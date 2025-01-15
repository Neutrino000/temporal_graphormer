import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proj_dir', default='/home/son/PycharmProjects/tg_revised/tg_cobre/', type=str, required=False, help='Path of projects')
    parser.add_argument('--dataset_dir', default='/media/son/yoshida2/tg_dataset/cobre')
    parser.add_argument('--temp_dir', default='template/AAL_61x73x61_YCG.nii')
    parser.add_argument('--datasets', default='cobre', type=str, required=False, help='adni-cobre, Directory with all datasets, including fmri_matrix, roi_time_course, correlation_matrix')
    parser.add_argument('--num_workers', default=4, type=int, help='Workers in dataloader')

    parser.add_argument('--model_name', default='tumor', type=str, required=False, help='pretrained feature extraction model to use--hippocampus, tumor')
    parser.add_argument('--config_name', default='', type=str, help='pretrained config path')
    parser.add_argument('--output_dir', default='/home/son/PycharmProjects/tg_revised/tg_cobre/output', type=str, help='output directory')
    parser.add_argument('--model_dir', default='/model/', type=str)
    parser.add_argument('--loss_dir', default='/loss/', type=str)
    parser.add_argument('--config_dir', default='/config_setup/', type=str)

    parser.add_argument('--ifgraphormer', default=True, help='if use graphormer in the first stage')
    parser.add_argument('--per_gpu_train_batch_size', default=1, type=int, help='Batch size per GPU for training')
    parser.add_argument('--per_gpu_eval_batch_size', default=1, type=int, help='Batch size per GPU for evaluation')
    parser.add_argument('--lr', '--learning_rate', default=5e-5, type=float, help='The initial lr')
    parser.add_argument('--lr_unet', '--learning_rate_unet', default=1e-5, type=float, help='The initial lr of unet')
    parser.add_argument('--unet_train', default=True, help='if train unet')
    parser.add_argument('--num_train_epochs', default=20, type=int, help='Total number of training epochs to perform')
    parser.add_argument('--drop_out', default=0.4, type=float, help='dropout ratio')
    parser.add_argument('--num_hidden_layers', default=4, type=int, required=False, help='Update model config if given')
    parser.add_argument('--hidden_size', default=-1, type=int, required=False, help='Update model config if given')
    parser.add_argument('--num_attention_heads', default=4, type=int, required=False,
                        help='The division of hidden_size / num_attention_heads should be in integer')
    parser.add_argument('--intermediate_size', default=-1, type=int, required=False,
                        help='Update model config if given')
    parser.add_argument('--input_feat_dim', default='512,256,128', type=str,
                        help='The feature dimension')
    parser.add_argument('--hidden_feat_dim', default='128,64,32', type=str,
                        help='The hidden feature dimension')
    parser.add_argument('--which_gcn', default='1,1,1', type=str,
                        help='which encoder block to have graph conv')
    parser.add_argument('--test_per_train', default=1, type=int, help='how many epochs to test')
    parser.add_argument('--which_class', default='all', type=str, required=False,
                        help='which steps to calssification--all, 01, 02, 03, 04, 05 if adni')

    parser.add_argument('--run_eval_only', default=True, action='store_true')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Log every X steps')
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
    parser.add_argument('--ratio_train', default=0.9, type=float, help='ratio of training samples')
    parser.add_argument('--seed', type=int, default=3407, help='random seed for initialization')
    parser.add_argument('--local_rank', type=int, default=0, help='For distributed training')
    parser.add_argument('--time_course', default='dpabi', type=str, help='time course of roi source--dpabi or fmri')
    parser.add_argument('--time_course_cate', default='mean', type=str, help='time course category--mean or seed')
    parser.add_argument('--feat_source', default='fmri', type=str, help='fmri source for feature extraction--dpabi or fmri')
    parser.add_argument('--ifgroup', default='individual', type=str, help='if use group analysis--individual or global')
    parser.add_argument('--fmri_cate', default='FunImgARCWsymS', type=str, help='fmri category--FunImgARCW, FunImgARCWsym'
                                                                                'FunImgARCWsymS, FunImgARglobalCW, '
                                                                                'FunImgARglobalCWsym, FunImgARglobalCWsymS')

    parser.add_argument('--sampling', default='continue', type=str, help='which sampling method to use--continue, discrete')
    parser.add_argument('--time_points', default=140, type=int, help='number of time points')

    '''parameters for continue sampling method'''
    parser.add_argument('--time_length_con', default=30, type=int, help='number of time points for each sample--continue sampling')
    parser.add_argument('--interval_con', default=0, type=int, help='number of time points for interval--continue sampling')
    parser.add_argument('--num_segment_con', default=2, type=int, help='number of segments--continue sampling')
    parser.add_argument('--num_encoder', default=6, type=int, help='number of encoder')

    '''parameters for discrete sampling method'''
    parser.add_argument('--time_length', default=40, type=int, help='number of time points for each segment--discrete sampling')
    parser.add_argument('--interval', default=5, type=int, help='number of time points for interval--discrete sampling')
    parser.add_argument('--num_segment', default=20, type=int, help='number of segments--discrete sampling')
    parser.add_argument('--segment_proc', default='point', type=str, help='type of process of segment--mean or point--discrete sampling')

    parser.add_argument('--num_roi', default=116, type=int, help='num of roi')
    parser.add_argument('--unet_cate', default='tumor', type=str, help='which pretrained unet is used--')
    parser.add_argument('--binary', default=True, help='if use binary adjacency matrix')
    parser.add_argument('--ratio_bi', default=0.65, type=float, help='ratio of binarized elements in adj matrix')
    parser.add_argument('--ratio_roi', default=8.0, type=float, help='ratio of roi to others')
    parser.add_argument('--if_readout', default=True, help='if use readout')
    parser.add_argument('--readout', default='oc', type=str, help='which readout layer to use')
    parser.add_argument('--k_oc', default=10, type=int, help='number of oc clustering')
    parser.add_argument('--init_oc', default='learnable', type=str, help='which init on oc')
    args = parser.parse_args()
    return args

def hippo_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_inchannels', default=116, type=int)
    parser.add_argument('--num_inchannels_ori', default=1, type=int)
    parser.add_argument('--base_num_features', default=32, type=int)
    parser.add_argument('--num_classes', default=3, type=int)
    parser.add_argument('--pool_kernel_size', default=[[2, 2, 2], [2, 2, 2], [2, 2, 2]], type=list)
    parser.add_argument('--pool_kernel_size_ori', default=[[2, 2, 2], [2, 2, 2], [2, 2, 2]], type=list)
    parser.add_argument('--conv_per_stage', default=2, type=int)
    parser.add_argument('--conv_kernel_size', default=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], type=list)
    parser.add_argument('--conv_kernel_size_ori', default=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], type=list)
    parser.add_argument('--path_model', type=str, default='/media/son/yoshida2/tg_dataset/pretrained_model/hippocampus/fold_0/model_final_checkpoint.model')
    args = parser.parse_args()
    return args

def tumor_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_inchannels', default=116, type=int)
    parser.add_argument('--num_inchannels_ori', default=4, type=int)
    parser.add_argument('--base_num_features', default=32, type=list)
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--pool_kernel_size', default=[[2, 2, 2], [2, 2, 2], [2, 2, 2]], type=list)
    parser.add_argument('--pool_kernel_size_ori', default=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]], type=list)
    parser.add_argument('--conv_per_stage', default=2, type=int)
    parser.add_argument('--conv_kernel_size', default=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], type=list)
    parser.add_argument('--conv_kernel_size_ori', default=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], type=list)
    parser.add_argument('--path_model', type=str, default='/media/pretrained_model/tumor/fold_0/model_final_checkpoint.model')
    args = parser.parse_args()
    return args


def tg_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_feat_dim', default='64,32', type=str, help='The Image Feature Dimension')
    parser.add_argument('--hidden_feat_dim', default='16,8', type=str, help='The Image Feature Dimension')
    parser.add_argument('--inter_out_dim', default=4, type=int, help='The output dim of inter graphormer layer')
    parser.add_argument('--which_gcn', default='1,1', type=str, help='which encoder block to have graph conv')
    parser.add_argument('--num_graph', default=2, type=int, help='number of graph_conv')
    parser.add_argument('--num_hidden_layers', default=4, type=int, required=False)
    parser.add_argument('--hidden_size', default=-1, type=int, required=False)
    parser.add_argument('--num_attention_heads', default=4, type=int, required=False)
    parser.add_argument('--intermediate_size', default=-1, type=int, required=False)
    parser.add_argument('--interm_size_scale', default=2, type=int)
    parser.add_argument('--config_name', default='/home/son/PycharmProjects/cobre_tg_10fold/pretrained_model/config.json')
    parser.add_argument('--ifsagp', default=True, help='if use self-attention graph pooling')
    parser.add_argument('--sagp_pool_ratio', default=0.7, type=float, help='ratio of sagp pooling')
    args = parser.parse_args()
    return args

def fg_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_feat_dim', default='224,64', type=str, help='The Image Feature Dimension')
    parser.add_argument('--hidden_feat_dim', default='32,16', type=str, help='The Image Feature Dimension')
    parser.add_argument('--linear_dim', default=[128, 64, 32], type=list, help='dim of the last linear layer')
    parser.add_argument('--inter_out_dim', default=8, type=int, help='The output dim of inter graphormer layer')
    parser.add_argument('--which_gcn', default='0,0,0', type=str, help='which encoder block to have graph conv')
    parser.add_argument('--num_hidden_layers', default=4, type=int, required=False)
    parser.add_argument('--hidden_size', default=-1, type=int, required=False)
    parser.add_argument('--num_attention_heads', default=4, type=int, required=False)
    parser.add_argument('--intermediate_size', default=-1, type=int, required=False)
    parser.add_argument('--interm_size_scale', default=2, type=int)
    parser.add_argument('--config_name', default='/home/son/PycharmProjects/cobre_tg_10fold/pretrained_model/config.json')
    parser.add_argument('--ifsagp', default=False, help='if use self-attention graph pooling')
    parser.add_argument('--sagp_pool_ratio', default=0.8, type=float, help='ratio of sagp pooling')
    args = parser.parse_args()
    return args
