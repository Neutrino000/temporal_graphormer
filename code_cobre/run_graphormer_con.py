import torch
import torch.nn as nn
import os
import argparse
import os.path as op
import json
import time
import cv2
import numpy as np
from src.utils.comm import synchronize, get_rank
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.logger import setup_logger
from data_loader import Mydataset
from torch.utils.data import DataLoader
from time_graphormer_batch_con import TimeGraphormer_con, parse_args, hippo_args, tumor_args, tg_args, fg_args
from time_graphormer_batch import TimeGraphormer
from src.utils.metric_logger import AverageMeter, EvalMetricsLogger
import datetime
from tqdm import tqdm
import warnings
from utilities_func import save_args_to_txt, merged_args

def get_current_time():
    current_time = datetime.now()
    year = current_time.year
    month = current_time.month
    day = current_time.day
    hour = current_time.hour
    minute = current_time.minute
    second = current_time.second
    formatted_time = f"{month:02d}-{day:02d}-{hour:02d}{minute:02d}{second:02d}"
    return formatted_time

def run(args, train_dataloader, test_dataloader, Graphormer_model, current_time, fd):
    if args.unet_train:
        params = [
            {'params': list(Graphormer_model.unet.parameters()), 'lr': args.lr_unet},
            {'params': list(Graphormer_model.final_graphormer.parameters()), 'lr': args.lr},
            {'params': list(Graphormer_model.linear1.parameters()), 'lr': args.lr},
            {'params': list(Graphormer_model.linear2.parameters()), 'lr': args.lr},
            {'params': list(Graphormer_model.linear3.parameters()), 'lr': args.lr},
            {'params': list(Graphormer_model.linear4.parameters()), 'lr': args.lr},
        ]
        for i in range(args.num_encoder):
            params.append({'params': list(Graphormer_model.encoder_list[i].parameters()), 'lr': args.lr})
        optimizer = torch.optim.Adam(params)
    else:
        optimizer = torch.optim.Adam(list(Graphormer_model.parameters()),
                                     lr=args.lr,
                                     betas=(0.9, 0.999),
                                     weight_decay=0)
    criterion = torch.nn.CrossEntropyLoss().cuda(args.device)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    if args.distributed:
        Graphormer_model = torch.nn.parallel.DistributedDataParallel(
            Graphormer_model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    #model_dir = args.dataset_dir + 'cobre/model_10fold/fold_'+ str(fd)+'/'
    mkdir('/media/son/yoshida2/tg_dataset/cobre/model_10fold_revised/fold_'+str(fd))
    #Graphormer_model.train()

    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []
    acc_compare = 0
    processing = tqdm(range(args.num_train_epochs))

    if args.unet_cate == 'tumor':
        unet_args = tumor_args()
    else:
        unet_args = hippo_args()
    fg = fg_args()
    tg = tg_args()
    args_list = merged_args([args, unet_args, fg, tg])
    save_args_to_txt(args_list, args.config_dir + current_time + '_fold_' +str(fd)+'.txt')

    for epoch in processing:
        num_batch = 0
        acc = 0
        loss_list = []
        for iteration, (fmri_mat, fc_mat, label, sub_name) in enumerate(train_dataloader):
            Graphormer_model.train()
            iteration += 1
            batch_size = fmri_mat.size(0)
            fmri_mat = fmri_mat.cuda(args.device)
            fc_mat = fc_mat.cuda(args.device).to(torch.float)
            label = label.cuda(args.device)
            output = Graphormer_model(fmri_mat, fc_mat)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.cpu().detach().numpy())
            num_batch += 1
            for i in range(batch_size):
                if torch.argmax(output[i, :]) == label[i,]:
                    acc += 1
            acc_inter = acc / (num_batch * batch_size)
            if epoch > 0:
                processing.set_description(
                    'Epoch %d, Train loss %.4f, Train acc %.3f, Test loss %.4f, Test acc %.3f' % (
                    epoch, np.mean(loss_list), acc_inter, loss_test[epoch - 1], acc_test[epoch - 1]))
        acc = acc / (len(train_dataloader) * batch_size)
        acc_train.append(acc)
        np.savetxt(args.output_dir + '/loss/acc_train_fold' + str(fd) + '.txt', acc_train)
        loss_train.append(np.mean(loss_list))
        np.savetxt(args.output_dir + '/loss/loss_train_fold' + str(fd) + '.txt', loss_train)

        scheduler.step()

        acc_t = 0
        loss_list = []
        Graphormer_model.eval()
        with torch.no_grad():
            for iteration, (fmri_mat, fc_mat, label, sub_name) in enumerate(test_dataloader):
                batch_size = fmri_mat.size(0)
                fmri_mat = fmri_mat.cuda(args.device)
                fc_mat = fc_mat.cuda(args.device).to(torch.float)
                label = label.cuda(args.device)
                output = Graphormer_model(fmri_mat, fc_mat)
                loss = criterion(output, label)
                loss_list.append(loss.cpu().detach().numpy())
                for i in range(batch_size):
                    if torch.argmax(output[i, :]) == label[i]:
                        acc_t += 1
            acc_t = acc_t / (len(test_dataloader) * batch_size)
            acc_test.append(acc_t)
            np.savetxt(args.output_dir + '/loss/acc_test_fold' + str(fd) + '.txt', acc_test)
            loss_test.append(np.mean(loss_list))
            np.savetxt(args.output_dir + '/loss/loss_test_fold' + str(fd) + '.txt', loss_test)
            # print('Test loss = '+ str(loss_test[epoch//args.test_per_train]) + ' Train loss = '+str(acc_train[epoch//args.test_per_train]))

            acc_compare = acc_t
            current_time = get_current_time()
            torch.save(Graphormer_model.state_dict(), '/media/son/yoshida2/tg_dataset/cobre/model_10fold_revised/fold_'+str(fd) + '/Graphormer_' + current_time + '_'+str(acc_t)+'.pth')
        torch.cuda.empty_cache()

def main():
    global logger
    for fd in range(10):
        print('=======fold_'+str(fd)+'=======')
        args = parse_args()
        current_time = get_current_time()
        args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
        print('set os.environ[OMP_NUM_THREADS] to {}'.format(os.environ['OMP_NUM_THREADS']))

        args.distributed = args.num_gpus > 1
        args.device = torch.device(args.device)
        args.temp_dir = args.proj_dir + args.temp_dir
        if args.distributed:
            print("Init distributed training on local rank {} ({}), rank {}, world size {}".format(args.local_rank, int(os.environ["LOCAL_RANK"]), int(os.environ["NODE_RANK"]), args.num_gpus))
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend='nccl', init_method='env://'
            )
            local_rank = int(os.environ["LOCAL_RANK"])
            args.device = torch.device("cuda", local_rank)
            synchronize()

        args.output_dir = args.output_dir + '_' + args.datasets
        args.model_dir = args.output_dir + args.model_dir
        args.loss_dir = args.output_dir + args.loss_dir
        args.config_dir = args.output_dir + args.config_dir
        mkdir(args.output_dir)
        if args.datasets=='cobre':
            args.time_points = 150
        else:
            args.time_points = 140
        logger = setup_logger('Graphormer', args.output_dir, get_rank())
        set_seed(args.seed, args.num_gpus)
        logger.info('Using {} GPUs'.format(args.num_gpus))
        if args.sampling=='continue':
            train_path = args.output_dir + '/random_list/train_fold'+str(fd)+'.txt'
            test_path = args.output_dir + '/random_list/test_fold'+str(fd)+'.txt'
        else:
            train_path = args.output_dir + '/random_list/train_fold'+str(fd)+'.txt'
            test_path = args.output_dir + '/random_list/test_fold'+str(fd)+'.txt'
        # temp_train_dir = 'train_fold' + str(fd)#_' + train_path[-16:-14] + train_path[-13:-11] + train_path[-10:-8]
        # temp_test_dir = 'test_fold' + str(fd)#_' + test_path[-16:-14] + test_path[-13:-11] + test_path[-10:-8]
    # mkdir(args.dataset_dir + args.datasets + '/' + 'temporary_dataset/' + temp_train_dir)
    # mkdir(args.dataset_dir + args.datasets + '/' + 'temporary_dataset/' + temp_test_dir)
        temp_dir = '/media/son/yoshida2/tg_dataset/cobre/dataset/clip_dataset_revised/'

        if args.sampling=='continue':
            model = TimeGraphormer_con(args).to(args.device)
        else:
            model = TimeGraphormer(args).to(args.device)

        if args.run_eval_only==True:
            test_data = Mydataset(args, temp_dir, test_path, fd)
            test_data_loader = DataLoader(test_data, batch_size=args.per_gpu_eval_batch_size, shuffle=False)

        else:
            train_data = Mydataset(args, temp_dir, train_path, fd)
            train_data_loader = DataLoader(train_data, batch_size=args.per_gpu_train_batch_size, shuffle=True)
            test_data = Mydataset(args, temp_dir, test_path, fd)
            test_data_loader = DataLoader(test_data, batch_size=args.per_gpu_eval_batch_size, shuffle=False)

            run(args, train_data_loader, test_data_loader, model, current_time, fd)

if __name__=='__main__':
    main()
