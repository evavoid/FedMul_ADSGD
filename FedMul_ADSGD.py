"""
-----------------------------------------------------------------------------------
--  @file       MLFoglib->FedMul_ADSGD.py
--  @author     Ma Haoming(En:louis)(https://github.com/evavoid)
--  @brief      xxxxxxxxx
    
--  @Ide        PyCharm
--  @time       2020/12/12-21:01
-----------------------------------------------------------------------------------
"""
import modules.modules_base as mm_base
import datasets.dataset_base as data_base
import browser.options_base as op_base
import matplotlib.pyplot as plt

import torch

args = op_base.args_parser()
dataset = data_base.DatasetGet(args)

mnist_local_net = [mm_base.LocalMnist(args, dataset["mnist_train_users"][i]).to(torch.device('cuda:0')) for i in
                   range(args.musers_num)]
fmnist_local_net = [mm_base.LocalMnist(args, dataset["fmnist_train_users"][i]).to(torch.device('cuda:0')) for i in
                    range(args.fusers_num)]
mnist_global_net = mm_base.GlobalMnist(args).to(torch.device('cuda:0'))
fmnist_global_net = mm_base.GlobalMnist(args).to(torch.device('cuda:0'))
# 主网络参数全部归零

for epo in range(args.epochs):
    for muser in range(args.musers_num):
        # 主网络参数更新子网络
        print("mnist_train--epochs:%d," % epo+"muser:%d" % muser)
        mnist_local_net[muser].start_local_train()
        mnist_local_net[muser].start_compress_grad()
    for fuser in range(args.fusers_num):
        # 主网络参数更新子网络
        print("fmnist_train--epochs:%d," % epo+"fuser:%d" % fuser)
        fmnist_local_net[fuser].start_local_train()
        fmnist_local_net[fuser].start_compress_grad()
    # grad合成
    # 解析出grad
    # 更新mnist_global_net参数
    # 更新fmnist_global_net参数
mnist_local_net[0].start_local_test(dataset["mnist_test_origin"])
