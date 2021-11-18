#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import math
#import tensorflow as tf
from numpy import linalg as LA
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    P_t = 1
    N = 5000
    d = 0;
    for k in w_glob.keys():
        d += torch.numel(w_glob[k])
    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        h_t_re = np.random.normal(0, math.sqrt(1/2), args.num_users)
        h_t_im = np.random.normal(0, math.sqrt(1/2), args.num_users)
        h_t = []
        for i in range(args.num_users):
            h_t.append(math.sqrt(h_t_re[i] * h_t_re[i] + h_t_im[i] * h_t_im[i]))
        noise_locals = np.random.normal(0, 1, args.num_users)
        C_t = []
        r_q = []
        for h in h_t:
            tempc = math.log(1 + math.pow(abs(h), 2) * P_t, 2)
            C_t.append(tempc)
            r_q.append(N * tempc)
        # q_t is q*(t) in paper
        q_t = []
        for r in r_q:
            q_local = 1
            while r > math.log(math.comb(d, q_local), 2) + 33 * q_local:
                q_local = q_local + 1
            q_t.append(q_local - 1)
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        #idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users = range(args.num_users)
        norm_deltaw = []
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            #calculate delta w
            #norm = 0
            #for k in w_glob.keys():
                #norm += math.pow(torch.norm(w[k]-w_glob[k]), 2)
            norm = 0
            deltaw_list = []
            for k in w_glob.keys():
                delta_w = w[k] - w_glob[k]
                deltaw_list.append(delta_w.resize(1, torch.numel(delta_w)))
            big_tensor = torch.cat(deltaw_list, 1)
            topvalues, topindices = torch.topk(big_tensor, q_t[idx])
            bottomvalues, bottomindices = torch.topk(big_tensor, q_t[idx], largest=False)
            values = torch.cat((topvalues, bottomvalues), 1).cpu().numpy()
            values = np.reshape(values, values[0].size)
            values = np.abs(values)
            largest_q_idx = np.argpartition(values, 0-q_t[idx])[(0-q_t[idx]):]
            norm = LA.norm([values[x] for x in largest_q_idx])
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
                norm_deltaw[idx] = copy.deepcopy(norm)
            else:
                w_locals.append(copy.deepcopy(w))
                norm_deltaw.append(copy.deepcopy(norm))
            loss_locals.append(copy.deepcopy(loss))
            #print(idx,'\n')

        # choose users to update in this round
        k = int(args.num_users * args.frac)
        norm_deltaw = np.array(norm_deltaw)
        idxs_users = norm_deltaw.argsort()[(0-k):]
        w_locals = [w_locals[x] for x in idxs_users]
        loss_locals = [loss_locals[x] for x in idxs_users]

        #calculate number of time slots allocated to clients
        n_k = []   # n_k[i]-->idxs_users[i]
        equation21_denominator = 0
        for j in idxs_users:
            temp = 1
            for i in idxs_users:
                if(i != j):
                    temp = temp * C_t[i]
            equation21_denominator = equation21_denominator + norm_deltaw[j] * temp
        for j in idxs_users:
            temp = 1
            for i in idxs_users:
                if(i != j):
                    temp = temp * C_t[i]
            n_k.append(int(norm_deltaw[j] * temp * N / equation21_denominator))

        #find q_m(t) for the selected clients
        q_m_t = []
        for i in idxs_users:
            q_local = 1
            while C_t[i] * n_k[idxs_users.tolist().index(i)] > math.log(math.comb(d, q_local), 2) + 33 * q_local:
                q_local = q_local + 1
            q_m_t.append(q_local - 1)
        
        #consider wireless communication, change w
        for idx in range(len(w_locals)):
            deltaw_list = []
            for k in w_glob.keys():
                delta_w = w_locals[idx][k] - w_glob[k]
                deltaw_list.append(delta_w.resize(1, torch.numel(delta_w)))
            big_tensor = torch.cat(deltaw_list, 1)
            topvalues, topindices = torch.topk(big_tensor, q_m_t[idx])
            bottomvalues, bottomindices = torch.topk(big_tensor, q_m_t[idx], largest=False)
            values = torch.cat((topvalues, bottomvalues), 1).cpu().numpy()
            values = np.reshape(values, values[0].size)
            values_abs = np.abs(values)
            largest_q_idx = np.argpartition(values_abs, 0-q_m_t[idx])[(0-q_m_t[idx]):]
            values = [values[x] for x in largest_q_idx]
            v_min = min(values)
            quantize_d = (max(values) - v_min) / math.pow(2, 33);
            quantized_values = []
            for v in values:
                temp_v = int((v - v_min) / quantize_d) * quantize_d + v_min
                quantized_values.append(temp_v)
            w_result = copy.deepcopy(w_glob)
            for i in range(len(values)): 
                for k in w_locals[idx].keys():
                    indices = (w_locals[idx][k] - w_glob[k] == values[i]).nonzero(as_tuple=False)
                    indices = indices.detach().to("cpu").numpy()
                    print(indices)
                    for index in indices:
                        w_result[k][tuple(index)] += quantized_values[i]
            w_locals[idx] = w_result

        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

