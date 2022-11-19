# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:12:27 2019

@author: gk
"""

import os
import random
import shutil
from time import time
from datetime import datetime
import configparser
import argparse
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from lib.utils_model_pre1 import *
from model.model_pre1 import Net_block as model
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:4', help='')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate ')
FLAGS = parser.parse_args()
decay = FLAGS.decay
num_nodes = 69
epochs = 50
batch_size= FLAGS.batch_size
points_per_hour = 1
num_for_predict = 1
num_of_weeks = 2
num_of_days = 1
num_of_hours =3

merge = False
model_name = 'BILSTM12'
params_dir = 'experiment'
prediction_path = 'BILSTM_prediction'
wdecay = 0.000

device = torch.device(FLAGS.device)
print('read matrix')
# read matrix
adj_mx_list=[]
adj1 = './data/nyc_adj.pkl'
adj_mx1 = load_graph_data_hz(adj1)  # (80, 80)
# print(adj_mx1.shape)
for i in range(len(adj_mx1)):
    adj_mx1[i, i] = 0
adj_mx_list.append(adj_mx1)  # (1,80,80)

adj_mx = np.stack(adj_mx_list, axis=-1)#(80,80,3)
# print(adj_mx.shape)
adj_mx = adj_mx / (adj_mx.sum(axis=0) + 1e-18)  #(80, 80, 3)#进行row normalization
src, dst = adj_mx.sum(axis=-1).nonzero()   #原站点：(3214,),目的地：(3214,),
# print(src) #[ 0  0  0 ... 79 79 79]
# print(dst)  #[ 1  2  3 ... 76 77 78]
# print(src.shape) #(3214,)
# print(dst.shape)  #(3214,)
edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)   #torch.Size([2, 3214])
# print('2345')
# print(edge_index.shape)
edge_attr = torch.tensor(adj_mx[adj_mx.sum(axis=-1) != 0],
                         dtype=torch.float,
                         device=device)   ##torch.Size([3214, 3])
print(edge_attr.shape)#torch.Size([3])
print(edge_attr[0].shape)#torch.Size([3])

Metro_edge_matrix = np.load('./data/npy2018_data_1hour.npy')  #
Metro_week_matrix = np.load('./data/ext2018_week_Matrix.npy')  #
Metro_hour_matrix = np.load('./data/ext2018_hour_Matrix.npy')  #




print('Model is %s' % (model_name))

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
if params_dir != "None":
    params_path = os.path.join(params_dir, model_name)
else:
    params_path = 'params/%s_%s/' % (model_name, timestamp)


def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



if __name__ == "__main__":
    seed_torch(12)
    print('12')
    # read all data from graph signal matrix file
    print("Reading data...")
    # Input: train / valid  / test : length x 3 x NUM_POINT x 12
    all_data,scaler = read_and_generate_dataset(Metro_edge_matrix,Metro_week_matrix,Metro_hour_matrix,
                                         num_of_weeks,
                                         num_of_days,
                                         num_of_hours,
                                         num_for_predict,
                                         points_per_hour,
                                         merge)
    print('scaler,mean: %.6f,  std: %.6f' % (scaler.mean, scaler.std))

    # test set ground truth
    true_value = all_data['test']['target']
    true_val_value = all_data['val']['target']
    print(true_value.shape)

    # training set data loader
    train_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['train']['week']).to(device),
            torch.Tensor(all_data['train']['day']).to(device),
            torch.Tensor(all_data['train']['recent']).to(device),
            torch.Tensor(all_data['train']['target']).to(device),
            torch.Tensor(all_data['train']['week_toweek']).to(device),
            torch.Tensor(all_data['train']['week_tohour']).to(device),
            torch.Tensor(all_data['train']['day_toweek']).to(device),
            torch.Tensor(all_data['train']['day_tohour']).to(device),
            torch.Tensor(all_data['train']['recent_toweek']).to(device),
            torch.Tensor(all_data['train']['recent_tohour']).to(device)
        ),
        batch_size=batch_size,
        shuffle=True
    )

    # validation set data loader
    val_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['val']['week']).to(device),
            torch.Tensor(all_data['val']['day']).to(device),
            torch.Tensor(all_data['val']['recent']).to(device),
            torch.Tensor(all_data['val']['target']).to(device),
            torch.Tensor(all_data['val']['week_toweek']).to(device),
            torch.Tensor(all_data['val']['week_tohour']).to(device),
            torch.Tensor(all_data['val']['day_toweek']).to(device),
            torch.Tensor(all_data['val']['day_tohour']).to(device),
            torch.Tensor(all_data['val']['recent_toweek']).to(device),
            torch.Tensor(all_data['val']['recent_tohour']).to(device)
        ),
        batch_size=batch_size,
        shuffle=False
    )

    # testing set data loader
    test_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['test']['week']).to(device),
            torch.Tensor(all_data['test']['day']).to(device),
            torch.Tensor(all_data['test']['recent']).to(device),
            torch.Tensor(all_data['test']['target']).to(device),
            torch.Tensor(all_data['test']['week_toweek']).to(device),
            torch.Tensor(all_data['test']['week_tohour']).to(device),
            torch.Tensor(all_data['test']['day_toweek']).to(device),
            torch.Tensor(all_data['test']['day_tohour']).to(device),
            torch.Tensor(all_data['test']['recent_toweek']).to(device),
            torch.Tensor(all_data['test']['recent_tohour']).to(device)
        ),
        batch_size=batch_size,
        shuffle=False
    )

    # loss function MSE
    loss_function = nn.MSELoss()

    # get model's structure
    net = model(device,edge_index,edge_attr)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     net = nn.DataParallel(net)
    net.to(device)  # to cuda
    scaler = scaler
    scaler_torch = StandardScaler_Torch(scaler.mean, scaler.std, device=device)

    optimizer = optim.Adam(net.parameters(), lr=0.0005, weight_decay=wdecay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)


    his_mae = []
    train_time = []
    for epoch in range(1, epochs + 1):
        train_l = []
        start_time_train = time()
        for train_w, train_d, train_r, train_t,train_w_toweek,train_w_tohour, train_d_toweek,train_d_tohour, \
            train_r_toweek,train_r_tohour in train_loader:
            train_w = train_w.to(device)#
            train_d = train_d.to(device)#
            train_r = train_r.to(device)#
            train_t = train_t.to(device)#t
            train_w_toweek = train_w_toweek.to(device)  # torch.Size([16, T, 7])
            train_w_tohour = train_w_tohour.to(device)  # torch.Size([16, T, 108])
            train_d_toweek = train_d_toweek.to(device)  # torch.Size([16, T, 7])
            train_d_tohour = train_d_tohour.to(device)  # torch.Size([16, T, 108])
            train_r_toweek = train_r_toweek.to(device)  # torch.Size([16, T, 7])
            train_r_tohour = train_r_tohour.to(device)  # torch.Size([16, T, 108])
            net.train()  # train pattern
            optimizer.zero_grad()  # grad to 0
            output = net([train_w, train_d, train_r],
                         [train_w_toweek,train_w_tohour, train_d_toweek,train_d_tohour,train_r_toweek,train_r_tohour])
            output = scaler_torch.inverse_transform(output)  # 是将标准化后的数据转换为原始数据  #torch.Size([16, 80, 80, 6])
            train_t = scaler_torch.inverse_transform(train_t)  # torch.Size([16, 80, 80, 6])


            loss = loss_function(output, train_t)
            # backward p
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), clip)

            # update parameter
            optimizer.step()

            training_loss = loss.item()
            train_l.append(training_loss)
        scheduler.step()
        end_time_train = time()
        train_l = np.mean(train_l)
        print('epoch step: %s, training loss: %.2f, time: %.2fs'
              % (epoch, train_l, end_time_train - start_time_train))
        train_time.append(end_time_train - start_time_train)

        # compute validation loss
        # print('compute_val_loss')
        valid_loss, val_mae, val_rmse = compute_val_loss(net, val_loader,true_val_value, loss_function,device, epoch,scaler)

        his_mae.append(val_mae)

        params_filename = os.path.join(params_path,
                                       '%s_epoch_%s_%s.params' % (model_name,
                                                                  epoch, str(round(val_mae, 4))))
        torch.save(net.state_dict(), params_filename)
        print('save parameters to file: %s' % (params_filename))

    print("Training finished")
    print("Training time/epoch: %.2f secs/epoch" % np.mean(train_time))

    bestid = np.argmin(his_mae)

    print("The valid loss on best model is epoch%s_%s" % (str(bestid + 1), str(round(his_mae[bestid], 4))))
    best_params_filename = os.path.join(params_path,
                                        '%s_epoch_%s_%s.params' % (model_name,
                                                                   str(bestid + 1), str(round(his_mae[bestid], 4))))
    net.load_state_dict(torch.load(best_params_filename))
    start_time_test = time()
    prediction= predict(net, test_loader,device)

    end_time_test = time()
    evaluate(net, test_loader, true_value, device, epoch,scaler)
    test_time = np.mean(end_time_test - start_time_test)
    print("Test time: %.2f" % test_time)

    # print('model_xiu2中')



    # prediction = scaler_torch.inverse_transform(prediction)  # 是将标准化后的数据转换为原始数据  #(316, 80, 80, 6)
    # true_value = scaler_torch.inverse_transform(true_value)  # (316, 80, 80, 6)
    #
    # np.save('model_prediction_pre1', prediction)
    # np.save('model_true_value_pre1', true_value)














