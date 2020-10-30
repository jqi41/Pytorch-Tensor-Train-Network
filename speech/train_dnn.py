#!/usr/bin/env python3 

import os 
import sys 
import numpy as np 
import argparse 
import time 
import h5py 

import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 

seed = 7 
np.random.seed(seed)
torch.manual_seed(seed)
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Training a tensor-train neural network...')
parser.add_argument('--batch_size', default=200, help='Mini-batch size')
parser.add_argument('--input_dim', default=17*256)
parser.add_argument('--output_dim', default=256)
parser.add_argument('--lr', default=0.001, help='Learning rate')
parser.add_argument('--hidden_layers', default=[1024, 1024, 1024, 2048])
parser.add_argument('--data_path', metavar='DIR', default='exp//test_feats//feats_0.h5', help='Feature container in h5 format')
parser.add_argument('--save_model_path', default='model_tt.hdf5', help='The path to the saved model')
parser.add_argument('--n_epochs', default=100, help='The total number of epochs', type=int)
parser.add_argument('--n_data', default=7, help='How many datasets do we need?')
args = parser.parse_args()


def param(nnet, Mb=True):
    """
    Return the number of parameters in nnet. 
    """
    nelems = float(sum([param.nelement() for param in nnet.parameters()]))

    return nelems / 10**6 if Mb else nelems


print('Building a DNN model...')
class dnn_model(nn.Module):
    def __init__(self, hidden_layers, input_dim, output_dim):
        super(dnn_model, self).__init__()
        self.Linear1 = nn.Linear(input_dim, hidden_layers[0])
        self.Linear2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.Linear3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.Linear4 = nn.Linear(hidden_layers[2], hidden_layers[3])
        self.Linear5 = nn.Linear(hidden_layers[3], output_dim)
        
        with torch.no_grad():
            self.Linear1.weight.div_(torch.norm(self.Linear1.weight, dim=1, keepdim=True))
            self.Linear2.weight.div_(torch.norm(self.Linear2.weight, dim=1, keepdim=True))
            self.Linear3.weight.div_(torch.norm(self.Linear3.weight, dim=1, keepdim=True))
            self.Linear4.weight.div_(torch.norm(self.Linear4.weight, dim=1, keepdim=True))
            #self.Linear5.weight.div_(torch.norm(self.Linear5.weight, dim=0, keepdim=True))

    def forward(self, inputs):
        out = F.relu(self.Linear1(inputs))
        out = F.relu(self.Linear2(out))
        out = F.relu(self.Linear3(out))
        out = F.relu(self.Linear4(out))
        out = self.Linear5(out)

        return out 


if __name__ == "__main__":

    output_dims = args.output_dim
    model = dnn_model(args.hidden_layers, args.input_dim, output_dims).to(device)

    print("Model's state_dict:")
    for param in model.state_dict():
        print(param, "\t", model.state_dict()[param].size())

    losses_train = []
    losses_test = []
    loss_mae = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))
    #optimizer = optim.SGD(model.parameters(), lr=float(args.lr), momentum=0.2)

    print('Training the Deep Neural Network...')
    test_data = h5py.File(args.data_path)
    test_clean = test_data['test_clean']
    test_noise = test_data['test_noisy']
    nframe_test_clean, _ = test_clean.shape 

    for epoch in range(args.n_epochs):
        total_loss_train = 0
        total_loss_test = 0
        for data_idx in range(int(args.n_data)): 
            train_clean = test_data['train_clean']
            train_noise = test_data['train_noisy']

            nframe_train_noise, input_dims = train_noise.shape 
            n_batch_train = int(nframe_train_noise / args.batch_size)
            n_batch_test = int(nframe_test_clean / args.batch_size)

            for idx in range(n_batch_train):
                model.zero_grad()
                train_clean_batch = torch.tensor(train_clean[idx*args.batch_size : (idx+1)*args.batch_size, :]).type(dtype)
                train_noisy_batch = torch.tensor(train_noise[idx*args.batch_size : (idx+1)*args.batch_size, :]).type(dtype)
        
                probs = model(train_noisy_batch.reshape(-1, np.prod(args.input_dim)))
                loss = loss_mae(probs, train_clean_batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss_train += loss.item()
                if idx % 200 == 0:
                    tmp_loss = total_loss_train / (idx+1)
                    print('Epoch{0}, data_idx={1} idx={2}, loss={3}'.format(epoch, data_idx, idx, tmp_loss))
            total_loss_train /= n_batch_train 

        for idx in range(n_batch_test):
            eval_clean_batch = torch.tensor(test_clean[idx*args.batch_size : (idx+1)*args.batch_size, :]).type(dtype)
            eval_noisy_batch = torch.tensor(test_noise[idx*args.batch_size : (idx+1)*args.batch_size, :]).type(dtype)
            probs = model(eval_noisy_batch.reshape(-1, args.input_dim))
            loss = loss_mae(probs, eval_clean_batch)
            total_loss_test += loss.item()
        total_loss_test /= n_batch_test

        print('Epoch: {}, total_loss: {}  |   test_loss: {}'.format(epoch, total_loss_train, total_loss_test))
        losses_train.append(total_loss_train)
        losses_test.append(total_loss_test)
        model_path = "dnn_model_" + str(epoch) + str(".hdf5")
        torch.save({ 
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_train': total_loss_train,
            'loss_test': total_loss_test
        }, model_path)
