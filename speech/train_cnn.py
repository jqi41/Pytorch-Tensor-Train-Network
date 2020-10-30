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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Training a Convolutional Neural Network...')
parser.add_argument('--batch_size', default=200, help='Mini-batch size')
parser.add_argument('--input_dim', default=17*256, help='Input dimension')
parser.add_argument('--hidden_layer', default=1024, help='Neurons in the hidden layer')
parser.add_argument('--output_dim', default=256, help='Output dimension')
parser.add_argument('--lr', default=0.001, help='Learning rate')
parser.add_argument('--n_chans', default=[32, 64, 128, 256], help='Convolutional channels')
parser.add_argument('--test_data_path', metavar='DIR', default='exp//test_feats//feats_0.h5', help='Feature container in h5 format')
parser.add_argument('--train_data_path', metavar='DIR', default='exp//train_feats//feats_', help='Feature container in h5 format')
parser.add_argument('--save_model_path', default='model_tt.hdf5', help='The path to the saved model')
parser.add_argument('--n_epochs', default=10, help='The total number of epochs', type=int)
parser.add_argument('--n_data', default=23, help='How many datasets do we need?')
args = parser.parse_args()


def param(nnet, Mb=True):
    """
    Return the number of parameters in nnet. 
    """
    nelems = float(sum([param.nelement() for param in nnet.parameters()]))

    return nelems / 10**6 if Mb else nelems


print('Building a CNN Model...')
class cnn_model(nn.Module):
    def __init__(self, n_chans, hidden_layer, output_dim):
        super(cnn_model, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, n_chans[0], 5, stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(n_chans[0], n_chans[1], 4, stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(n_chans[1], n_chans[2], 4, stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(n_chans[2], n_chans[3], 4, stride=(2, 2), padding=(1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(15*n_chans[3], hidden_layer),
            nn.Linear(hidden_layer, 400),
            nn.Linear(400, output_dim)
        )
        self.output_dim = output_dim 
        self.n_chans = n_chans 

    def forward(self, x):
        x = x.view(x.size(0), 1, self.n_chans[3], 17)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), 1*15*self.n_chans[3])
        out = self.fc(x)

        return out 


if __name__ == "__main__":

    output_dims = args.output_dim
    model = cnn_model(args.n_chans, args.hidden_layer, args.output_dim).to(device)

    print("Model's state_dict:")
    for param in model.state_dict():
        print(param, "\t", model.state_dict()[param].size())
        
    print('The model size = '.format(param(model, False)))

    losses_train = []
    losses_test = []
    loss_mse = nn.MSELoss()
    loss_mae = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))
    #optimizer = optim.SGD(model.parameters(), lr=float(args.lr), momentum=0.2)

    print('Training the Deep Neural Network...')
    test_data = h5py.File(args.test_data_path, 'r')
    test_clean = test_data['clean']
    test_noise = test_data['noise']
    nframe_test_clean, _ = test_clean.shape 

    for epoch in range(args.n_epochs):
        total_loss_train = 0
        total_loss_test = 0
        for data_idx in range(int(args.n_data)): 
            train_data_fn = h5py.File(args.train_data_path + str(data_idx) + '.h5', 'r')
            train_clean = train_data_fn['clean']
            train_noise = train_data_fn['noise']

            nframe_train_noise, input_dims = train_noise.shape 
            n_batch_train = int(nframe_train_noise / args.batch_size)
            n_batch_test = int(nframe_test_clean / args.batch_size)

            for idx in range(n_batch_train):
                model.zero_grad()
                train_clean_batch = torch.tensor(train_clean[idx*args.batch_size : (idx+1)*args.batch_size, :]).type(dtype)
                train_noisy_batch = torch.tensor(train_noise[idx*args.batch_size : (idx+1)*args.batch_size, :]).type(dtype)

                probs = model(train_noisy_batch.reshape(-1, input_dims))
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
        model_path = "exp/cnn_models/model_" + str(epoch) + str(".hdf5")
        torch.save({ 
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_train': total_loss_train,
            'loss_test': total_loss_test
        }, model_path)

