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

import tc 
from tc.tc_fc import TTLinear 

seed = 7 
np.random.seed(seed)
torch.manual_seed(seed)
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Training a tensor-train neural network...')
parser.add_argument('--batch_size', default=200, help='Mini-batch size')
parser.add_argument('--input_tensor', default=[4, 4, 16, 3])
parser.add_argument('--hidden_tensors', default=[[8, 4, 8, 4], [8, 4, 8, 4], [8, 4, 8, 8]])
parser.add_argument('--data_path', metavar='DIR', default='exp/feats/feats_m109_15dB.h5', help='Feature container in h5 format')
parser.add_argument('--save_model_path', default='model_tt.hdf5', help='The path to the saved model')
parser.add_argument('--n_epochs', default=5, help='The total number of epochs', type=int)
args = parser.parse_args()
 
# Reading data
data = h5py.File(args.data_path, 'r')
train_clean = np.asarray(data['train_clean'])
train_noisy = np.asarray(data['train_noisy'])
test_clean = np.asarray(data['test_clean'])
test_noisy = np.asarray(data['test_noisy'])

nframe_train_clean, input_dims = train_noisy.shape 
_, output_dims = train_clean.shape 
nframe_test_clean, _ = test_clean.shape 

print('Building a Tensor-Train model...')
class tt_model(nn.Module):
    def __init__(self, hidden_tensors, input_tensor, output_dim, tt_rank):
        super(tt_model, self).__init__()
        if len(hidden_tensors) != 3:
            raise ValueError('The depth of hidden layers should be 3!')
        if np.prod(input_tensor) != input_dims:
            raise ValueError('The product of input tensors must be equal to input dimension.')

        self.TTLinear1 = TTLinear(input_tensor, hidden_tensors[0], tt_rank=tt_rank)
        self.TTLinear2 = TTLinear(hidden_tensors[0], hidden_tensors[1], tt_rank=tt_rank)
        self.TTLinear3 = TTLinear(hidden_tensors[1], hidden_tensors[2], tt_rank=tt_rank)
        self.Linear4 = nn.Linear(np.prod(hidden_tensors[2]), 256)

    def forward(self, inputs):
        out = self.TTLinear1(inputs)
        out = self.TTLinear2(out)
        out = self.TTLinear3(out)
        out = self.Linear4(out)

        return out 

tt_rank = [1, 2, 2, 2, 1]
model = tt_model(args.hidden_tensors, args.input_tensor, output_dims, tt_rank).to(device)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

n_batch_train = int(nframe_train_clean / args.batch_size)
losses_train = []
losses_test = []
loss_mse = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

print('Training the TensorTrain Neural Network...')
for epoch in range(args.n_epochs):
    n_batch_train = int(nframe_train_clean / args.batch_size)
    n_batch_test = int(nframe_test_clean / args.batch_size)
    total_loss_train = 0
    total_loss_test = 0

    for idx in range(n_batch_train):
        model.zero_grad()
        train_clean_batch = torch.tensor(train_clean[idx*args.batch_size : (idx+1)*args.batch_size, :]).type(dtype)
        train_noisy_batch = torch.tensor(train_noisy[idx*args.batch_size : (idx+1)*args.batch_size, :]).type(dtype)
        
        #print("train_noisy_batch.shape = {}".format(train_noisy_batch.shape))
        probs = model(train_noisy_batch)
        loss = loss_mse(probs, train_clean_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss_train += loss.item()
        if idx % 500 == 0:
            tmp_loss = total_loss_train / (idx+1)
            print('Epoch{0}, idx={1}, Training Loss={2}'.format(epoch, idx, tmp_loss))
    total_loss_train /= n_batch_train 

    for idx in range(n_batch_test):
        eval_clean_batch = torch.tensor(test_clean[idx*args.batch_size : (idx+1)*args.batch_size, :]).type(dtype)
        eval_noisy_batch = torch.tensor(test_noisy[idx*args.batch_size : (idx+1)*args.batch_size, :]).type(dtype)
        probs = model(eval_noisy_batch)
        loss = loss_mse(probs, eval_clean_batch)
        total_loss_test += loss.item()
    total_loss_test /= n_batch_test

    print('Epoch: {}, total_loss: {}  |   test_loss: {}'.format(epoch, total_loss_train, total_loss_test))
    losses_train.append(total_loss_train)
    losses_test.append(total_loss_test)
    model_path = "exp/models/model" + str(epoch) + str(".h5")
    torch.save({ 
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_train': total_loss_train,
        'loss_test': total_loss_test
    }, model_path)


    
