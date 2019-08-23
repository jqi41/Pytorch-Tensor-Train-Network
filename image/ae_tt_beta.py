#!/usr/bin/env python3

import os
import sys
import numpy as np 
import argparse 
import time 

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from tc.tc_fc import TTLinear 

from torchvision import datasets, transforms

seed = 7
np.random.seed(seed)
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Training a tensor-train neural network...')
parser.add_argument('--batch_size', default=200, help='Mini-batch size')
parser.add_argument('--input_tensor', default=[7, 4, 7, 4])
parser.add_argument('--hidden_tensors', default=[[8, 4, 8, 4], [8, 4, 8, 4], [8, 4, 8, 8]])
parser.add_argument('--data_path', metavar='DIR', default='exp/feats/feats_m109_15dB.h5', help='Feature container in h5 format')
parser.add_argument('--save_model_path', default='model_tt.hdf5', help='The path to the saved model')
parser.add_argument('--n_epochs', default=5, help='The total number of epochs', type=int)
parser.add_argument('--save_model', default='tt_mnist.pt', help='Directory of the saved model path')
args = parser.parse_args()

train_data = datasets.MNIST(root = './data', train = True,
                            transform = transforms.ToTensor(), download = True)

test_data = datasets.MNIST(root = './data', train = False,
                           transform = transforms.ToTensor())

train_gen = torch.utils.data.DataLoader(dataset = train_data,
                                        batch_size = args.batch_size,
                                        shuffle = True)

test_gen = torch.utils.data.DataLoader(dataset = test_data,
                                       batch_size = args.batch_size, 
                                       shuffle = False)

class tt_autoencoder(nn.Module):
    def __init__(self, hidden_tensors, input_tensor, output_dim, tt_rank):
        super(tt_autoencoder, self).__init__()
        self.encoder1 = TTLinear(input_tensor, hidden_tensors[0], tt_rank=tt_rank)
            # TTLinear(hidden_tensors[0], hidden_tensors[1], tt_rank=tt_rank),
            # TTLinear(hidden_tensors[1], hidden_tensors[2], tt_rank=tt_rank))
        self.decoder1 = TTLinear(hidden_tensors[0],input_tensor, tt_rank=tt_rank)
            # TTLinear(hidden_tensors[2],hidden_tensors[1], tt_rank=tt_rank),
            # TTLinear(hidden_tensors[1],hidden_tensors[0], tt_rank=tt_rank),
            # TTLinear(hidden_tensors[0],input_tensor, tt_rank=tt_rank), nn.Tanh())

    def forward(self, inputs):
        ### Encoder layer
        out = self.encoder1(inputs)
        ### Decoder Layer with activation
        out = F.sigmoid(self.decoder1(out))
        return out



if __name__=='__main__':


    ### get data
    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # load the training and test datasets
    train_data = datasets.MNIST(root='data', train=True,
                                       download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False,
                                      download=True, transform=transform)
    # Create training and test dataloaders

    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 20

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
    tt_rank = [1, 2, 2, 2, 1]
    print('Building a Tensor-Train model...')
    model = tt_autoencoder(args.hidden_tensors, args.input_tensor, 10, tt_rank).to(device)
    
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    lr = 0.001
    # specify loss function
    criterion = nn.MSELoss()

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    # number of epochs to train the model
    n_epochs = 20

    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for data in train_loader:
            # _ stands in for labels, here
            images, _ = data
            # flatten images
            images = images.view(images.size(0), -1)
            images = images.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)
            # calculate the loss
            loss = criterion(outputs, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*images.size(0)
                
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            ))

    if (args.save_model):
        torch.save(model.state_dict(),"ae_tt.pt")

