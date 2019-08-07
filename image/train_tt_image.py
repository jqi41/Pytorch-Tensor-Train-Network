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

class tt_model(nn.Module):
    def __init__(self, hidden_tensors, input_tensor, output_dim, tt_rank):
        super(tt_model, self).__init__()
        if len(hidden_tensors) != 3:
            raise ValueError('The depth of hidden layers should be 3!')

        self.TTLinear1 = TTLinear(input_tensor, hidden_tensors[0], tt_rank=tt_rank)
        self.TTLinear2 = TTLinear(hidden_tensors[0], hidden_tensors[1], tt_rank=tt_rank)
        self.TTLinear3 = TTLinear(hidden_tensors[1], hidden_tensors[2], tt_rank=tt_rank)
        self.fc4 = nn.Linear(np.prod(hidden_tensors[2]), 10)

    def forward(self, inputs):
        out = self.TTLinear1(inputs)
        out = self.TTLinear2(out)
        out = self.TTLinear3(out)
        out = self.fc4(out)

        return F.log_softmax(out, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, 28*28)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)   # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


if __name__=='__main__':
    tt_rank = [1, 2, 2, 2, 1]
    print('Building a Tensor-Train model...')
    model = tt_model(args.hidden_tensors, args.input_tensor, 10, tt_rank).to(device)
    
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    loss_function = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = torch.optim.Adam( model.parameters(), lr=lr)


    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.n_epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_tt.pt")

