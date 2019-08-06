#!/usr/bin/env python3

import sys 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim

import tc
from tc.tc_cores import TensorTrain, _are_tt_cores_valid
import tc.tc_math 
from tc.tc_init import get_variables, glorot_initializer, he_initializer, lecun_initializer
import tc.tc_decomp

from torch.utils.data import TensorDataset, DataLoader 

activations = ['relu', 'sigmoid', 'tanh', 'softmax', 'linear']
inits = ['glorot', 'he', 'lecun']

class TTLinear(nn.Module):
    """A Tensor-Train linear layer is implemented by Pytorch in this calss. The 
    Tensor-Train linear layer replaces a fully-connected one by factorizing 
    it into a smaller 4D tensors and reducing the total number of parameters 
    of the dense layer. So the training and inference process can be significantly 
    sped up without a loss of model performance. That's particularly important 
    for the tasks of feature selection or dimension reduction where the information 
    redudancy always exists. """
    def __init__(self, inp_modes, out_modes, tt_rank, init='glorot',
                bias_init=0.1, activation='relu', **kwargs):
        super(TTLinear, self).__init__()
        self.ndims = len(inp_modes)
        self.inp_modes = inp_modes 
        self.out_modes = out_modes 

        self.tt_shape = [inp_modes, out_modes]
        self.activation = activation 
        self.init = init 
        self.tt_rank = tt_rank

        if self.init is 'glorot':
            initializer = glorot_initializer(self.tt_shape, tt_rank=tt_rank)
        elif self.init is 'he':
            initializer = he_initializer(self.tt_shape, tt_rank=tt_rank)
        elif self.init is 'lecun':
            initializer = lecun_initializer(self.tt_shape, tt_rank=tt_rank)
        else:
            raise ValueError('Unknown init "%s", only %s are supported'%(self.init, inits))

        self.W_cores = get_variables(initializer)
        _are_tt_cores_valid(self.W_cores, self.tt_shape, self.tt_rank)
        self.b = torch.nn.Parameter(torch.ones(1))

    
    def forward(self, x):
        TensorTrain_W = TensorTrain(self.W_cores, self.tt_shape, self.tt_rank)
        h = tc.tc_math.matmul(x, TensorTrain_W) + self.b
        if self.activation is not None:
            if self.activation in activations:
                if self.activation is 'sigmoid':
                    h = torch.sigmoid(h)
                elif self.activation is 'tanh':
                    h = torch.tanh(h)
                elif self.activation is 'relu':
                    h = torch.relu(h)
                elif self.activation is 'linear':
                    h = h 
            else:
                raise ValueError('Unknown activation "%s", only %s and None \
                    are supported'%(self.activation, activations))

        return h

    def extra_repr(self):
        return '(TTLayer): inp_modes={}, out_modes={}, mat_ranks={}'.format(
                list(self.inp_modes), \
                list(self.out_modes), \
                list(self.tt_rank))

    def nelements(self):
        """
        Returns the number of parameters of TTLayer.
        """
        num = 0
        for i in range(self.ndims):
            num += self.inp_modes[i] * self.out_modes[i] * \
                self.tt_rank[i] * self.tt_rank[i+1] 
        num = num + np.prod(self.out_modes)

        return num

    @property
    def name(self):
        return "TTLinear"


if __name__=="__main__":

    X = torch.tensor(torch.rand(784, 625).normal_()).type(torch.FloatTensor)
    B = torch.tensor(np.random.rand(625, 16)).type(torch.FloatTensor)
    Y = torch.sigmoid(torch.mm(X, B))
    loader = DataLoader(TensorDataset(X, Y), batch_size=50)

    inp_modes = [5, 5, 5, 5]
    out_modes = [2, 2, 2, 2]
    tt_rank = [1, 4, 4, 4, 1]
    tt_layer = TTLinear(inp_modes, out_modes, tt_rank, activation='relu')
    
#   optimizer = optim.SGD(tt_layer.parameters(), lr=0.05, momentum=0.5) 
    optimizer = optim.Adam(tt_layer.parameters(), lr=0.001)
    total_iters = 1
    pred = tt_layer(X)
    print(pred.shape)

    for iter in range(total_iters):
        for x_batch, y_batch in loader:
            y_pred = tt_layer(torch.tensor(x_batch).clone().detach())
            loss = torch.nn.functional.mse_loss(y_batch, y_pred)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(Y, tt_layer(X))
        print('Iter {}: {}'.format(iter, loss))

    print("num of params = {}".format(tt_layer.nelements()))
    
