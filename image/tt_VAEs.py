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

class tt_autoencoder(nn.Module):
    def __init__(self, hidden_tensors, input_tensor, output_dim, tt_rank):
        super(tt_autoencoder, self).__init__()
        self.encoder1 = nn.Sequential(TTLinear(input_tensor, hidden_tensors[0], tt_rank=tt_rank),
                                      TTLinear(hidden_tensors[0], hidden_tensors[1], tt_rank=tt_rank),
                                      TTLinear(hidden_tensors[1], hidden_tensors[2], tt_rank=tt_rank))
        self.decoder1 = nn.Sequential(TTLinear(hidden_tensors[2], hidden_tensors[1], tt_rank=tt_rank),
                                      TTLinear(hidden_tensors[1], hidden_tensors[0], tt_rank=tt_rank),
                                      TTLinear(hidden_tensors[0], input_tensor, tt_rank=tt_rank))
        self.lin = nn.Linear(np.prod(input_tensor), np.prod(input_tensor))
        self.model_name = "Tensor_Train_Autoencoder"
    def forward(self, inputs):
        ### Encoder layer
        out = self.encoder1(inputs)
        ### Decoder Layer with activation
        out = torch.sigmoid(self.lin(self.decoder1(out)))

        return out

class tt_VAE(nn.Module):
    def __init__(self, hidden_tensors, input_tensor, output_dim, tt_rank):
        super(tt_VAE, self).__init__()
        
        self.encoder1 = nn.Sequential(TTLinear(input_tensor, hidden_tensors[0], tt_rank=tt_rank),
                                      TTLinear(hidden_tensors[0], hidden_tensors[1], tt_rank=tt_rank))
        self.fc21 = TTLinear(hidden_tensors[1], hidden_tensors[2], tt_rank=tt_rank)
        self.fc22 = TTLinear(hidden_tensors[1], hidden_tensors[2], tt_rank=tt_rank)
        self.decoder1 = nn.Sequential(TTLinear(hidden_tensors[2],hidden_tensors[1], tt_rank=tt_rank),
                                      TTLinear(hidden_tensors[1],hidden_tensors[0], tt_rank=tt_rank),
                                      TTLinear(hidden_tensors[0],input_tensor, tt_rank=tt_rank))
        
    def encoder(self, inputs):
        out = self.encoder1(inputs)
        return self.fc21(out), self.fc32(out) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        out = F.sigmoid(self.decoder1(z))
        return out
    
    def forward(self, x):
        mu, log_var = self.encoder1(inputs)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var