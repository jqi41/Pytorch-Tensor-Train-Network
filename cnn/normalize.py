#!/usr/bin/env python3 

import math 
import torch
import torch.nn as nn 

import torch.nn.functional as F 

class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel-wise layer normalization.
    """
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accepts 3D tensor as input".format(self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C 
        x = torch.transpose(x, 1, 2)

        return x 


class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization.
    """
    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim 
        self.elementwise_affine = elementwise_affine 
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(dim, 1))
            self.gamma = nn.Parameter(torch.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accepts 3D tensor as an input".format(self.__name__))
        # N x 1 x 1 
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean)**2, (1, 2), keepdim=True)

        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta 
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)

        return x 

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)

def build_norm(norm, dim):
    """
    Build a normalized layer. 
    LN costs more memory than BN.
    """
    if norm not in ["cLN", "gLN", "BN1", "BN2"]:
        raise RuntimeError("Unsupervised normalized layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN1":
        return nn.BatchNorm1d(dim)
    elif norm == "BN2":
        return nn.BatchNorm2d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)

    

