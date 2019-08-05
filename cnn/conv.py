#!/usr/bin/env python3 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from normalize import build_norm

def param(nnet, Mb=True):
    """
    Return the number of parameters in nnet
    """
    nelems = sum([param.nelement() for param in nnet.parameters()])

    return nelems / 10**6 if Mb else nelems


class Conv1D(nn.Conv1d):
    """
    1D convoluational layer
    """
    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accepts 2/3D tensor as an input.".format(self.__name__))
        x = super().forward(x if x.dim()==3 else torch.unsqueeze(x, 1))

        if squeeze:
            x = torch.squeeze(x)

        return x 


class Conv2D(nn.Conv2d):
    """
    2D convolutional layer
    """
    def __init__(self, *args, **kwargs):
        super(Conv2D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x H x W or N x C x H x W
        """
        if x.dim() not in [3, 4]:
            raise RuntimeError("{} accepts 3/4D tensor as an input".format(self.__name__))
        x = super().forward(x if x.dim()==4 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)

        return x 
    

class ConvTrans1D(nn.ConvTranspose1d):
    """
    1D transpose convolutional layer.
    """
    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accepts 2/3D tensor as the input.".format(self.__name__))
        
        x = super().forward(x if x.dim() == 3 else torch.squeeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)

        return x 


class ConvTrans2D(nn.ConvTranspose2d):
    """
    2D transpose convolutional layer. 
    """
    def __init__(self, *args, **kwargs):
        super(ConvTrans2D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x H x W or N x C x H x W
        """
        if x.dim() not in [3, 4]:
            raise RuntimeError("{} accepts 3/4D tensor as the input".format(self.__name__))

        x = super().forward(x if x.dim()==4 else torch.squeeze(x, 1))
        if squeeze:
            x = torch.squeeze()

        return x 


class DWConv2D(nn.Module):
    """
    DepthWise Convolutional layer 2D.
    """
    def __init__(self, 
                in_channels=256,    # number of input channels
                out_channels=256,   # number of output channels
                stride=1,           # stride steps
                dilation=1,         # dilated rate
                reduction=1,
                norm="BN2"):       # factor for bottleneck feature map
        super(DWConv2D, self).__init__()
        self.expansion = 1 / float(reduction)
        self.in_channels = in_channels
        self.conv_channels = conv_channels = int(self.expansion * out_channels)
        self.out_channels = out_channels 

        self.conv1 = Conv2D(
            in_channels = in_channels, 
            out_channels = conv_channels,
            kernel_size = 1, 
            bias = False)
        self.bn1 = build_norm(norm, conv_channels)
        self.depth = Conv2D(conv_channels, conv_channels, kernel_size=3, padding=1,
                                stride=1, bias=False, groups=conv_channels)
        self.bn2 = build_norm(norm, conv_channels)
        self.conv3 = Conv2D(conv_channels, out_channels, kernel_size=1, bias=False, stride=stride)
        self.bn3 = build_norm(norm, out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                build_norm(norm, out_channels)
            )
    
    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once.')
        (_, _, int_h, int_w), (_, _, out_h, out_w) = self.int_nchw, self.out_nchw
        flops = int_h * int_w * self.conv_channels * self.in_channels + out_h * out_w * self.conv_channels * self.out_channels
        flops += out_h * out_w * self.conv_channels * 9  # depth-wise convolution
        if len(self.shortcut) > 0:
            flops += self.in_channels * self.out_channels * out_h * out_w

        return flops

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        self.int_nchw = out.size()
        out = self.bn2(self.depth(out))
        out = self.bn3(self.conv3(out))
        self.out_nchw = out.size()
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class DWConv1D(nn.Module):
    """
    DepthWise Convolutional layer 1D.
    """
    def __init__(self, 
                in_channels=256,    # number of input channels
                out_channels=256,   # number of output channels
                stride=1,           # stride steps
                dilation=1,         # dilated rate
                reduction=1,
                norm="BN1"):       # factor for bottleneck feature map
        super(DWConv1D, self).__init__()
        self.expansion = 1 / float(reduction)
        self.in_channels = in_channels
        self.conv_channels = conv_channels = int(self.expansion * out_channels)
        self.out_channels = out_channels 

        self.conv1 = Conv1D(
            in_channels = in_channels, 
            out_channels = conv_channels,
            kernel_size = 1, 
            bias = False)
        self.bn1 = build_norm(norm, conv_channels)
        self.depth = Conv1D(conv_channels, conv_channels, kernel_size=3, padding=1,
                                stride=1, bias=False, groups=conv_channels)
        self.bn2 = build_norm(norm, conv_channels)
        self.conv3 = Conv1D(conv_channels, out_channels, kernel_size=1, bias=False, stride=stride)
        self.bn3 = build_norm(norm, out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                Conv1D(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                build_norm(norm, out_channels)
            )
    
    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once.')
        (_, _, int_h), (_, _, out_h) = self.int_nchw, self.out_nchw
        flops = int_h * self.conv_channels * self.in_channels + out_h * self.conv_channels * self.out_channels
        flops += out_h * self.conv_channels * 9  # depth-wise convolution
        if len(self.shortcut) > 0:
            flops += self.in_channels * self.out_channels * out_h

        return flops

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        self.int_nchw = out.size()
        out = self.bn2(self.depth(out))
        out = self.bn3(self.conv3(out))
        self.out_nchw = out.size()
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out


if __name__ == "__main__":

    x = torch.randn(10, 256, 50, 50)
    nnet = DWConv2D()
    print("ConvTasNet #param: {:.2f}".format(param(nnet)))
    x = nnet(x)
    s1 = x[0]
    print(s1.shape)
