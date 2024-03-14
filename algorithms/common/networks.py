"""Series of networks used"""
import math
from typing import Tuple
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

class CNN(nn.Module):
    """
    Simple MLP network
    
    Args:
        input_shape : observation shape of the environment
        n_actions : number of discrete actions available in the enviornment
    """
    
    def __init__(self, input_shape, n_actions):
        super(CNN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        self.head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape) -> int:
        """
        Calculate the output size of last conv layer
        
        Args:
            shape : input dimensions
            
        Returns:
            size of the conv output
        """
        conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))
    
    def forward(self, input_x) -> Tensor:
        """
        Forward pass through network
        
        Args:
            x: input to network
            
        Returns:
            output of network
        """
        
        conv_out = self.conv(input_x).view(input_x.size()[0], -1)
        return self.head(conv_out)
    
class MLP(nn.Module):
    """
    Simple MLP network

    Args:
        input_shape: observation shape of the environment
        n_actions: number of discrete actions available in the environment
        hidden_size: size of hidden layers
    """

    def __init__(self, input_shape: Tuple, n_actions: int, hidden_size: int = 128):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, input_x):
        """
        Forward pass through network

        Args:
            x: input to network

        Returns:
            output of network
        """
        return self.net(input_x.float())

    