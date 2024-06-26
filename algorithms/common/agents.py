"""Module containing basic type of agents used by the various algorithms"""

from random import randint

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Agent:
    """Basic agent that always returns 0"""
    def __init__(self, net: nn.Module):
        self.net = net
        
    def __call__(self, state: torch.Tensor, device: str) -> int:
        """
        Using the given network, decide what action to carry
        
        Args:
            state: current state of the enviornment
            device: device used for current batch
            
        Returns:
            action
        """
        return 0
    
class ValueAgent(Agent):
    """Value based agent that returns an action based on the Q values from the networks"""
    def __init__(self, 
                 net: nn.Module, 
                 action_space: int,
                 eps_start: float = 1.0,
                 eps_end: float = 0.2,
                 eps_frames: float = 1000):
        super().__init__(net)
        self.action_space = action_space
        self.eps_start = eps_start
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_frames = eps_frames
        
    def __call__(self, state: torch.Tensor, device: str) -> int:
        """
        Takes in the current state and returns the action based on the agents policy
        
        Args:
            state: current state of the enviornment
            device: the device used for the current batch
            
        Returns:
            action defined by policy
        """
        # Epsilon greedy
        if np.random.random() < self.epsilon:
            action = self.get_random_action()
        else: 
            action = self.get_action(state, device)    
            
        return action
    
    def get_random_action(self) -> int:
        """returns a random action"""
        action = randint(0, self.action_space - 1)

        return action    

    def get_action(self, state: torch.Tensor, device: torch.device):
        """
            Returns 
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor([state])
            
        if device.type != 'cpu':
            state = state.cuda(device)
            
        q_values = self.net(state)
        
        # _, action = torch.max(q_values, dim=1)
        # return int(action.item())
        
        _, action = torch.max(q_values, dim=0) # tuple(max, max_indices)
        return int(action.item())
    
    def update_epsilon(self, step: int) -> None:
        """
        Updates the epsilon value based on the current step
        
        Args:
            step: current global step
        """
        self.epsilon = max(self.eps_end, self.eps_start - (step + 1) / self.eps_frames)