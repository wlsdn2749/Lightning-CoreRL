""" Experience sources to be used as datasets for Lightning DataLoaders"""

from collections import deque
from typing import List, Tuple

import numpy as np
from gymnasium import Env
from torch.utils.data import IterableDataset
from common.agents import Agent
from common.memory import Experience, Buffer

class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the Experience Buffer
    which will be updated with new experiences during training
    
    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """
    
    def __init__(self, buffer: Buffer, sample_size: int = 1) -> None:
        self.buffer = buffer
        self.sample_size = sample_size
        
    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        
        for idx, _ in enumerate(dones):
            yield (states[idx], 
                   actions[idx], 
                   rewards[idx], 
                   dones[idx], 
                   new_states[idx])
            
    def __getitem__(self, item):
        """Not used"""
        return None

class ExperienceSource:
    """
    Basic single step experience source
    
    Args:
        env: Environment that is being used
        agent: Agent being used to make decisions
    """
    
    def __init__(self, env: Env, agent: Agent, device):
        self.env = env
        self.agent = agent
        self.state = self.env.reset()
        self.device = device
    
    def _reset(self) -> None:
        """resets the env and state"""
        self.state = self.env.reset()
        
        
    def step(self) -> Tuple[Experience, float, bool]:
        """ Takes a single step through the environment"""
        action = self.agent(self.state, self.device)
        new_state, reward, _, done, _, = self.env.step(action)
        experience = Experience(state=self.state,
                                action=action,
                                reward=reward,
                                new_state=new_state,
                                done=done) 
        
        self.state = new_state
        
        if done:
            self.state = self.env.reset()
            
        return experience, reward, done
    
    def run_episode(self) -> float:
        """Carries out a single episode and returns the total reward. This is used for testing..."""
        
        done = False
        total_reward = 0
        
        while not done:
            _, reward, done = self.step()
            total_reward += reward
            
        return total_reward
        
            
        
        
        
    