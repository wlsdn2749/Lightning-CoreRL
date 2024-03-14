""" Series of memory buffers used"""

# Named tuple for storing experience steps gathered in training
import collections
from typing import Tuple, List, Union
from collections import deque, namedtuple

import numpy as np

Experience = namedtuple(
    'Experience',
    field_names=['state', 'action', 'reward', 'done', 'new_state']
)


class Buffer:
    """
    Basic Buffer for storing a single experience at a time
    
    Args:
        capacity: size of the buffer
    
    """
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)
        
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer
        
        Args:
            experience: tuple (state, action reward, done, new_state)
        """
        self.buffer.append(experience)
        
    def sample(self, *args) -> Union[Tuple, List[Tuple]]:
        """
        returns everything in the buffer so far it is then reset
        
        Returns:
            a batch of tuple np arrays of state, action, reward, done, next_state

        """
        states, actions, rewards, dones, next_states =\
            zip(*[self.buffer[idx] for idx in range(self.__len__())])
            
        self.buffer.clear()
        
        return (np.array(states), 
                np.array(actions), 
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool_),
                np.array(next_states))
        
class ReplayBuffer(Buffer):
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    
    Args:
        capacity: size of the buffer
    """
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Takes a sample of the buffer
        Args:
            batch_size: current batch_size
            
            
        Returns:
            a batch of tuple np arrays of state, action reward, done, next_state
        """
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states =\
            zip(*[self.buffer[idx] for idx in indices])
            
        
        return (np.array(states), 
                np.array(actions), 
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool_),
                np.array(next_states))
        
    