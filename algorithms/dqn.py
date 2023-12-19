import pytorch_lightning as pl
import argparse
import wandb
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from typing import Tuple, List, Dict
from collections import OrderedDict
from common import cli
from common import wrappers
from common.networks import CNN

from common.agents import ValueAgent
from common.experience import ExperienceSource, RLDataset
from common.memory import ReplayBuffer

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

class DQNLightning(pl.LightningModule):
    """ Basic DQN model"""
    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__()
        self._hparams = hparams
        
        device = torch.device("cuda:0" if self._hparams.gpus > 0 else "cpu")
        
        self.env = wrappers.make_env(self._hparams.env) # common
        self.env.seed(123)
        
        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        
        self.net = None
        self.target_net = None
        self.buffer = None
        self.build_networks()
        
        self.agent = ValueAgent(
            self.net,
            self.n_actions,
            eps_start = hparams.eps_start,
            eps_end = hparams.eps_end,
            eps_frames = hparams.eps_last_frame
        )
        
        self.source = ExperienceSource(self.env, self.agent, device)
        
        self.total_reward = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.episode_steps = 0
        self.total_episode_steps = 0
        self.reward_list = []
        for _ in range(100):
            self.reward_list.append(-21) # ? why -21..
        self.avg_reward = -21
        
        
    def populate(self, warm_start: int) -> None:
        """Populates the buffer with initial experience"""
        if warm_start > 0:
            for _ in range(warm_start):
                self.source.agent.epsilon = 1.0
                exp, _, _ = self.source.step()
                self.buffer.append(exp)
        
    def build_networks(self) -> None:
        """Initializes the DQN train and target networks"""
        self.net = CNN(self.obs_shape, self.n_actions)
        self.target_net = CNN(self.obs_shape, self.n_actions)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values 
        of each actions as an output
        
        Args:
            x: environment state
            
        Returns:
            q values
        """
        output = self.net(x)
        return output
    
    def loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer
        
        Args:
            batch: current mini batch of replay data
            
        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch
        
        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()
            
        expected_state_action_values = next_state_values * self._hparams.gamma + rewards
        
        return nn.MSELoss()(state_action_values, expected_state_action_values)
        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> OrderedDict:
        """
        
        Carries out a single step through the environemnt to update the replay buffer.
        Then calculates loss based on the minibatch received
        
        Args:
            batch: current mini batch of replay data
            _: batch number, not used
            
        Returns:
            Training loss and log metrics
        
        """
        self.agent.update_epsilon(self.global_step)
        
        # step through environment with agent and add to buffer
        exp, reward, done = self.source.step()
        self.buffer.append(exp)
        
        self.episode_reward += reward
        self.episode_steps += 1
        
        # calculates training loss
        loss = self.loss(batch)
        
        if self.trainer.use_dp or self.trainer_use_ddp2:
            loss = loss.unsqueeze(0) # 병렬처리
            
        if done:
            self.total_reward = self.episode_reward
            self.reward_list.append(self.total_reward)
            self.avg_reward = sum(self.reward_list[-100:]) / 100
            self.episode_count += 1
            self.episode_reward = 0
            self.total_episode_steps = self.episode_steps
            self.episode_steps = 0
            
        # Soft update of target network
        if self.global_step % self._hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.next.state_dict())
            
        log = {'total_reward' : torch.tensor(self.total_reward).to(self.device),
               'avg_reward': torch.tensor(self.avg_reward),
               'train_loss': loss,
               'episode_steps': torch.tensor(self.total_episode_steps)
               }
        
        status = {'steps': torch.tensor(self.global_step).to(self.device),
                  'avg_reward': torch.tensor(self.avg_reward),
                  'total_reward': torch.tensor(self.total_reward).to(self.device),
                  'episodes': self.episode_count,
                  'episode_steps': self.episode_steps,
                  'epsilon': self.agent.epsilon
                  }
        
        return OrderedDict({'loss': loss,
                            'avg_reward': torch.tensor(self.avg_reward),
                            'log': log,
                            'progress_bar': status})

        
    def test_step(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Evaluate the agent for 10 episodes"""
        self.agent.epsilon = 0.0 # Greedy Search
        test_reward = self.source.run_episode()
        
        return {'test_reward': test_reward}
    
    def test_epoch_end(self, outputs) -> Dict[str, torch.Tensor]:
        """Log the avg of the test results"""
        rewards = [ x['test_reward'] for x in outputs ]
        avg_reward = sum(rewards) / len(rewards)
        logs = {'avg_test_reward': avg_reward}
        return {'avg_test_reward': avg_reward, 'log': logs}
    
    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self._hparams.lr)
        return [optimizer]
    
    def _dataloader(self) -> DataLoader:
        """Initialize the Replay buffer dataset used for retrieving experiences"""
        self.buffer = ReplayBuffer(self._hparams.replay_size)
        self.populate(self._hparams.warm_start_size)
        
        dataset = RLDataset(self.buffer, self._hparams.episode_length)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self._hparams.batch_size,
                                )
        return dataloader
    
    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self._dataloader()

    def test_dataloader(self) -> DataLoader:
        """Get test loader"""
        return self._dataloader()
        
    
    @staticmethod
    def add_model_specific_args(arg_parser) -> argparse.ArgumentParser:
        """
        Adds arguments for DQN model
        
        Note: these params are fine tuned for Pong env
        
        Args:
            parent
        """
        arg_parser.add_argument("--sync_rate", type=int, default=1000,
                                help="how many frames do we update the target network")
        arg_parser.add_argument("--replay_size", type=int, default=100000,
                                help="capacity of the replay buffer")
        arg_parser.add_argument("--warm_start_size", type=int, default=10000,
                                help="how many samples do we use to fill our buffer at the start of training")
        arg_parser.add_argument("--eps_last_frame", type=int, default=150000,
                                help="what frame should epsilon stop decaying")
        arg_parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
        arg_parser.add_argument("--eps_end", type=float, default=0.02, help="final value of epsilon")
        arg_parser.add_argument("--warm_start_steps", type=int, default=10000,
                                help="max episode reward in the environment")

        return arg_parser
    
if __name__ == "__main__":

    wandb.init(
        project='basic_dqn'
    )
    
    wandb_logger = WandbLogger(
        project='basic_dqn',
        log_model='all'
    )
    
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = cli.add_base_args(parent=parent_parser)
    parent_parser = DQNLightning.add_model_specific_args(parent_parser)
    
    args = parent_parser.parse_args()
    
    # args.env = 'CartPole-v1' # use DQN
    wandb.config.update(args)
    
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    model = DQNLightning(args)
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='avg_reward', 
        mode='max'
    )
    
    trainer = pl.Trainer(
        gpus=args.gpus, # how many gpu you use?
        distribued_backend=args.backend,
        max_steps=args.max_steps,
        max_epochs=args.max_steps,
        val_check_interval=1000,
        profiler=True,
        checkpoint_callback=checkpoint_callback,
        logger=wandb_logger        
    )
    
    trainer.fit(model)
    trainer.test()
    
    wandb.finish()
