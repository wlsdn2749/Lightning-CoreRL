"""
Set of wrapper functions for gym environments taken from
https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/
"""
import collections
import numpy as np
import cv2
import gymnasium as gym
import gymnasium.spaces as gymspaces
import torch


class ToTensor(gym.Wrapper):
    """For environments where the user need to press FIRE for the game to start."""

    def __init__(self, env=None):
        super(ToTensor, self).__init__(env)

    def step(self, action):
        """Take 1 step and cast to tensor"""
        state, reward, terminated, truncated, info = self.env.step(action)
        
        return torch.tensor(state), torch.tensor(reward), terminated, truncated, info # v0.26

    def reset(self):
        """reset the env and cast to tensor"""
        observation, info = self.env.reset()
        return torch.tensor(observation), info


class FireResetEnv(gym.Wrapper):
    """For environments where the user need to press FIRE for the game to start."""

    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        """Take 1 step"""
        return self.env.step(action)

    def reset(self):
        """reset the env"""
        self.env.reset()
        
        obs, reward, terminated, truncated, info = self.env.step(1)
        done = terminated or truncated
        if done:
            self.env.reset()
            
        obs, reward, terminated, truncated, info = self.env.step(2)
        done = terminated or truncated
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame"""

    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        """take 1 step"""
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs, _ = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    """preprocessing images from env"""

    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gymspaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        """preprocess the obs"""
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        """image preprocessing, formats to 84x84"""
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(
                np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(
                np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(
            img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    """converts image to pytorch format"""

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gymspaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    @staticmethod
    def observation(observation):
        """convert observation"""
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """scales the pixels"""

    @staticmethod
    def observation(obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    """"Wrapper for image stacking"""

    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        self.buffer = None
        old_space = env.observation_space
        self.observation_space = gymspaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        """reset env"""
        self.buffer = np.zeros_like(
            self.observation_space.low, dtype=self.dtype)
        
        obs, info = self.env.reset()
        return self.observation(obs)

    def observation(self, observation):
        """convert observation"""
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class DataAugmentation(gym.ObservationWrapper):
    """
    Carries out basic data augmentation on the env observations

    - ToTensor
    - GrayScale
    - RandomCrop
    """

    def __init__(self, env=None):
        super().__init__(env)
        self.observation_space = gymspaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        """preprocess the obs"""
        return ProcessFrame84.process(obs)


def make_env(env_name):
    """Convert environment with wrappers"""
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ToTensor(env)
    return ScaledFloatFrame(env)