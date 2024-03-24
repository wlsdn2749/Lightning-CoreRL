import argparse

def add_base_args(parent) -> argparse.ArgumentParser:
    """
    Adds arguments for DQN model

    Note: these params are fine tuned for Pong env
        
    Args:
        lr: learning rate, default 1e-4 0.0001
        alpha: alpha, default 0.5 
        episode_length: default 500
        gamma: default 0.99
    """
    arg_parser = argparse.ArgumentParser(parents=[parent])

    
    # arg_parser.add_argument("--algo", type=str, default="dqn", help="algorithm to use for training")
    arg_parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    arg_parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    arg_parser.add_argument("--alpha", type=float, default=0.1, help="alpha")
    arg_parser.add_argument("--env", type=str, default="CartPole-v1", help="gym environment tag")
    arg_parser.add_argument("--render_mode", type=str, default="rgb_array", help="gym render_mode | human, rgb_array, ansl, None")
    arg_parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    arg_parser.add_argument("--episode_length", type=int, default=500, help="max length of an episode")
    # arg_parser.add_argument("--max_episode_reward", type=int, default=18,
    #                         help="max episode reward in the environment")
    arg_parser.add_argument("--max_steps", type=int, default=500000,
                            help="max steps to train the agent")
    # arg_parser.add_argument("--n_steps", type=int, default=4,
    #                         help="how many steps to unroll for each update")
    arg_parser.add_argument("--devices", type=int, default=1,
                            help="number of devices to use for training")
    arg_parser.add_argument("--seed", type=int, default=3721,
                            help="seed for training run")
    arg_parser.add_argument("--strategy", type=str, default="auto",
                            help="distributed backend strategy to be used by lightning")
    return arg_parser
    