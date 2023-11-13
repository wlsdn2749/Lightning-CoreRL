import gymnasium as gym
import numpy as np
import random
import argparse
import wandb

from common.env import Dyna_env
from common import cli
import time


env = Dyna_env()

# algorithm parameters


def epsilon_greedy(state, epsilon, Q):
    random_number = random.random()
    
    if random_number < epsilon: # random
        action = np.random.choice([0,1,2,3]) # sample action space
    else: # best action
        action = np.argmax(Q[state])
    
    return action

def print_policy(Q, env):
    best_actions = {}
    for s in env.accessible_states:
        a = np.argmax(Q[s])
        if a == 1:
            a = '^'
        if a == 0:
            a = '<'
        if a == 2:
            a = '>'
        if a == 3:
            a = 'v'
        best_actions[s] = a
    # env.clear()
    print("----------------BEST POLICY----------------")
    row1 = ['-', '-', '-', '-', '-', '-', '-', '-', 'G']
    row2 = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
    row3 = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
    row4 = ['-', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']
    row5 = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
    row6 = ['-', '-', '-', 'S', '-', '-', '-', '-', '-']
    rows = [row1, row2, row3, row4, row5, row6]
    for s in env.accessible_states:
        row_num = s//9
        col_num = s%9
        rows[row_num][col_num] = best_actions[s]
    rows[0][8] = 'G' # End
    rows[5][4] = 'S' # Start
    print(rows[0])
    print(rows[1])
    print(rows[2])
    print(rows[3])
    print(rows[4])
    print(rows[5])
    print("-------------------------------------------")

# Initalize Q(s, a) and Model(s, a) for all s in S and a in A(s)

# alpha = 0.5
# gamma = 0.9
# epsilon = 1.0
# epsilon_decay = 0.001

# num_episodes = 10
# num_eval_epochs = 100


def dyna_q(args):
    lr = args.lr
    alpha = args.alpha
    epsilon = args.eps
    gamma = args.gamma
    epsilon_decay = args.eps_decay
    num_eval_epochs = 50
    episode_length = args.episode_length
    
    Q = {}
    model = {}

    # env.accessible_states.pop(8) #  ??? ternimal state는 더이상 할 수 있는 action이 없음

    for s in env.accessible_states: # 방문가능한 state
        Q[s] = []
        model[s] = [] 
        for a in range(4):
            Q[s] += [np.random.random()] # maybe change into np.zeros?
            model[s] += [np.random.random()]
            

    # train 할때마다 reset 해주어야함

    running_average = []

    state_memory = []
    action_memory = {}
    for i in range(episode_length):
        
        state = env.reset() # player pos
        
        done = False
        total_reward = 0
        total_step = 0
        # episode_num = 0
        # running_average = []
        start = time.time()
        while not done:
            # S <- current (nonterminal) state
            state = env.player_pos
            # A <- epsilon-greedy(S,Q)
            
            action = epsilon_greedy(state, epsilon, Q)
            
            state_memory.append(state) # 방문한 상태 기록
            if state not in action_memory:
                action_memory[state] = []
            action_memory[state] += [action] # 액션 기록
            
            # Take action A; observe resultant reward, R, and state, S'

            next_state, reward, done = env.step(action)
            if done:
                print(next_state, reward, done) # debug
                
            # print(next_state, reward, done) # debug
            
            # Q(S,A) <- Q(S,A) + alpha[R + lambda * max Q(S',a) - Q(S,A)]
            td_target = reward + gamma * np.max(Q[next_state])
            td_error = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + alpha * td_error
            
            total_reward += reward
            total_step += 1
            # --- Q-learning end ---
            
            # --- Q-planning start ---
            # Model(S, A) <- R, S'
            model[state][action] = (reward, next_state)   
                
            if done:
                state = env.reset()
                # env.clear()
                # episode_num += 1
                end = time.time()
                print("Attained total reward at {}th episode: {:.2f}, total_step : {}".format(i+1, total_reward, total_step))
                print(f"total time is ", {end-start})
                
                # sleep(1.5)
                wandb.log({"total rewards": total_reward})
                running_average.append(total_reward)
                total_reward = 0
            

            for j in range(num_eval_epochs):
                # S <- random previously observed state
                # A <- random action previously taken in S
                s1 = np.random.choice(state_memory)
                a1 = np.random.choice(action_memory[s1])
                
                # R, S' <- Model(S,A)
                r1, next_s1 = model[s1][a1]
                
                # Q Update
                td_target = r1 + gamma * np.max(Q[next_s1])
                td_error = td_target - Q[s1][a1]
                Q[s1][a1] = Q[s1][a1] + alpha * td_error
                
        epsilon = max(epsilon - epsilon_decay, 0.1) # epsilon decay
        
    # wandb.log(running_average)        
    print(running_average)
    print_policy(Q, env)


        
if __name__ == "__main__":
    '''
    
    Args:
        lr: learning rate, default 1e-4 0.0001
        alpha: alpha, default 0.1
        episode_length: default 500
        gamma: default 0.99
        eps : epsilon, default 1.0
        eps_decay : epslion decay default : 1e-3 0.001
    '''
    wandb.init(
        project = "dyna-q"
    )
    
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = cli.add_base_args(parent=parent_parser)

    parent_parser.add_argument("--eps", type=float, default=1.0, help="epsilon")
    parent_parser.add_argument("--eps_decay", type=float, default=1e-3, help="epslion decay")
    
    
    args = parent_parser.parse_args()
    args.episode_length = 10
    args.gamma = 0.9
    args.alpha = 0.5
    
    wandb.config.update(args)
    average_rewards = dyna_q(args) # reward
    
    # parent_parser = DQNLightning.add_model_specific_args(parent_parser)
    # parent_parser = VPGLightning.add_model_specific_args(parent_parser)