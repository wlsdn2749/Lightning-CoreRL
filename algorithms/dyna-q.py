import gymnasium as gym
import numpy as np
import random

from common.env import Dyna_env
import time
env = Dyna_env()

# algorithm parameters
alpha = 0.5
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.001

num_episodes = 1000
num_eval_epochs = 100

def eplison_greedy(state):
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
    env.clear()
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
    rows[0][8] = 'G'
    print(rows[0])
    print(rows[1])
    print(rows[2])
    print(rows[3])
    print(rows[4])
    print(rows[5])
    print("-------------------------------------------")

# Initalize Q(s, a) and Model(s, a) for all s in S and a in A(s)

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
state_memory = []
action_memory = {}

running_average = []

for i in range(num_episodes):
    state = env.reset() # player pos
    
    done = False
    total_reward = 0
    # episode_num = 0
    # running_average = []
    start = time.time()
    while not done:
        # S <- current (nonterminal) state
        state = env.player_pos
        # A <- epsilon-greedy(S,Q)
        
        action = eplison_greedy(state)
        
        state_memory.append(state) # 방문한 상태 기록
        if state not in action_memory:
            action_memory[state] = []
        action_memory[state] += [action] # 액션 기록
        
        # Take action A; observe resultant reward, R, and state, S'

        next_state, reward, done = env.step(action)
        if done:
            print(next_state, reward, done) # debug
        
        # Q(S,A) <- Q(S,A) + alpha[R + lambda * max Q(S',a) - Q(S,A)]
        td_target = reward + gamma * np.max(Q[next_state])
        td_error = td_target - Q[state][action]
        Q[state][action] = Q[state][action] + alpha * td_error
        
        total_reward += reward
        # --- Q-learning end ---
        
        # --- Q-planning start ---
        # Model(S, A) <- R, S'
        model[state][action] = (reward, next_state)   
             
        if done:
            state = env.reset()
            env.clear()
            # episode_num += 1
            end = time.time()
            print("Attained total reward at {}th episode: {}".format(i+1, total_reward))
            print(f"total time is ", {end-start})
            
            # sleep(1.5)
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
            
    
            
print(running_average)
print_policy(Q, env)
    
    
    
        

        
