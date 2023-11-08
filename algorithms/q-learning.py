import gymnasium as gym
import numpy as np
import random
import time

env_name = "FrozenLake-v1"
env = gym.make(env_name, is_slippery=False)

# algorithm parameters
alpha = 0.5
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.001

# initialize Q(s,a) for all s, a except that Q(terminal, .) = 0

Q = np.zeros((env.observation_space.n, env.action_space.n))

num_episodes = 1000

def eplison_greedy():
    random_number = random.random()
    
    if random_number < epsilon: # random
        action = env.action_space.sample() 
    else: # best action
        action = np.argmax(Q[state, :])
    
    return action

n_success = 0
total_reward = 0
    
for i in range(num_episodes):
    state, _ = env.reset() # state, info

    done = False
    episode_reward = 0
    
    while not done:
        # epsilon_greedy()
        action = eplison_greedy()
        
        # take an action A, observe R, S'
        next_state, reward, done, truncated, info = env.step(action)
        
        # update Q(s,a) using TD
        td_target = reward + gamma * np.max(Q[next_state, :])
        td_error = td_target - Q[state, action]
        Q[state, action] = Q[state, action] + alpha * td_error
        # S <- S'
        state = next_state
        episode_reward += reward
        
        if done:
            n_success += 1
            env.render()
            print(f"({i+1}/{num_episodes}) Success")

        # calculate total reward
        total_reward += reward
        
                                                        
    # calculate average reward
    average_reward = total_reward / num_episodes
    
    env.close()
        
    # until S is terminal
    
    # decay epsilon
    epsilon = max(epsilon - epsilon_decay ,  0)
        
env.close()

print("number of success: {}".format(n_success))
print("average reward: {}".format(average_reward))
