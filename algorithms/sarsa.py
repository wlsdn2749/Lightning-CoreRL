import numpy as np
import gymnasium as gym
import pickle
from common.env import run_environment
from common.utils import save_obj, load_obj
import time

def sarsa(env, alpha=0.8, gamma=0.98, epsilon=1.0, epsilon_decay=0.001, num_episodes=1000, save_path="q_tables", render=False):
    """
    SARSA algorithm
        Q(s,a) <- Q(s,a) + alpha*(reward + gamma*Q(s',a) - Q(s,a))
    
    Args:
        env: openAI Gym environment
        alpha: learning rate (0~1)
        gamma: discount factor
        epsilon: probability to select a random action
        epsilon_decay: parameters to decay epsilon
        n_iterations: number of episodes to run    
        render: visualize or not
    Return:
        Q: action-value function
    """    

    # Algorithm parameters ----------------------------------------------------------
    #   set function arguments
    # alpha = 0.8               # learning rate
    # gamma = 0.98              # discount factor
    # epsilon = 1.0             # starting value of epsilon-greedy probability
    # epsilon_decay = 0.001     # epsilon decay rate
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Initialize Q(s,a) for all s, a except that Q(terminal,.)=0 --------------------
    # -----------------------------------------------------------------pp--------------
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))


    n_success = 0 # success rate
    # -------------------------------------------------------------------------------
    # Loop for each episide ---------------------------------------------------------
    # -------------------------------------------------------------------------------
    
    for i in range(num_episodes):    
        # Initialize S --------------------------------------------------------------
        state, state_info = env.reset()  # gymnasium has two return arguments
        # visualize
        if render:
            env.render()

        # Choose A from S using policy derived from Q with e-greedy -----------------
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        
        # Loop for each step of episode ----------------------------------------------
        done = False
        while not done:        
            # Take an action A, observe R, S' ---------------------------------------- 
            next_state, reward, done, trancated, info = env.step(action)
            
            # Choose A' from S' using policy derived from Q with e-greedy ------------
            if np.random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state, :])
            
            # Q(s,a) <- Q(s,a) + alpha*(reward + gamma*Q(s',a') - Q(s,a)) -------------
            td_target = reward + gamma * Q[next_state, next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] = Q[state, action] + alpha * td_error
                    
            # S <- S', A -> A' -------------------------------------------------------
            state = next_state
            action = next_action

            # visualize
            # if render:
            #     env.render()
            #     time.sleep(0.1)

            # until S is terminal ------------------------------------------
            if done:
                n_success += 1
                env.render()
                print(f"({i+1}/{num_episodes}) Success")

        # decay epsilon
        epsilon = max(epsilon - epsilon_decay, 0)
        
    #save Q function (q_table)
    if save_path:
        save_obj(Q, 'cliffwalking_sarsa')
    
    return Q

if __name__ == "__main__":
    np.random.seed(1)
    
    env_name = 'CliffWalking-v0'
    env = gym.make(env_name, render_mode='rgb_array')
    env.reset(seed = 0)
    
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.01
    num_episodes = 1000
    
    Q = sarsa(
        env, 
        alpha=alpha, 
        gamma=gamma, 
        epsilon=epsilon,  
        epsilon_decay=epsilon_decay, 
        num_episodes=num_episodes,
        render=True 
    )
    
    print(Q)
    
    Q = load_obj('cliffwalking_sarsa')
    n_success, average_reward = run_environment(env, Q, n_iterations=5, render=True)
    print("number of success: {}".format(n_success))
    print("average reward: {}".format(average_reward))
    
    