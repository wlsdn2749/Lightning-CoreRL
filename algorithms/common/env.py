import numpy as np

def run_environment(env, Q=None, n_iterations=10, render=True):
    """
    This function runs gym environment using q-value or random policy
    
    Args:
        env: openAI Gym environment
        Q: q-value function
        n_iterations: number of episodes to run
        render: visualization flag
    Return:
        n_success: total number of successes playing n_iterations
        avg_reward: average reward of n_iterations
    
    """
    # intialize success and total reward
    n_success = 0
    total_reward = 0
    
    # loop over number of episodes to play
    for i in range(n_iterations):        
        # reset the enviorment every time when playing a new episode
        state, state_info = env.reset()
        
        done = False
        while not done:    
            # Visualize
            if render:
                env.render()
                                
            # if Q is not given, take a random action, else take an action by the policy
            if Q is None:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            # take the next step
            next_state, reward,  done, trancated, info = env.step(action)
            
            # change the state
            state = next_state

            # check if game is over with positive reward
            if done:
                n_success += 1
                env.render()
                print("({}/{}) Success".format(i+1, n_iterations))

            # calculate total reward
            total_reward += reward
                                                        
    # calculate average reward
    average_reward = total_reward / n_iterations
    
    env.close()
    
    return n_success, average_reward

