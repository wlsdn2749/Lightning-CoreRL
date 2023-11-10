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



import numpy as np
import os
# import matplotlib.pyplot as plt
# from time import sleep

class Dyna_env:
    def __init__(self):
        self.states = [State(i) for i in range(54)] # 6x9
        self.states[28].accessible = False
        self.states[29].accessible = False
        self.states[30].accessible = False
        self.states[31].accessible = False
        self.states[32].accessible = False
        self.states[33].accessible = False
        self.states[34].accessible = False
        self.states[35].accessible = False
        self.states[8].reward = 1 # 오른쪽 가장 끝
        self.player_pos = 48 # 시작 자리
        self.done = False
        self.accessible_states = [state.id for state in self.states if state.accessible == True]

    def reset(self):
        self.player_pos = 48
        self.done = False
        return self.player_pos

    def step(self, action):
        if action == 1: # up
            if self.states[self.player_pos].id <= 8:
                return self.player_pos, self.states[self.player_pos].reward, self.done
            self.player_pos_change(self.player_pos, -9)
            self.check_done()
            return self.player_pos, self.states[self.player_pos].reward, self.done

        if action == 0: # left
            if self.states[self.player_pos].id in [0, 9, 18, 27, 36, 45]:
                return self.player_pos, self.states[self.player_pos].reward, self.done
            self.player_pos_change(self.player_pos, -1)
            self.check_done()
            return self.player_pos, self.states[self.player_pos].reward, self.done 

        if action == 2: # right
            if self.states[self.player_pos].id in [8, 17, 26, 35, 44, 53]:
                return self.player_pos, self.states[self.player_pos].reward, self.done
            self.player_pos_change(self.player_pos, 1)
            self.check_done()
            return self.player_pos, self.states[self.player_pos].reward, self.done

        if action == 3: # down
            if self.states[self.player_pos].id >= 45:
                return self.player_pos, self.states[self.player_pos].reward, self.done
            self.player_pos_change(self.player_pos, 9)
            self.check_done()
            return self.player_pos, self.states[self.player_pos].reward, self.done

    def check_done(self):
        if self.player_pos == 8:
            self.done = True

    def player_pos_change(self, pos, value):
        if pos + value in self.accessible_states:
            self.player_pos += value

    # def render(self):
    #     row1 = ['-', '-', '-', '-', '-', '-', '-', 'X', 'G']
    #     row2 = ['-', '-', 'X', '-', '-', '-', '-', 'X', '-']
    #     row3 = ['S', '-', 'X', '-', '-', '-', '-', 'X', '-']
    #     row4 = ['-', '-', 'X', '-', '-', '-', '-', '-', '-']
    #     row5 = ['-', '-', '-', '-', '-', 'X', '-', '-', '-']
    #     row6 = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
    #     rows = [row1, row2, row3, row4, row5, row6]
    #     loc = self.player_pos
    #     row_num = loc//9
    #     col_num = loc%9
    #     rows[row_num][col_num] = 'o'
    #     print(rows[0])
    #     print(rows[1])
    #     print(rows[2])
    #     print(rows[3])
    #     print(rows[4])
    #     print(rows[5])

    def clear(self):
        os.system("clear")


class State:
    def __init__(self, id):
        self.id = id 
        self.reward = 0  
        self.accessible = True


class Agent:
    def __init__(self, env):
        self.env = env
        self.Q = {}
        self.model = {}
        self.env.accessible_states.pop(8) 
        for s in self.env.accessible_states:
            self.Q[s] = []
            self.model[s] = []
            for a in range(4):
                self.Q[s] += [np.random.random()]
                self.model[s] += [np.random.random()]
                
    def sample_action(self, s):
        if np.random.random() < 0.1:
            return np.random.choice([0, 1, 2, 3])
        return np.argmax(self.Q[s])

    # def train(self, episode_nums, env, alpha, gamma, eval_epochs, render=True):
    #     total_reward = 0
    #     episode_num = 0
    #     running_average = []
    #     while episode_num < episode_nums:
    #         s = env.player_pos
    #         a = self.sample_action(s)
    #         p_s = s
    #         StateMemory.append(s)
    #         if s not in ActionMemory:
    #             ActionMemory[s] = []
    #         ActionMemory[s] += [a]
    #         s, r, done = env.step(a)
    #         if render == True:
    #             env.clear()
    #             env.render()
    #             print("Cumulative Reward this episode: %.2f"%total_reward)
    #         else:
    #             print("Please wait, training is in progess.")
    #             env.clear()
    #             print("Please wait, training is in progess..")
    #             env.clear()
    #             print("Please wait, training is in progess...")
    #             env.clear()
    #         total_reward += r
    #         self.Q[p_s][a] += alpha * (r + (gamma * np.max(self.Q[s])) - self.Q[p_s][a])
    #         self.model[p_s][a] = (r, s)
    #         if done:
    #             s = env.reset()
    #             env.clear()
    #             episode_num += 1
    #             # print("Attained total reward at {}th episode: {}".format(episode_num, total_reward))
    #             # sleep(1.5)
    #             running_average.append(total_reward)
    #             total_reward = 0
    #         for n in range(eval_epochs):
    #             s1 = np.random.choice(StateMemory)
    #             a1 = np.random.choice(ActionMemory[s1])
    #             r1, s_p1 = self.model[s1][a1]
    #             self.Q[s1][a1] += alpha * (r1 + (gamma * np.max(self.Q[s_p1])) - self.Q[s1][a1])
    #     return running_average


    # def print_policy(self):
    #     best_actions = {}
    #     for s in self.env.accessible_states:
    #         a = np.argmax(self.Q[s])
    #         if a == 1:
    #             a = '^'
    #         if a == 0:
    #             a = '<'
    #         if a == 2:
    #             a = '>'
    #         if a == 3:
    #             a = 'v'
    #         best_actions[s] = a
    #     self.env.clear()
    #     print("----------------BEST POLICY----------------")
    #     row1 = ['-', '-', '-', '-', '-', '-', '-', 'X', 'G']
    #     row2 = ['-', '-', 'X', '-', '-', '-', '-', 'X', '-']
    #     row3 = ['S', '-', 'X', '-', '-', '-', '-', 'X', '-']
    #     row4 = ['-', '-', 'X', '-', '-', '-', '-', '-', '-']
    #     row5 = ['-', '-', '-', '-', '-', 'X', '-', '-', '-']
    #     row6 = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
    #     rows = [row1, row2, row3, row4, row5, row6]
    #     for s in self.env.accessible_states:
    #         row_num = s//9
    #         col_num = s%9
    #         rows[row_num][col_num] = best_actions[s]
    #     rows[0][8] = 'G'
    #     print(rows[0])
    #     print(rows[1])
    #     print(rows[2])
    #     print(rows[3])
    #     print(rows[4])
    #     print(rows[5])
    #     print("-------------------------------------------")

# def play_human(env):
#     s = env.reset()
#     done = False
#     total_reward = 0
#     env.render()
#     while not done:
#         a = input("Enter action: ")
#         if a == 'w':
#             a = 1
#         if a == 'a':
#             a = 0
#         if a == 's':
#             a = 3
#         if a == 'd':
#             a = 2
#         s, r, done = env.step(a)
#         env.clear()
#         env.render()
#         total_reward += r
#     print("Total reward attained is: ", total_reward)

# env = Environment()
# agent1 = Agent(env)
# agent2 = Agent(env)
# agent3 = Agent(env)
# agent4 = Agent(env)
# running_average1 = agent1.train(50, env, 0.1, 0.95, 0)
# StateMemory = []
# ActionMemory = {}
# running_average2 = agent2.train(50, env, 0.1, 0.95, 5)
# StateMemory = []
# ActionMemory = {}
# running_average3 = agent3.train(50, env, 0.1, 0.95, 50)
# StateMemory = []
# ActionMemory = {}
# running_average4 = agent4.train(50, env, 0.1, 0.95, 100)
# agent1.print_policy()
# plt.plot(running_average1, label="Planning 0 steps")
# plt.plot(running_average2, label="Planning 5 steps")
# plt.plot(running_average3, label="Planning 50 steps")
# plt.plot(running_average4, label="Planning 100 steps")
# plt.legend()
# plt.title("Running Average")
# plt.show()