import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

### makes environment deterministic
#from gym.envs.registration import register
#register(
#    id='FrozenLakeNotSlippery-v0',
#    entry_point='gym.envs.toy_text:FrozenLakeEnv',
#    kwargs={'map_name' : '4x4', 'is_slippery': False},
#    max_episode_steps=100,
#    reward_threshold=0.78, # optimum = .8196
#)

# Q table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameter
gamma = 0.95
alpha = 0.8
epsilon = 0.1

# Plotting Metrix
reward_list = []

episode_number = 75000
for i in range(1,episode_number):
    
    state = env.reset()
    
    reward_count = 0
    # for step in range(100):
    while True:
              
        # exploit vs explore to find action
        # %10 = explore, %90 exploit
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        # action process and take reward/ observation
        next_state, reward, done, _ = env.step(action)
        
        # Q learning function
        old_value = q_table[state,action] # old_value
        next_max = np.max(q_table[next_state]) # next_max
        
        next_value = (1-alpha)*old_value + alpha*(reward + gamma*next_max)
        
        # Q table update 
        q_table[state,action] = next_value
        
        # update state
        state = next_state
        
        reward_count += reward 
        
        if done:
            break
                
    if i%10 == 0:
        reward_list.append(reward_count)
        print("Episode: {}, reward {}".format(i,reward_count))
        
        
plt.plot(reward_list)
        
        

        

