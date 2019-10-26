import gym

env = gym.make("Taxi-v2").env

env.render() # show

"""
blue = passenger
purple = destination
yellow/red = empty taxi
green = full taxi
RGBY = location for destination and passanger
"""

env.reset() # reset env and return  random initial state

# %% 

print("State space: ",env.observation_space) # 500
print("Action space: ", env.action_space) # 6

# taxi row, taxi column, passenger index, destination
state = env.encode(3,1,2,3)
print("State number: ",state)

env.s = state
env.render()

# %%
"""
Actions: 
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
"""
# probability, next_state, 
env.P[331]

# %%

total_reward_list = []
# episode
for j in range(5):
    env.reset()
    time_step = 0
    total_reward = 0
    list_visualize = []
    while True:
        
        time_step += 1
        
        # choose action
        action = env.action_space.sample()
        
        # perform action and get reward
        state, reward, done, _ =  env.step(action) # state = next state
        
        # total reward
        total_reward += reward
    
        # visualize
        list_visualize.append({"frame": env,
                               "state": state, "action": action, "reward":reward,
                               "Total Reward": total_reward})
        
        # env.render()
        
        if done:
            total_reward_list.append(total_reward)
            break
 
# %%
import time       
for i, frame in enumerate(list_visualize):
    print(frame["frame"].render())
    print("Timestep: ", i + 1)
    print("State: ", frame["state"])
    print("action: ", frame["action"])
    print("reward: ", frame["reward"])
    print("Total Reward: ", frame["Total Reward"])
    # time.sleep(2)


    

    
    









































