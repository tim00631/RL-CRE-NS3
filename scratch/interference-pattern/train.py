#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import numpy as np
np.random.seed(44)

import matplotlib as mpl
import matplotlib.pyplot as plt

from ns3gym import ns3env
from DQN_model import Memory, DRQN
from tensorflow import keras
env = ns3env.Ns3Env(debug=False)

# env = gym.make('ns3-v0')
ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.n)

state_size = ob_space.shape[0]
action_size = ac_space.n

# hidden_size = 128           # Number of hidden neurons
learning_rate = 0.001       # learning rate
time_steps=1+2+2            # length of history sequence for each datapoint in batch
total_episodes = 500
max_env_steps = 100
env._max_episode_steps = max_env_steps

epsilon = 0.9              # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.9999
memory_size = 5000
batch_size = 64
gamma_r = 0.95
replace_target_iter = 32

is_training = True
if is_training :
    # eval_model, target model
    model = DRQN(state_size, action_size, epsilon, epsilon_decay, epsilon_min, time_steps, learning_rate)
    target_model = DRQN(state_size, action_size, epsilon, epsilon_decay, epsilon_min, time_steps, learning_rate)
else:
    model = keras.models.load_model('model/drqn_model')
    target_model = keras.models.load_model('model/drqn_target')
# Replay Memory
memory = Memory(memory_size, batch_size, time_steps)


state_history = np.zeros([time_steps, state_size])
time_history = []
reward_history = []

def replay_experience(model, target_model, memory):
    states, actions, rewards, next_states = memory.sample()
    targets_value = model.predict(states)
    next_q_values = target_model.predict(next_states).max(axis=1)
    targets_value[:memory.batch_size, actions] = rewards + gamma_r * next_q_values
    model.train(states, targets_value)

def update_state_history(state):
    global state_history
    state_history = np.roll(state_history, -1,axis=0)
    state_history[-1] = state

def update_target_weight():
    weights = model.model.get_weights()
    target_model.model.set_weights(weights)

for e in range(total_episodes):
    reward_sum = 0
    state_history = np.zeros([time_steps, state_size])
    state = env.reset()
    #print('Observation Space: ', obs.shape)
    state = np.reshape(state,[1, state_size])
    update_state_history(state)

    for time in range(max_env_steps):
        action = model.get_action(state_history,is_training)
        # Step
        next_state, reward, done, _ = env.step(action)

        if done:
            print("episode: {}/{}, time: {}, rew: {}, eps: {:.2}"
                  .format(e, total_episodes, time, reward_sum, model.epsilon))
            break

        next_state = np.reshape(next_state, [1, state_size])
        prev_state_history = state_history
        update_state_history(next_state)
        memory.store(prev_state_history, action, reward, state_history)
        reward_sum += reward
        
        if is_training:
            if memory.size() > batch_size:
                replay_experience(model, target_model, memory)
            if time % replace_target_iter == 0:  
                update_target_weight()
    
    time_history.append(time)
    reward_history.append(reward_sum)

if is_training:
    model.model.save('model/drqn_model')
    target_model.model.save('model/drqn_target')
    
env.close()
#for n in range(2 ** s_size):
#    state = [n >> i & 1 for i in range(0, 2)]
#    state = np.reshape(state, [1, s_size])
#    print("state " + str(state) 
#        + " -> prediction " + str(model.predict(state)[0])
#        )

#print(model.get_config())
#print(model.to_json())
#print(model.get_weights())

print("Plot Learning Performance")
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(10,4))
plt.grid(True, linestyle='--')
plt.title('Learning Performance')
plt.plot(range(len(time_history)), time_history, label='Steps', marker="^", linestyle=":")#, color='red')
plt.plot(range(len(reward_history)), reward_history, label='Reward', marker="", linestyle="-")#, color='k')
plt.xlabel('Episode')
plt.ylabel('Time')
plt.legend(prop={'size': 12})

plt.savefig('learning.png', bbox_inches='tight')
plt.show()
