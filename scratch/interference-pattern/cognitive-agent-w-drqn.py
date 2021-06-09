#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import tensorflow as tf
# tf_slim as slim
import numpy as np
np.random.seed(44)

import matplotlib as mpl
import matplotlib.pyplot as plt

import memory
import network

from tensorflow import keras
from ns3gym import ns3env

env = ns3env.Ns3Env(debug=False)

# env = gym.make('ns3-v0')
ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.n)

state_size = ob_space.shape[0]
action_size = ac_space.n

hidden_size = 128                       #Number of hidden neurons
learning_rate = 0.0001                  #learning rate
step_size=1+2+2                         #length of history sequence for each datapoint  in batch

optimizer = keras.optimizers.Nadam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, 
                                    epsilon=1e-08, name='Nadam')
loss = keras.losses.MeanSquaredError()
model = keras.Sequential()
model.add(keras.Input((step_size, state_size)))
model.add(keras.layers.LSTM(32, activation='tanh'))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(action_size))
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

total_episodes = 200
max_env_steps = 100
env._max_episode_steps = max_env_steps

epsilon = 1.0               # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.999

input_history = np.zeros((1, step_size, state_size))
time_history = []
rew_history = []

batch_size = 64
my_memory = memory.TimeMemory(state_size, batch_size=batch_size, step_size=step_size)

for e in range(total_episodes):

    state = env.reset()
    #print('Environment State: ', state)
    state = np.reshape(state, [1, state_size])
    rewardsum = 0
    for time in range(max_env_steps):

        # collect single-step state into multiple-step states
        input_history[0, time%step_size, :] = state[0]

        # Choose action
        if np.random.rand() < epsilon or time < step_size:
            action = np.random.randint(action_size)
        else:
            #print('predict: ', model.predict(input_history))
            action = np.argmax(model.predict(input_history))
            #print('agent select: ', action)

        # Step
        next_state, reward, done, _ = env.step(action)

        if done:
            print("episode: {}/{}, time: {}, rew: {}, eps: {:.2}"
                  .format(e, total_episodes, time, rewardsum, epsilon))
            break

        next_state = np.reshape(next_state, [1, state_size])
        input_history[0, (time+1)%step_size, :] = next_state[0]
        #print('input history: \n', input_history)
        # Train
        target = reward
        if not done and time > step_size:
            target = (reward + 0.95 * np.amax(model.predict(input_history)))

        my_memory.push(state, action, next_state, target)

        if len(my_memory) > batch_size:
            # For feed-forward net (MLP)
            #state, action, target = my_memory.sample() # state : batch*n_channnel, action: batch*1, target: batch*1
            
            # For LSTM
            # state : batch*time_step*state_size (n_channel)
            # action: batch*time_step*1
            # target: batch*time_step*1
            state, action, target = my_memory.sample() 

            target_f = model.predict(state)
            #print('action: ', action.shape)
            #print('target_f: ', target_f.shape)
            target_f[:, action] = target
            #model.fit(state, target_f, epochs=1, verbose=0)
            
            model.fit(state, target_f, batch_size=16, epochs=128, verbose=0)

        state = next_state
        rewardsum += reward
        if epsilon > epsilon_min: epsilon *= epsilon_decay
        
    time_history.append(time)
    rew_history.append(rewardsum)

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
plt.plot(range(len(rew_history)), rew_history, label='Reward', marker="", linestyle="-")#, color='k')
plt.xlabel('Episode')
plt.ylabel('Time')
plt.legend(prop={'size': 12})

plt.savefig('learning.png', bbox_inches='tight')
plt.show()
