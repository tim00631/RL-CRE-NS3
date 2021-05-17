import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

from collections import deque

from drqn import QNetwork, Memory
from multi_user_network_env import env_network

TIME_SLOTS = 100    # number of time-slots to run simulation
NUM_CHANNELS = 2    # Total number of channels
NUM_USERS = 3       # Total number of users
ATTEMPT_PROB = 1    # attempt probability of ALOHA based models

#It creates a one hot vector of a number as num with size as len
def one_hot(num, len):
    '''
    Parameters
    ----------
    num : int
        which position is set 1
    len : int
        length of vector
    '''
    assert num >=0 and num < len ,"error"
    vec = np.zeros([len],np.int32)
    vec[num] = 1
    return vec

#generates next-state from action and observation
def state_generator(action, obs):
    input_vector = []
    if action is None:
        print ('None')
        sys.exit()
    for user_i in range(action.size):
        input_vector_i = one_hot(action[user_i],NUM_CHANNELS+1)
        channel_alloc = obs[-1]
        input_vector_i = np.append(input_vector_i,channel_alloc)
        input_vector_i = np.append(input_vector_i,int(obs[user_i][0]))    #ACK
        input_vector.append(input_vector_i)
    return input_vector

if __name__ == "__main__":
    a = one_hot(0, 10)
    print(a)
    state_generator(a, )
