from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam

class Memory:
    def __init__(self,memory_size,batch_size,time_steps):
        self.buffer = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.time_steps = time_steps
    def store(self, state, action, reward, next_state):
        self.buffer.append([state, action, reward, next_state])

    def sample(self):
        sample = random.sample(self.buffer,self.batch_size)
        states, actions, rewards, next_states = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(self.batch_size, self.time_steps, -1)
        next_states = np.array(next_states).reshape(
            self.batch_size, self.time_steps, -1
        )
        return states, actions, rewards, next_states
    def size(self):
        return len(self.buffer)

class DRQN:
    def __init__(self, state_dim, action_dim, epsilon, epsilon_decay, epsilon_min, time_steps, learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.time_steps = time_steps
        self.model = self.nn_model()
        self.model.compile(optimizer=Adam(learning_rate),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy']
        )
    def nn_model(self):
        return tf.keras.Sequential(
            [
                Input((self.time_steps, self.state_dim)),
                LSTM(32, activation="tanh"),
                Dense(16, activation="relu"),
                Dense(self.action_dim),
            ]
        )

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state):
        state = np.reshape(state, [1, self.time_steps, self.state_dim])
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        q_value = self.predict(state)[0]
        # random select an action from action space
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        # select an action based on max Q_value
        else:
            return np.argmax(q_value)
    def train(self, states, targets):
        # TODO
        self.model.fit(states, targets)
        pass