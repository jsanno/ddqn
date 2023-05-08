#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Code adapted from https://github.com/rlcode/reinforcement-learning/tree/master/2-cartpole

import random
import numpy as np
from collections import deque
import os

# Force CPU use (GPU not worth it as DNN is small)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.models import Sequential
from keras.layers import Dense
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size, max_steps, episodes):
        # Define state and action space sizes
        self.state_size = state_size
        self.action_size = action_size
        # Hyper-parameters for the Double-DQN architecture
        self.discount_factor = np.exp(np.log(0.8) / max_steps)  # Discount factor for Bellman equation (chosen as a function of max number of steps!)
        self.learning_rate = 1e-3  # Learning rate for ADAM optimizer
        self.epsilon = 1.0  # Initial epsilon value (for epsilon greedy policy)
        self.epsilon_min = 1e-4  # Minimal epsilon value (for epsilon greedy policy)
        self.epsilon_decay = 2*(1 - self.epsilon_min) / episodes  # Epsilon decay (for epsilon greedy policy, explore half of the episodes!
        #self.epsilon_decay = 3 * (1 - self.epsilon_min) / episodes 
        self.batch_size = 512  # Batch size for replay
        self.train_start = 1000  # Adds a delay, for the memory to have data before starting the training
        # Create a replay memory using deque
        self.memory = deque(maxlen=max_steps * 1000)  # To store up to 1000 longest episodes
        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()  # The target model is a NN used to increase stability
        # Initialize target model
        self.update_target_model()
        self.loss = []

    # NN input is the state, output is the estimated Q value for each action
    def build_model(self):
        # We build a model with 3 layers
        model = Sequential()
        model.add(Dense(72, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        # model.summary() # Uncomment to see the model summary provided by Keras
        # Compile the model: use Mean Squared Error as loss function, ADAM as optimizer
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    # Function to update the target model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Epsilon greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def update_epsilon(self):
        if self.epsilon - self.epsilon_decay > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    # Save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) < self.train_start:
            return  # Start training only when there are some samples in the memory
        # Pick samples randomly from replay memory (with batch_size)
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)
        # Preprocess the batch by storing the data in different vectors
        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])
        # Obtain the targets for the NN training phase
        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target)  # Use the target network HERE for further stability

        for i in range(self.batch_size):
            # Get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.discount_factor * (target_val[i][a])
        # Fit the model!
        h = self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)
        self.loss.append(h.history['loss'])

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)