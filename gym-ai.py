#!/usr/bin/python
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import random

import numpy as np
# fix random seed for reproducibility
np.random.seed(7)

import gym
env = gym.make('CartPole-v0')


# https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
# Well, there are two ways to approach this. One is to pass the action in as an input, and the other is to have one output per action (which is possible because we have relatively few actions). The latter option is much faster, as clearly explained by the DeepMind paper:

# https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288

ALPHA = 0.9
LEARNING_RATE = 0.001

DISCOUNT_FACTOR = 0.8
BATCH_SIZE = 20

class DQNSolver():
  def __init__(self, observation_space_size, action_space_size):
    self.model = Sequential()
    self.model.add(Dense(24, input_dim=observation_space_size, activation='relu'))
    self.model.add(Dense(24, activation="relu"))
    self.model.add(Dense(action_space_size, activation='linear'))
    self.model.compile(loss="mean_squared_error", optimizer=Adam(lr=LEARNING_RATE))
    self.memory = []

  def get_action(self, state):
    candidate_actions = self.model.predict(state).flatten()

    # Since our model outputs 0-1 (sigmoid) for every action, we can treat them as a probability
    for i in range(len(candidate_actions)):
      if candidate_actions[i] > random.random():
        return i

  def remember(self, state, action, reward, state_next, done):
    self.memory.append((state, action, reward, state_next, done))

  # Experience replay is needed because it makes the target more stable
  # i.e. as we are learning, our "correct" label changes, so we train on a batch to mitigate the changes
  def experience_replay(self):
    if len(self.memory) < BATCH_SIZE:
      return
  
    batch = random.sample(self.memory, BATCH_SIZE)

    # Q(s, a)_t+1 = Q(s, a)_t + a * (R(s, a) + Y * maxQ(s', a')) 
    for state, action, reward, state_next, done in batch:
      # Q(s, a)_t
      Q_s = self.model.predict(state)
      Q_sa = Q_s[action]

      Q_prime_s = self.model.predict(state_next)

      Q_sa_next = Q_sa + ALPHA * (reward + DISCOUNT_FACTOR * Q_prime_s[np.argmax(Q_prime_s)])

      # When done, we don't use the formula above but we just use the final reward
      if done:
        Q_sa_next = reward

      self.model.fit(state, Q_sa_next)

observation_space = env.observation_space.shape[0]
dqn_solver = DQNSolver(observation_space, env.action_space.n)

while True:
    state = env.reset()
    state = np.reshape(state, [1, observation_space])
      
    while True:
      env.render()

      action = dqn_solver.get_action(state)
      state_next, reward, done, info = env.step(action)
      state_next = np.reshape(state_next, [1, observation_space])

      reward if not done else -reward


      dqn_solver.remember(state, action, reward, state_next, done)
      dqn_solver.experience_replay()
      
      state = state_next
      
      if done:
        break
    
    # Backprop
    model.fit()

    prevObservation = observation


    # Input = observation
    # Output = reward per action


    backwards_pass()
    input()