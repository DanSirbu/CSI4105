#!/usr/bin/python
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import os
import random
import numpy as np
# fix random seed for reproducibility
np.random.seed(7)

import gym

#ENVIRONMENT = 'CartPole-v0'
#ENVIRONMENT = 'LunarLander-v2'
#ENVIRONMENT = 'Acrobot-v1'
#ENVIRONMENT = 'MountainCar-v0'
ENVIRONMENT = 'MsPacman-ram-v0'


SAVED_MODEL_FILE = ENVIRONMENT + "-model.h5"
STATE_FILE = ENVIRONMENT + "-state.save"

env = gym.make(ENVIRONMENT)

# https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
# Well, there are two ways to approach this. One is to pass the action in as an input, and the other is to have one output per action (which is possible because we have relatively few actions). The latter option is much faster, as clearly explained by the DeepMind paper:

# https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288


#Hyperparameters
ALPHA = 0.9
ALPHA_INVERSE = (1 - ALPHA)

LEARNING_RATE = 0.001

EXPLORATION_RATE_INIT = 1.0
EXPLORATION_RATE_MIN = 0.01
EXPLORATION_RATE_DECAY = 0.95

DISCOUNT_FACTOR = 0.95
BATCH_SIZE = 20

PLOT_SCORE_ON_LOAD = False

class DQNSolver():
  def __init__(self, observation_space_size, action_space_size):
    if os.path.isfile(SAVED_MODEL_FILE):
      print("Using saved model: ", SAVED_MODEL_FILE)
      self.model = load_model(SAVED_MODEL_FILE)
    else:
      self.model = Sequential()
      self.model.add(Dense(24, input_dim=observation_space_size, activation='relu'))
      self.model.add(Dense(24, activation="relu"))
      self.model.add(Dense(action_space_size, activation='linear'))
      self.model.compile(loss="mean_squared_error", optimizer=Adam(lr=LEARNING_RATE))
    
    self.memory = []
    self.action_space = action_space_size
    self.exploration_rate = EXPLORATION_RATE_INIT

  def save(self):
    self.model.save(SAVED_MODEL_FILE)

  #Returns the next action to take
  def get_action(self, state):
    # We need to also explore the action space in order to build the Q network
    if random.random() < self.exploration_rate:
      return random.randrange(0, self.action_space)
    
    candidate_actions = self.model.predict(state)

    return np.argmax(candidate_actions[0])

  #Store state,next state pairs for training
  def remember(self, state, action, reward, state_next, done):
    self.memory.append((state, action, reward, state_next, done))

  # Experience replay is needed because it makes the target more stable
  # i.e. as we are learning, our "correct" label changes, so we train on a batch to mitigate the changes
  def experience_replay(self):
    if len(self.memory) < BATCH_SIZE:   #don't q learn before batch_size number of examples
      return
  
    batch = random.sample(self.memory, BATCH_SIZE)

    # Q(s, a)_t+1 = (1-a)*Q(s, a)_t + a * (R(s, a) + Y * maxQ(s', a'))
    for state, action, reward, state_next, done in batch:
      # Q(s, a)_t
      Q_s = self.model.predict(state)
      Q_sa = Q_s[0][action]       #Q_sa reward of the action taken

      Q_prime_s = self.model.predict(state_next)    #reward of next state actions

      Q_sa_next = ALPHA_INVERSE * Q_sa + ALPHA * (reward + DISCOUNT_FACTOR * Q_prime_s[0][np.argmax(Q_prime_s)])

      # When done, we don't use the formula above but we just use the final reward
      if done:
        Q_sa_next = reward

      
      Q_s[0][action] = Q_sa_next
      self.model.fit(state, Q_s, verbose=0)

    if self.exploration_rate > EXPLORATION_RATE_MIN:
      self.exploration_rate *= EXPLORATION_RATE_DECAY

#bookkeeping: plot scores
def plotScore():
    plt.plot(score)
    plt.xticks(list(range(len(score))))
    plt.show()

observation_space = env.observation_space.shape[0]
dqn_solver = DQNSolver(observation_space, env.action_space.n)

episode = 0
score = []

if os.path.isfile(STATE_FILE):
  with open(STATE_FILE, 'rb') as f:
    loadedData = pickle.load(f)

    dqn_solver.exploration_rate = loadedData[0]
    score = loadedData[-1]
    episode = len(score)
    if PLOT_SCORE_ON_LOAD:
      plotScore()

#loop 50 episodes (small reasonable number)
for i in range(50):
    #initializing environment
    state = env.reset()
    state = np.reshape(state, [1, observation_space])

    print("Episode ", episode)
    episode += 1

    #Loop to be ran every frame(game), will loop till game end
    step = 0    # number of frames taken for game to end
    game_reward = 0   #cumulutive reward total for scorekeeping
    while True:
      env.render()

      action = dqn_solver.get_action(state)
      state_next, reward, done, info = env.step(action)
      state_next = np.reshape(state_next, [1, observation_space])

      #reward if not done else -reward # Wut? Why use -reward?


      dqn_solver.remember(state, action, reward, state_next, done)
      dqn_solver.experience_replay()
      
      state = state_next
      step += 1
      game_reward += reward

      if done:
        score.append(game_reward / step)
        print("Game score: ", score[-1])
        break

    # Post game
    if episode % 10 == 0:
      dqn_solver.save()   #save model every 10 episodes
      with open(STATE_FILE, 'wb') as f:
        pickle.dump([dqn_solver.exploration_rate, score], f)
    
plotScore()