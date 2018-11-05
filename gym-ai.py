#!/usr/bin/python
import gym
env = gym.make('BipedalWalker-v2')
env.reset()

# Best video
# https://www.youtube.com/watch?v=JgvyzIkgxF0
# And the blog post to accompany it
# http://karpathy.github.io/2016/05/31/rl/

# Policy network -> the actual network that takes input and outputs an action (actor?)
# Credit assignment problem -> if you get reward now, what was the action that gave you reward? ex. is it 20 frames ago?
# Monotonic -> always increasing or decreasing

# Policy gradient -> apply gradient with whatever the network chose to encourage it to take the same action again
# At the end of a training game (ex. play pong until you win or lose), we reverse the gradient if we lost, since it was a bad choice

# Ex.
# Play 100 games of pong "policy rollouts"


for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action