import gym
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from solvers import value_iteration as V
from utilities import trajectory as T
import plot as P
from gym import spaces
env = gym.make('gym_objectworld:objectworld-gridworld-v0')

print(env.action_space)

print(env.act(env.reset()))

print(isinstance(env.action_space, spaces.Discrete))
exit()

def Random_game():

	env.reset()

	for t in range(100):

		env.render()

		action=env.action_space.sample()

		next_state, reward, done, _ = env.step(action)

		print(t, next_state, reward, done, action)

Random_game()