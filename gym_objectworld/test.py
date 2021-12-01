import gym
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
from solvers import value_iteration as V
from utilities import trajectory as T

env = gym.make('gym_objectworld:objectworld-gridworld-v0', size = 10, p_slip=0.01)
np.set_printoptions(suppress=True, precision=5)

ALPHA = 0.1
GAMMA = 0.95
EPISODE = 4000
EPSILON = 0.7
MIN_EPSILON = 0.1
DECAY_RATE = 0.9995
TAU=0.25

Q = np.ones((env.observation_space.n, env.action_space.n))

def choose_action(state):
	action = 0
	if np.random.rand() < EPSILON:
		action = env.action_space.sample()
	else:
		prob = softmax(Q[state,:]/TAU)
		action = np.random.choice(env.action_space.n, p=prob)
	return action

def policy_eval(env, Q):
	pol = np.zeros((env.observation_space.n, env.action_space.n))
	for s in range(env.observation_space.n):
		prob = softmax(Q[s,:]/TAU)
		pol[s,:] = prob
	return pol

reward = np.zeros((env.observation_space.n))
reward[-1] = 10.0

POL = V.find_policy(env, reward, GAMMA)

ts= list(T.generate_trajectories(1000, env, POL))

tot, tot1 = T.in_out_calc_it_all_about(env, ts)

print(tot1/1000)

x1 = np.arange(10)