"""
Value iteration policy and value function approximations

Authored by B. I. Lyons and Johannes Vallikivi

Adapted from
https://github.com/qzed/irl-maxent
https://github.com/MatthewJA/Inverse-Reinforcement-Learning
"""

import numpy as np
from itertools import product
from scipy.special import logsumexp as sp_lse
from gym_objectworld.envs import GridWorldEnv
from gym_objectworld.envs import ObjectWorldEnv
np.random.seed(0)
def convert_transition_array(env):

	# Can I change this?

	# If ObjectWorld (walls, and dictionary set up slightly differently, need to change Gridworld later)

	if isinstance(env, ObjectWorldEnv):
		transition_array = np.zeros(((env.grid_size-2)**2, env.action_space.n, (env.grid_size-2)**2))
		for (y_i, x_i) in product(range(1, env.grid_size-1), range(1, env.grid_size-1)):
			for a in range(len(env.actions)):
				summed = 0
				for (y_k, x_k) in product(range(1, env.grid_size-1), range(1, env.grid_size-1)):
					transition_array[(y_i-1)*(env.grid_size-2) + (x_i-1)][a][(y_k-1)*(env.grid_size-2) + (x_k-1)] = env.P[(y_i, x_i)][a][(y_k, x_k)]
		return transition_array

	elif isinstance(env, GridWorldEnv):
		transition_array = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))

		for i in range(env.observation_space.n):
			for a in range(env.action_space.n):
				for j in range(len(env.P[i][a])):
					dest = env.P[i][a][j][1]
					transition_array[i][a][dest] += env.P[i][a][j][0]

		return transition_array

def value_iteration(env, reward, discount, eps=1e-3):
	# Convert dictionary to array, need to find a more efficient method
	t = convert_transition_array(env)
	n_states, n_actions, _ = t.shape
	v = np.zeros(n_states)
	q = np.zeros((n_states, n_actions))	
	delta = np.inf
	while delta>eps:
		v_old = v.copy()
		for s in range(n_states):
			for a in range(n_actions):
				# print(reward+discount*v_old)
				q[s,a] = np.dot(t[s,a,:], reward+discount*v_old)
			v[s] = sp_lse(q[s,:])
		delta = np.max(np.abs(v_old - v))
	return v, t

def find_policy(env, reward, discount, eps=1e-3):

    v, t = value_iteration(env, reward, discount)

    n_states, n_actions, _ = t.shape

    Q = np.zeros((n_states, n_actions))

    y = [np.matrix(t[:,a,:]) for a in range(n_actions)]
    for a in range(n_actions):
        Q[:,a] = y[a].dot(reward + discount*v)
    Q -= Q.max(axis=1).reshape(n_states, 1)
    Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape(n_states, 1)

    return Q

def policy_eval(policy, reward, env, nS, nA, discount_factor=0.9, theta=0.001):
	"""
	Policy Evaluation.
	"""
	transition_probabilities = convert_transition_array(env)
	V = np.zeros(nS)
	while True:
		delta = 0
		for s in range(nS):
			v = 0
			for a, a_prob in enumerate(policy[s]):
				if a_prob == 0.0:
					continue
				ns_prob = transition_probabilities[s, a]
				next_v = V[np.arange(nS)]
				r = reward[s]
				v += np.sum(ns_prob * a_prob * (r + discount_factor * next_v))
			delta = max(delta, np.abs(v - V[s]))
			V[s] = v
		if delta < theta:
			break
	return np.array(V)
