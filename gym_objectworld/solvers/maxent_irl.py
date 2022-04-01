"""
Contains multiple methods of performing inverse reinforcement learning for analysis
"""
import numpy as np
from time import sleep
from gym_objectworld.solvers import value_iteration_objectworld as V
from gym_objectworld.utilities import discrete as D
from itertools import product
# Main call - Not working yet - why?
def irl(env, trajectories, lr, gamma, transition_prob):

	feature_matrix = env._feature_matrix()
	n_states, n_features = feature_matrix.shape
	n_actions = env.action_space.n
	
	delta = np.inf
	theta = np.random.rand(n_features)

	e_features = feature_expectations(feature_matrix, trajectories)
	p_initial = initial_probabilities(feature_matrix, trajectories)

	t=1
	eps = 1e-4
	while t<=100:
		theta_old = theta.copy()

		# Per state reward
		r  = feature_matrix.dot(theta) # Essentially just alpha but could have different features

		# Find policy based on r
		policy = V.find_policy(env, r, gamma)
		# Backwards pass

		# policy = causal_expectations(transition_prob, r, gamma, eps=1e-4)

		e_svf = find_expected_svf(env, r, gamma, trajectories)

		grad = e_features - feature_matrix.T.dot(e_svf)

		theta += lr*grad

		# theta = D.normalize(theta)

		delta = np.max(np.abs(theta_old - theta))

		print("main delta ",delta, t)

		t+=1

		# print(feature_matrix.dot(theta).reshape((32,32)))

	return feature_matrix.dot(theta), policy

# Expected state visitatoin


def find_expected_svf(env, r, gamma, trajectories):
    n_trajectories = len(trajectories)
    trajectory_length = 9

    n_states, _ = env._feature_matrix().shape
    n_actions = env.action_space.n
    transition_probability = V.convert_transition_array(env)

    # policy = find_policy(n_states, r, n_actions, gamma,
    #                                 transition_probability)
    policy = V.find_policy(env, r, gamma)

    start_state_count = np.zeros(n_states)
    for trajectory in trajectories:
        start_state_count[trajectory.transitions()[0][0]] += 1
    p_start_state = start_state_count/n_trajectories

    expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[k, t] += (expected_svf[i, t-1] *
                                  policy[i, j] * # Stochastic policy
                                  transition_probability[i, j, k])

    return expected_svf.sum(axis=1)

def causal_expectations(transition_prob, reward, discount, eps=1e-4):
	n_states, n_actions, _ = transition_prob.shape

	reward_terminal = -np.inf * np.ones(n_states)

	p = [np.array(transition_prob[:, a, :]) for a in range(n_actions)]

	v = -1e200 * np.ones(n_states)

	delta = np.inf
	while delta >eps:
		v_old = v

		q = np.array([reward + discount * p[a].dot(v_old) for a in range(n_actions)]).T

		v = reward_terminal
		for a in range(n_actions):
			v = softmax(v, q[:, a])

		# for some reason numpy chooses an array of objects after reduction, force floats here
		v = np.array(v, dtype=np.float)

		delta = np.max(np.abs(v - v_old))

		# print("d causal ", delta)

	# compute and return policy
	return np.exp(q - v[:, None])
	
# Functions for everyone ###############################################################

def feature_expectations(feature_matrix, trajectories):
	n_states, n_features = feature_matrix.shape

	fe = np.zeros(n_features)

	for t in trajectories:
		for i in t.states():
			fe += feature_matrix[i,:]
	return fe/len(trajectories)

def initial_probabilities(feature_matrix, trajectories):
	# Updatable initial probabilty, takes in previous counts and a batch of trajectories >= 1
	# And returns initial probability and count
	n_states, n_features = feature_matrix.shape
	p = np.zeros(n_states)

	for t in trajectories:
		p[t.transitions()[0][0]] += 1.0
	return p/len(trajectories)

def softmax(x1, x2):
    """
    Computes a soft maximum of both arguments.

    In case `x1` and `x2` are arrays, computes the element-wise softmax.

    Args:
        x1: Scalar or ndarray.
        x2: Scalar or ndarray.

    Returns:
        The soft maximum of the given arguments, either scalar or ndarray,
        depending on the input.
    """
    x_max = np.maximum(x1, x2)
    x_min = np.minimum(x1, x2)
    return x_max + np.log(1.0 + np.exp(x_min - x_max))