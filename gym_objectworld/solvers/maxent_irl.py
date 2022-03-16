"""
Contains multiple methods of performing inverse reinforcement learning for analysis
"""
import numpy as np
from time import sleep
from gym_objectworld.solvers import value_iteration_objectworld as V
from gym_objectworld.solvers import optimizer as O
# Max Ent IRL ###########################################################################

def normalize(vals):
	"""
	normalize to (0, max_val)
	input:
	vals: 1d array
	"""
	min_val = np.min(vals)
	max_val = np.max(vals)
	return (vals - min_val) / (max_val - min_val)

# Main call - Not working yet - why?
def m_irl(env, trajectories, lr):

	feature_matrix = env._feature_matrix()
	n_states, n_features = feature_matrix.shape
	n_actions = env.action_space.n
	p_transition = V.convert_transition_array(env)

	delta = np.inf
	eps=1e-5

	theta = np.random.rand(n_states)

	print(feature_matrix)
	e_features = feature_expectations(feature_matrix, trajectories)
	p_initial = initial_probabilities(feature_matrix, trajectories)

	optimiser = O.ExpSga(lr=O.linear_decay(lr0=0.1))

	optimiser.reset(theta)
	while delta > eps:
		theta_old = theta.copy()

		# Per state reward
		r  = feature_matrix.dot(theta) # Essentially just alpha but could have different features

		# Backwards pass
		e_svf = expected_svf(action_dim, feature_matrix, transition_prob, p_initial, r)

		grad = e_features - features_matrix.T.dot(e_svf)

		optimiser.step(grad)

		delta = np.max(np.abs(theta_old - theta))
		print(delta)
	return normalize(feature_matrix.dot(theta))

# Expected state visitatoin

def expected_svf(action_dim, feature_matrix, transition_prob, p_initial, rewards):
	p_action = local_action_probability(action_dim, feature_matrix, transition_prob, rewards)
	return expected_svf_from_policy(action_dim, feature_matrix, transition_prob, p_initial, p_action)

def local_action_probability(action_dim, feature_matrix, transition_prob, rewards):
	n_states, _ = feature_matrix.shape
	n_actions = action_dim
	z_states = np.zeros((n_states))
	z_action = np.zeros((n_states, n_actions))
	p_transition = np.copy(transition_prob)

	p = [np.array(world.transition_prob[:, a, :]) for a in range(n_actions)]
	er = np.exp(rewards)*np.eye((n_states))

	zs = np.zeros(n_states)
	za = np.zeros((n_states, n_actions))

	for _ in range(2 * n_states):
		for a in range(n_actions):
			za[:,a] = np.matmul(er, np.matmul(p_transition[:,a,:], zs.T))
		zs = za.sum(axis=1)
	return za / zs[:, None]

def expected_svf_from_policy(action_dim, feature_matrix, transition_prob, p_initial, p_action, eps = 1e-5):
	n_states, _ = feature_matrix.shape
	n_actions = action_dim
	p_transition = np.copy(transition_prob)

	p_transition = [np.array(p_transition[:,a,:]) for a in range(n_actions)]

	# print(p_action)
	# forward computation of state expectations
	d = np.zeros(n_states)

	delta = np.inf

	while delta > eps:
		# print([p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)])
		d_ = [p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)]
		d_ = p_initial + np.array(d_).sum(axis=0)
		delta, d = np.max(np.abs(d_ - d)), d_
		# print(delta)
	return d

# Functions for everyone ###############################################################

def feature_expectations(feature_matrix, trajectories):
	n_states, n_features = feature_matrix.shape

	fe = np.zeros(n_features)

	for t in trajectories:
		for i in range(len(t.transitions())):
			fe += feature_matrix[t.transitions()[i][0]]
	return fe/len(trajectories)

def initial_probabilities(feature_matrix, trajectories):
	# Updatable initial probabilty, takes in previous counts and a batch of trajectories >= 1
	# And returns initial probability and count
	n_states, n_features = feature_matrix.shape
	p = np.zeros(n_states)

	for t in trajectories:
		p[t.transitions()[0][0]] += 1.0
	return p/len(trajectories)