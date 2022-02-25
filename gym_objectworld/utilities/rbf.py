import numpy as np
from scipy import *
from scipy.linalg import norm, pinv
from itertools import chain
import random

class RBFs:

	def __init__(self, env, n_rbfs):
		self.env = env
		self.observation_space = env.observation_space
		self.observation_dim = env.observation_space.shape[0]
		self.n_rbfs = n_rbfs
		self.centres = self._construct()
		
	def _construct(self):
		x_high = self.observation_space.high[0]
		y_high = self.observation_space.high[1]
		v_high = self.observation_space.high[2]

		h_range = np.array((x_high, y_high, v_high))
		l_range = -h_range

		centres = [np.random.uniform(l_range, h_range, self.observation_space.shape[0]) for i in range(self.n_rbfs)]
		return centres

	def _signal_strength(self, c, d):
		assert len(d) == self.observation_dim
		return exp(-norm(c-d)**2)

	# def _signal_strength2(self, c, d):
	# 	assert len(d) == self.observation_dim
	# 	return exp(-norm(c-d)**2)

	def _cal_activation(self, X):
		G = np.zeros((X.shape[0], self.n_rbfs), float)
		for ci, c in enumerate(self.centres):
			for xi, x in enumerate(X):
				G[xi, ci] = self._signal_strength(c, x)
		return G

	# def _cal_activation2(self, X):
	# 	G = np.zeros((X.shape[0], self.n_rbfs), float)
	# 	for ci, c in enumerate(self.centres):
	# 		for xi, x in enumerate(X):
	# 			G[xi, ci] = self._signal_strength2(c, x)
	# 	return G