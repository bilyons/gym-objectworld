import random
import numpy as np
import gym
import os
import sys
from scipy.special import softmax
from scipy.stats import pearsonr
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gym_objectworld.envs import ObjectWorldEnv
from gym_objectworld.solvers import value_iteration_objectworld as V
from gym_objectworld.utilities import trajectory as T
from gym_objectworld import plot as P
from gym_objectworld.utilities import discrete as D
from gym_objectworld.solvers import iavi
from gym_objectworld.solvers import iql
from gym_objectworld.solvers import maxent
import time
import pandas as pd
import pickle
np.set_printoptions(threshold=sys.maxsize)
size = 32
n_states = int(size**2)
n_actions = 5
gamma = 0.9
# Open env
# Generate linear space grid
x = np.linspace(0, size-1, size, dtype=np.int64)
y = np.linspace(0, size-1, size, dtype=np.int64)
file = open(os.path.abspath(os.getcwd())+"/trajectories/env.pkl",'rb')
env = pickle.load(file)
file.close()

# Load ground truth
file = open(os.path.abspath(os.getcwd())+"/trajectories/ground_r.pkl",'rb')
ground_r = pickle.load(file)
file.close()

# Load true value function
file = open(os.path.abspath(os.getcwd())+"/trajectories/value_func.pkl",'rb')
value_func = pickle.load(file)
file.close()

file = open(os.path.abspath(os.getcwd())+"/trajectories/v_true.pkl",'rb')
v_true = pickle.load(file)
file.close()


num_t = [1,2,4,8,16,32,64,128,256,512,1024]

# Define divergence function
def divergence(f, h):
	num_dims = len(f)
	return np.ufunc.reduce(np.add, [np.gradient(f[i], h[i],axis=i) for i in range(num_dims)])

for i in range(5):

	# i = 4
	df = pd.DataFrame(columns=["Number of Trajectories", "EVD IQL", "PR IQL", "PR True IQL", "PR Reward IQL", "Runtime IQL"])

	# Load trajectories
	file = open(os.path.abspath(os.getcwd())+"/trajectories/t_set_{}".format(i),'rb')
	ts = pickle.load(file)
	file.close()

	for t in range(len(num_t)):

		trajectories = ts[:num_t[t]]

		action_probabilities = np.zeros((n_states, n_actions))
		t_alternate = T.convert_trajectory_style(trajectories)
		for traj in t_alternate:
			for (s, a, ns) in traj:
				action_probabilities[s][a] += 1
		action_probabilities[action_probabilities.sum(axis=1)==0] = 1e-5
		action_probabilities/=action_probabilities.sum(axis=1).reshape(n_states,1)
		
		# IQL Loop
		start = time.time()
		q, r, boltz = iql.inverse_q_learning(n_states, n_actions,  gamma, t_alternate, \
	                                         alpha_r=0.0001, alpha_q=0.01, alpha_sh=0.01, epochs=100, real_distribution=action_probabilities)

		# print(q)

		# exit()
		end = time.time()

		t_iql = end - start
		# Evaluate IQL
		v_iql = V.policy_eval(boltz, ground_r, env, np.int(size**2), 5, gamma)
		v_q_iql = np.amax(q, axis= 1)
		pr_iql = np.corrcoef(value_func, v_iql)[0,1]
		pr_q_iql = np.corrcoef(v_true, v_q_iql)[0,1]
		pr_r_iql = np.corrcoef(np.amax(r, axis=1), ground_r)[0,1]

		print(f"IQL Complete {t_iql}")
		print(f"EVD iql: {np.square(v_true - v_iql).mean()} PR: iql: {pr_q_iql}")

		# Plot 
		ax = plt.subplot(111,aspect='equal',title='IQL Value Function')
		im = plt.pcolormesh(x, y, v_iql.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.05)
		cbar = plt.colorbar(im, cax = cax)

		plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/{}/iql_value_after_{}_trajectories.png".format(i, len(trajectories)))
		plt.clf()

		ax = plt.subplot(111,aspect='equal',title='True IQL Value Function')
		im = plt.pcolormesh(x, y, v_q_iql.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.05)
		cbar = plt.colorbar(im, cax = cax)

		plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/{}/true_iql_value_after_{}_trajectories.png".format(i, len(trajectories)))
		plt.clf()

		df.loc[ t ] = [ num_t[t], np.square(v_true - v_iql).mean(), pr_iql, pr_q_iql, pr_r_iql, t_iql]

	df.to_csv(os.path.abspath(os.getcwd())+'/data/{}/iql.csv'.format(i), index=False)
	del [df]