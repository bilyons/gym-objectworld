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

size = 32
n_states = int(size**2)
n_actions = 5
gamma = 0.9
# Open env
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

# Generate linear space grid
x = np.linspace(0, size-1, size, dtype=np.int64)
y = np.linspace(0, size-1, size, dtype=np.int64)
xx,yy = np.meshgrid(x,y)
# Set up meshgrid
NY = env.grid_size-2
NX = NY
ymin = 0
ymax = env.grid_size-2
xmin = 0
xmax = env.grid_size-2
dx = (xmax -xmin)/(NX-1.)
dy = (ymax -ymin)/(NY-1.)
h=[dx,dy]

num_t = [1,2,4,8,16,32,64,128,256,512,1024]

# Define divergence function
def divergence(f, h):
	num_dims = len(f)
	return np.ufunc.reduce(np.add, [np.gradient(f[i], h[i],axis=i) for i in range(num_dims)])

for i in range(5):
	df = pd.DataFrame(columns=["Number of Trajectories", "EVD VAIRL POL", "PR VAIRL POL", "PR True VAIRL POL", "PR Reward", "Runtime VAIRL POL"])

	# Load trajectories
	file = open(os.path.abspath(os.getcwd())+"/trajectories/t_set_{}".format(i),'rb')
	ts = pickle.load(file)
	file.close()

	for t in range(len(num_t)):

		trajectories = ts[:num_t[t]]

		start = time.time()
		tot, tot1, tot2 = T.vector_field_objectworld(env, trajectories)
		t_alternate = T.convert_trajectory_style(trajectories)
		action_probabilities = np.zeros((n_states, n_actions))
		t_alternate = T.convert_trajectory_style(trajectories)
		for traj in t_alternate:
			for (s, a, ns) in traj:
				action_probabilities[s][a] += 1
		action_probabilities[action_probabilities.sum(axis=1)==0] = 1e-5
		action_probabilities/=action_probabilities.sum(axis=1).reshape(n_states,1)
			# Get outputs of vector field
		Fx = tot[xx+yy*size, 1]
		Fy = tot[xx+yy*size, 0]

		Fx1 = tot1[xx+yy*size, 1]
		Fy1 = tot1[xx+yy*size, 0]

		Fx2 = tot2[xx+yy*size, 1]
		Fy2 = tot2[xx+yy*size, 0]

		# Calculate divergence
		F= [Fx2, Fy2]
		g = divergence(F, h)
		end = time.time()
		# End VAIRL loop

		# Evaluate VAIRL
		div = g.ravel()
		# Calculate value based on this reward
		norm_val, _ = V.value_iteration(env, div, gamma)
		# Calculate policy as if the reward you have is true
		learned_pol = V.find_policy(env, div, gamma)
		# Compare
		v_learned = V.policy_eval(action_probabilities, ground_r, env, np.int(size**2), 5, gamma)
		# Time Check
		t_vairl = end-start
		# Correlation check
		pr_vairl = np.corrcoef(value_func, v_learned)[0,1]
		pr_true_vairl = np.corrcoef(v_true, norm_val)[0,1]
		pr_vairl_reward = np.corrcoef(div, ground_r)[0,1]
		print(f"VAIRL Complete {t_vairl}")
		print(f"EVD VAIRL: {np.square(v_true - v_learned).mean()} PR: VAIRL: {pr_vairl}")

		# Plot Normalized divergence
		ax = plt.subplot(111,aspect='equal',title='Divergence Value Function')
		im = plt.pcolormesh(x, y, norm_val.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
		divider = make_axes_locatable(ax)
		plt.quiver(x,y,Fx2,Fy2)
		cax = divider.append_axes("right", size="5%", pad=0.05)

		plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/{}/pol_quiver_div_value_after_{}_trajectories.svg".format(i, len(trajectories)))
		plt.clf()

		ax = plt.subplot(111,aspect='equal',title='Divergence Value Function')
		im = plt.pcolormesh(x, y, norm_val.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.05)

		plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/{}/pol_div_value_after_{}_trajectories.svg".format(i, len(trajectories)))
		plt.clf()

		df.loc[ t ] = [ num_t[t], np.square(v_true - v_learned).mean(), pr_vairl, pr_true_vairl, pr_vairl_reward, t_vairl]

		np.save(f'vairl_pol_value_trajectory_{len(trajectories)}.npy', v_learned)
		np.save(f'vairl_pol_pr_trajectory_{len(trajectories)}.npy', pr_vairl)
		np.save(f'vairl_pol_truepr_trajectory_{len(trajectories)}.npy', pr_true_vairl)
		np.save(f'vairl_pol_runtime_trajectory_{len(trajectories)}.npy', t_vairl)

	df.to_csv(os.path.abspath(os.getcwd())+'/data/{}/vairl_pol.csv'.format(i), index=False)
	del [df]