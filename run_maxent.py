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
gamma = 0.90
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

# for i in range(5):
df = pd.DataFrame(columns=["Number of Trajectories", "EVD MaxEnt", "PR MaxEnt", "PR Reward MaxEnt", "Runtime MaxEnt"])

# Load trajectories
i=4
file = open(os.path.abspath(os.getcwd())+"/trajectories/t_set_{}".format(i),'rb')
ts = pickle.load(file)
file.close()
env.discrete = True


for t in range(len(num_t)):

	trajectories = ts[:num_t[t]]

	start = time.time()
	r = maxent.irl(env, gamma, ts, 0.01)
	end = time.time()
	t_maxent = end-start
	# Calculate value based on this reward
	maxent_val, _ = V.value_iteration(env, r, gamma)

	learned_pol = V.find_policy(env, r, gamma)
	# Compare
	v_maxent = V.policy_eval(learned_pol, ground_r, env, np.int(size**2), 5)
	t_vairl = end-start
	# Correlation check
	pr_maxent = np.corrcoef(value_func, maxent_val)[0,1]
	pr_maxent_reward = np.corrcoef(r, ground_r)[0,1]
	print(f"Maxent Complete Complete {t_maxent}")
	print(f"EVD Maxent: {np.square(v_true - v_maxent).mean()} PR: Maxent: {pr_maxent}")

	# Plot 
	ax = plt.subplot(111,aspect='equal',title='MaxEnt Value Function')
	im = plt.pcolormesh(x, y, v_maxent.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cbar = plt.colorbar(im, cax = cax)

	plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/{}/evd_maxent_value_after_{}_trajectories.png".format(i, len(trajectories)))
	plt.clf()

	# Plot 
	ax = plt.subplot(111,aspect='equal',title='True MaxEnt Value Function')
	im = plt.pcolormesh(x, y, maxent_val.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cbar = plt.colorbar(im, cax = cax)

	plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/{}/maxent_value_after_{}_trajectories.png".format(i, len(trajectories)))
	plt.clf()

	df.loc[ t ] = [ num_t[t], np.square(v_true - v_maxent).mean(), pr_maxent, pr_maxent_reward, t_maxent]

df.to_csv(os.path.abspath(os.getcwd())+'/data/{}/maxent.csv'.format(i), index=False)
del [df]