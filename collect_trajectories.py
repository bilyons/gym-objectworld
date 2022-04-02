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
# Set seed
random.seed(0)
np.random.seed(0)

# Set size
size = 32
n_states = int(size**2)
n_actions = 5
gamma = 0.9
# Create World
env = ObjectWorldEnv(size+2, 64, 4, 0.3, False)

filehandler = open(os.path.abspath(os.getcwd())+"/trajectories/env.pkl", "wb")
pickle.dump(env, filehandler)
filehandler.close()

num_t = 1024

ground_r = np.array([env._reward((y_i, x_i)) for (y_i, x_i) in product(range(1, env.grid_size-1), range(1, env.grid_size-1))])

print("World and reward made")

policy = V.find_policy(env, ground_r, gamma)
x = np.linspace(0, size-1, size, dtype=np.int64)
y = np.linspace(0, size-1, size, dtype=np.int64)
# Plot base cases
# Ground Truth
ax = plt.subplot(111,aspect='equal',title='True Reward Function')
im = plt.imshow(ground_r.reshape((size,size)), origin='lower')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax = cax)
plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/ground_truth.png")
plt.clf()
# Value Function
value_func, _ = V.value_iteration(env, ground_r, gamma)
v_true = V.policy_eval(policy, ground_r, env, np.int(size**2), 5, gamma)
# value_func = D.normalize(value_func)

ax = plt.subplot(111,aspect='equal',title='Normalised True Value Function')
im = plt.pcolormesh(x, y, value_func.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax = cax)
plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/value_func.png")
plt.clf()
print("Base cases saved")

# Save reward
filehandler = open(os.path.abspath(os.getcwd())+"/trajectories/ground_r.pkl", "wb")
pickle.dump(ground_r, filehandler)
filehandler.close()

# Save value function
filehandler = open(os.path.abspath(os.getcwd())+"/trajectories/value_func.pkl", "wb")
pickle.dump(value_func, filehandler)
filehandler.close()

# Save value function
filehandler = open(os.path.abspath(os.getcwd())+"/trajectories/v_true.pkl", "wb")
pickle.dump(v_true, filehandler)
filehandler.close()


for t_set in range(5):

	ts = list(T.generate_trajectories_objectworld(num_t, env, policy))

	filehandler = open(os.path.abspath(os.getcwd())+"/trajectories/t_set_{}".format(t_set), "wb")
	pickle.dump(ts, filehandler)
	filehandler.close()