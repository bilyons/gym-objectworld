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
# Set seed
random.seed(0)
np.random.seed(0)

# Set size
size = 10
n_states = int(size**2)
n_actions = 5

# Create World
env = ObjectWorldEnv(size+2, 64, 4, 0.3, False)

transition_matrix = V.convert_transition_array(env)

# Set hyper parameters
GAMMA = 0.9

# Define divergence function
def divergence(f, h):
	num_dims = len(f)
	return np.ufunc.reduce(np.add, [np.gradient(f[i], h[i],axis=i) for i in range(num_dims)])

# Determine ground truth array
ground_r = np.array([env._reward((y_i, x_i)) for (y_i, x_i) in product(range(1, env.grid_size-1), range(1, env.grid_size-1))])

print("World and reward made")

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
POL = V.find_policy(env, ground_r, GAMMA)

# Get size
size = np.int(env.grid_size-2)
epochs=100
# Generate linear space grid
x = np.linspace(0, size-1, size, dtype=np.int64)
y = np.linspace(0, size-1, size, dtype=np.int64)
xx,yy = np.meshgrid(x,y)

# Plot base cases
# Ground Truth
ax = plt.subplot(111,aspect='equal',title='True Reward Function')
im = plt.imshow(ground_r.reshape((size,size)), origin='lower')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax = cax)
plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/{}/ground_truth.png".format(epochs))
plt.clf()
# Value Function
value_func, _ = V.value_iteration(env, ground_r, GAMMA)
v_true = V.policy_eval(POL, ground_r, env, np.int(size**2), 5)
# value_func = D.normalize(value_func)

ax = plt.subplot(111,aspect='equal',title='Normalised True Value Function')
im = plt.pcolormesh(x, y, value_func.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax = cax)
plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/{}/value_func.png".format(epochs))
plt.clf()
print("Base cases saved")

init_t = 0
num_t = [1,2,4,8,16,32,64,128,256,512,1024]

# Repeat 5 times
# for run in range(0,5):
df = pd.DataFrame(columns=["Number of Trajectories", "EVD VAIRL", "EVD IAVI", "EVD IQL", "EVD MAXENT", "PR VAIRL", "PR IAVI", "PR True IAVI", "PR Maxent", "PR IQL", "PR True IQL", "Runtime VAIRL", "Runtime IAVI", "Runtime IQL", "Runtime AP", "Runtime Maxent"]) 

run = 0
random.seed(run)
np.random.seed(run)
for t in range(len(num_t)):
	# Add trajectories
	if t == 0:
		ts = list(T.generate_trajectories_objectworld(100, env, POL))
	else:
		ts += list(T.generate_trajectories_objectworld(num_t[t-1], env, POL))

	# Get trajectories in appropriate form for IAVI and IQL
	t_alternate = T.convert_trajectory_style(ts)
	action_probabilities = np.zeros((n_states, n_actions))

	start = time.time()
	for traj in t_alternate:
		for (s, a, ns) in traj:
			action_probabilities[s][a] += 1
	action_probabilities[action_probabilities.sum(axis=1)==0] = 1e-5
	action_probabilities/=action_probabilities.sum(axis=1).reshape(n_states,1)
	end = time.time()
	a_p_time = end - start
	print("Trajectories added")
	# Calculate vector field

	# VAIRL Loop ########################################################################################
	start = time.time()
	tot, tot1, tot2 = T.vector_field_objectworld(env,ts)

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
	# Normalise divergence rewards
	norm_div = D.normalize(div)
	# Calculate value based on this reward
	norm_val, _ = V.value_iteration(env, norm_div, GAMMA)
	# Calculate policy as if the reward you have is true
	learned_pol = V.find_policy(env, norm_div, GAMMA)
	# Compare
	v_learned = V.policy_eval(learned_pol, ground_r, env, np.int(size**2), 5)
	# Time Check
	t_vairl = end-start
	# Correlation check
	pr_vairl = np.corrcoef(v_true, norm_val)[0,1]

	print(f"VAIRL Complete {t_vairl}")
	print(f"EVD VAIRL: {np.square(v_true - v_learned).mean()} PR: VAIRL: {pr_vairl}")

	#######################################################################################################
	# IQL Loop
	start = time.time()
	q, r, boltz = iql.inverse_q_learning(n_states, n_actions,  GAMMA, t_alternate, \
                                         alpha_r=0.0001, alpha_q=0.01, alpha_sh=0.01, epochs=100, real_distribution=action_probabilities)
	end = time.time()

	t_iql = end - start
	# Evaluate IQL
	v_iql = V.policy_eval(boltz, ground_r, env, np.int(size**2), 5)
	v_q_iql = np.amax(q, axis= 1)
	pr_iql = np.corrcoef(v_true, v_iql)[0,1]
	pr_q_iql = np.corrcoef(v_true, v_q_iql)[0,1]

	print(f"IQL Complete {t_iql}")
	print(f"EVD iql: {np.square(v_true - v_iql).mean()} PR: iql: {pr_iql}")

	# End IQL Loop
	#######################################################################################################
	# IAVI Loop
	start = time.time()
	q, r, boltz = iavi.inverse_action_value_iteration(n_states, n_actions, GAMMA, transition_matrix, action_probabilities, epochs=100, theta=0.01)
	end = time.time()
	v_iavi = V.policy_eval(boltz, ground_r, env, np.int(size**2), 5)
	v_q_iavi = np.amax(q, axis= 1)
	pr_iavi = np.corrcoef(v_true, v_iavi)[0,1]
	pr_q_iavi = np.corrcoef(v_true, v_q_iavi)[0,1]
	
	# Evaluate IAVI
			
	# End IAVI Loop
	t_iavi = end - start
	print(f"IAVI Complete {t_iavi}")
	print(f"EVD iavi: {np.square(v_true - v_iavi).mean()} PR: iavi: {pr_iavi}")

	# Maxent Loop
	start = time.time()
	r = maxent.irl(env, GAMMA, ts, 0.01)
	end = time.time()
	t_maxent = end-start
	norm_r = D.normalize(r)
	# Calculate value based on this reward
	maxent_val, _ = V.value_iteration(env, norm_r, GAMMA)

	learned_pol = V.find_policy(env, norm_r, GAMMA)
	# Compare
	v_maxent = V.policy_eval(learned_pol, ground_r, env, np.int(size**2), 5)
	t_vairl = end-start
	# Correlation check
	pr_maxent = np.corrcoef(v_true, maxent_val)[0,1]
	print(f"Maxent Complete Complete {t_maxent}")
	print(f"EVD Maxent: {np.square(v_true - v_maxent).mean()} PR: Maxent: {pr_maxent}")
	df.loc[ np.int(t+run*len(num_t)) ] = [ num_t[t], np.square(v_true - v_learned).mean() , np.square(v_true - v_iavi).mean(), np.square(v_true - v_iql).mean(), np.square(v_true - v_maxent).mean(), \
																pr_vairl, pr_iavi, pr_q_iavi, pr_iql, pr_q_iql , pr_maxent, \
																t_vairl, t_iavi, t_iql, a_p_time, t_maxent ]
	# Plot graphs

	# Plot Normalized divergence
	ax = plt.subplot(111,aspect='equal',title='Divergence Value Function')
	im = plt.pcolormesh(x, y, norm_val.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
	divider = make_axes_locatable(ax)
	plt.quiver(x,y,Fx2,Fy2)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cbar = plt.colorbar(im, cax = cax)

	plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/{}/{}/quiver_div_value_after_{}_trajectories_run_{}.png".format(epochs, run,len(ts), run))
	plt.clf()

	ax = plt.subplot(111,aspect='equal',title='Divergence Value Function')
	im = plt.pcolormesh(x, y, norm_val.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cbar = plt.colorbar(im, cax = cax)

	plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/{}/{}/div_value_after_{}_trajectories_evd_{}.png".format(epochs, run, len(ts),  run))
	plt.clf()

	ax = plt.subplot(111,aspect='equal',title='IQL Value Function')
	im = plt.pcolormesh(x, y, norm_val.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cbar = plt.colorbar(im, cax = cax)

	plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/{}/{}/iql_value_after_{}_trajectories_evd_{}.png".format(epochs, run, len(ts),  run))
	plt.clf()

	ax = plt.subplot(111,aspect='equal',title='IAVI Value Function')
	im = plt.pcolormesh(x, y, norm_val.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cbar = plt.colorbar(im, cax = cax)

	plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/{}/{}/iavi_value_after_{}_trajectories_evd_{}.png".format(epoch, run, len(ts),  run))
	plt.clf()

	ax = plt.subplot(111,aspect='equal',title='True IQL Value Function')
	im = plt.pcolormesh(x, y, v_q_iql.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cbar = plt.colorbar(im, cax = cax)

	plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/{}/{}/iql_value_after_{}_trajectories_evd_{}.png".format(epochs, run, len(ts),  run))
	plt.clf()

	ax = plt.subplot(111,aspect='equal',title='True IAVI Value Function')
	im = plt.pcolormesh(x, y, v_q_iavi.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cbar = plt.colorbar(im, cax = cax)

	plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/{}/{}/iavi_value_after_{}_trajectories_evd_{}.png".format(epochs, run, len(ts),  run))
	plt.clf()

	ax = plt.subplot(111,aspect='equal',title='MaxEnt Value Function')
	im = plt.pcolormesh(x, y, v_maxent.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cbar = plt.colorbar(im, cax = cax)
	plt.show()
	plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/{}/{}/iavi_value_after_{}_trajectories_evd_{}.png".format(epochs, run, len(ts),  run))
	plt.clf()
	# Save data to table

	print(f"Completed run with {len(ts)} trajectories")

df.to_csv(os.path.abspath(os.getcwd())+'/data/{}/run_{}.csv'.format(epochs, run), index=False)
del [df]
