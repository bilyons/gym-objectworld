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
random.seed(0)
np.random.seed(0)

size = 32

env = ObjectWorldEnv(size+2, 64, 4, 0.3, False)

style = {
	'border': {'color': 'red', 'linewidth': 0.5},
}

ALPHA = 0.1
GAMMA = 0.9
EPISODE = 4000
EPSILON = 0.7
MIN_EPSILON = 0.1
DECAY_RATE = 0.9995
TAU=0.25
np.set_printoptions(threshold=sys.maxsize)
Q = np.ones(((env.grid_size-2)**2, env.action_space.n))

def divergence(f, h):
	num_dims = len(f)
	return np.ufunc.reduce(np.add, [np.gradient(f[i], h[i],axis=i) for i in range(num_dims)])

ground_r = np.array([env._reward((y_i, x_i)) for (y_i, x_i) in product(range(1, env.grid_size-1), range(1, env.grid_size-1))])

print("World and reward made")

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

# Generate linear space grid
x = np.linspace(0, size-1, size, dtype=np.int64)
y = np.linspace(0, size-1, size, dtype=np.int64)
xx,yy = np.meshgrid(x,y)

# Ground Truth
ax = plt.subplot(111,aspect='equal',title='True Reward Function')
im = plt.imshow(ground_r.reshape((size,size)), origin='lower')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax = cax)
plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/ground_truth.png")
plt.clf()
# Value Function
value_func, _ = V.value_iteration(env, ground_r, GAMMA)
v_true = V.policy_eval(POL, ground_r, env, np.int(32**2), 5)
# value_func = D.normalize(value_func)

ax = plt.subplot(111,aspect='equal',title='Normalised True Value Function')
im = plt.pcolormesh(x, y, value_func.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax = cax)
plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/value_func.png")
plt.clf()
print("Base cases saved")

init_t = 0
num_t = [1,2,4,8,16,32,64,128,256,512,1024, 2048]

for t in range(len(num_t)):
	# Add trajectories
	if t == 0:
		ts = list(T.generate_trajectories_objectworld(num_t[0], env, POL))
	else:
		ts += list(T.generate_trajectories_objectworld(num_t[t-1], env, POL))

	# Remember to add another transition array the way they like it in IQL and IAVI



	print("Trajectories added")
	# Calculate vector field
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

	print("Divergence calculated")

	# Normalize divergences
	div = g.ravel()
	norm_div = D.normalize(div)
	norm_val, _ = V.value_iteration(env, norm_div, GAMMA)
	learned_pol = V.find_policy(env, norm_div, GAMMA)
	v_learned = V.policy_eval(learned_pol, ground_r, env, np.int(32**2), 5)
	# norm_val = D.normalize(norm_val)

	# print(value_func, norm_val)
	# print(np.square(value_func - norm_val).mean())
	# exit()
	# print("New value function learned")

	# Plot Normalized divergence
	ax = plt.subplot(111,aspect='equal',title='Divergence Value Function')
	im = plt.pcolormesh(x, y, norm_val.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
	divider = make_axes_locatable(ax)
	plt.quiver(x,y,Fx,Fy)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cbar = plt.colorbar(im, cax = cax)

	plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/quiver_div_value_after_{}_trajectories_evd_{}.png".format(len(ts), np.square(v_true - v_learned).mean()))
	plt.clf()

	ax = plt.subplot(111,aspect='equal',title='Divergence Value Function')
	im = plt.pcolormesh(x, y, norm_val.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cbar = plt.colorbar(im, cax = cax)

	plt.savefig(os.path.abspath(os.getcwd())+"/img/ow_img/div_value_after_{}_trajectories_evd_{}.png".format(len(ts),  np.square(v_true - v_learned).mean()))
	plt.clf()

	print(f"Completed run with {len(ts)} trajectories")

exit()