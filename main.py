import random
import numpy as np
import gym
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

size = 32

env = ObjectWorldEnv(size+2, 50, 4, 0.3, False)

style = {
	'border': {'color': 'red', 'linewidth': 0.5},
}

ALPHA = 0.1
GAMMA = 0.95
EPISODE = 4000
EPSILON = 0.7
MIN_EPSILON = 0.1
DECAY_RATE = 0.9995
TAU=0.25

Q = np.ones(((env.grid_size-2)**2, env.action_space.n))

def choose_action(env, state):
	actionable_state = (state[0]-1)*env.grid_size-2 + state[1]-1

	action = 0
	if np.random.rand() < EPSILON:
		action = env.action_space.sample()
	else:
		prob = softmax(Q[state,:]/TAU)
		action = np.random.choice(env.action_space.n, p=prob)
	return action

def divergence(f, h):
	num_dims = len(f)
	return np.ufunc.reduce(np.add, [np.gradient(f[i], h[i],axis=i) for i in range(num_dims)])

ground_r = np.array([env._reward((y_i, x_i)) for (y_i, x_i) in product(range(1, env.grid_size-1), range(1, env.grid_size-1))])


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

ts= list(T.generate_trajectories_objectworld(3000, env, POL))

tot, tot1, tot2 = T.vector_field_objectworld(env,ts)

size = np.int(env.grid_size-2)

x = np.linspace(0, size-1, size, dtype=np.int64)
y = np.linspace(0, size-1, size, dtype=np.int64)

xx,yy = np.meshgrid(x,y)
zz = xx+size*yy

# print(tot)
# print(zz)
# exit()

Fx = tot[xx+yy*size, 1]
Fy = tot[xx+yy*size, 0]
# print(Fx)
# exit()
Fx1 = tot1[xx+yy*size, 1]
Fy1 = tot1[xx+yy*size, 0]

Fx2 = tot2[xx+yy*size, 1]
Fy2 = tot2[xx+yy*size, 0]



F= [Fx2, Fy2]
g = divergence(F, h)
# print(g.shape)
# exit()
# print(Fx)
# print(Fy)
# exit()

# Plotting

rows=3
cols=3

# Divergences
ax = plt.subplot(rows,cols,1,aspect='equal',title='div numerical outward moves')
#im=plt.pcolormesh(x, y, g)
im = plt.pcolormesh(x, y, g, shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
plt.quiver(x,y,Fy,Fx)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax = cax)

ax = plt.subplot(rows,cols,2,aspect='equal',title='div numerical inward moves')
#im=plt.pcolormesh(x, y, g)
im = plt.pcolormesh(x, y, g, shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
plt.quiver(x,y,Fy1,Fx1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax = cax)

ax = plt.subplot(rows,cols,3,aspect='equal',title='div numerical outward-inward moves')
#im=plt.pcolormesh(x, y, g)
im = plt.pcolormesh(x, y, g, shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
plt.quiver(x,y,Fy2,Fx2)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax = cax)

# Ground Truth

ax = plt.subplot(rows,cols,4,aspect='equal',title='ground_truth')
im = plt.imshow(ground_r.reshape((size,size)), origin='lower')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax = cax)

# Value Function

value_func, _ = V.value_iteration(env, ground_r, GAMMA)
value_func = D.normalize(value_func)

# 

div = g.ravel()
norm_div = D.normalize(div)
norm_val, _ = V.value_iteration(env, norm_div, GAMMA)
norm_val = D.normalize(norm_val)

ax = plt.subplot(rows,cols,5,aspect='equal',title='Normalised True Value Function')
im = plt.pcolormesh(x, y, value_func.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax = cax)

ax = plt.subplot(rows,cols,6,aspect='equal',title='Normalised Divergence - Pearson Coeff {}'.format(np.corrcoef(value_func, norm_val)[0,1]))
im = plt.pcolormesh(x, y, norm_val.reshape((size,size)), shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
divider = make_axes_locatable(ax)
plt.quiver(x,y,Fx,Fy)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax = cax)

plt.show()
exit()