import gym
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from solvers import value_iteration as V
from utilities import trajectory as T
import plot as P

env = gym.make('gym_objectworld:objectworld-gridworld-v0', size = 9, p_slip=0.000000001, n_rewards=2, rand=True)
np.set_printoptions(suppress=True, precision=5)

ALPHA = 0.1
GAMMA = 0.95
EPISODE = 4000
EPSILON = 0.7
MIN_EPSILON = 0.1
DECAY_RATE = 0.9995
TAU=0.25

Q = np.ones((env.observation_space.n, env.action_space.n))

def choose_action(state):
	action = 0
	if np.random.rand() < EPSILON:
		action = env.action_space.sample()
	else:
		prob = softmax(Q[state,:]/TAU)
		action = np.random.choice(env.action_space.n, p=prob)
	return action

def policy_eval(env, Q):
	pol = np.zeros((env.observation_space.n, env.action_space.n))
	for s in range(env.observation_space.n):
		prob = softmax(Q[s,:]/TAU)
		pol[s,:] = prob
	return pol

def divergence_calc(array):
	length, width, action_size = array.shape
	# div_array = np.zeros((length, width))
	div_array = np.gradient(array, axis=0) +np.gradient(array, axis=1)
	# for a in range(action_size):
	return div_array

def divergence(f, h):
	num_dims = len(f)
	return np.ufunc.reduce(np.add, [np.gradient(f[i], h[i],axis=i) for i in range(num_dims)])

reward = np.zeros((env.observation_space.n))
reward[-1] = 10.0
reward[72] = 10.0

NY = 9
NX = NY
ymin = 0
ymax = 9
xmin = 0
xmax = 9
dx = (xmax -xmin)/(NX-1.)
dy = (ymax -ymin)/(NY-1.)
h=[dx,dy]

POL = V.find_policy(env, reward, GAMMA)
style = {
	'border': {'color': 'red', 'linewidth': 0.5},
}

ts= list(T.generate_trajectories_gridworld(3000, env, POL))

tot, tot1, tot2 = T.vector_field_gridworld(env,ts)
print(tot1)

size = np.int(np.sqrt(env.observation_space.n))

x = np.linspace(0, size-1, size, dtype=np.int64)
y = np.linspace(0, size-1, size, dtype=np.int64)

xx,yy = np.meshgrid(x,y)
zz = xx+size*yy

Fx = tot[xx+yy*size, 1]
Fy = tot[xx+yy*size, 0]

Fx1 = tot1[xx+yy*size, 1]
Fy1 = tot1[xx+yy*size, 0]

Fx2 = tot2[xx+yy*size, 1]
Fy2 = tot2[xx+yy*size, 0]

F= [Fx, Fy]
g = divergence(F, h)
g/=3000
rows=1
cols=3
# g = divergence(F,h)
ax = plt.subplot(rows,cols,1,aspect='equal',title='div numerical outward moves')
#im=plt.pcolormesh(x, y, g)
im = plt.pcolormesh(x, y, g, shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
plt.quiver(x,y,Fy,Fx)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax = cax,format='%.1f')

ax = plt.subplot(rows,cols,2,aspect='equal',title='div numerical inward moves')
#im=plt.pcolormesh(x, y, g)
im = plt.pcolormesh(x, y, g, shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
plt.quiver(x,y,Fy1,Fx1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax = cax,format='%.1f')

ax = plt.subplot(rows,cols,3,aspect='equal',title='div numerical outward-inward moves')
#im=plt.pcolormesh(x, y, g)
im = plt.pcolormesh(x, y, g, shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
plt.quiver(x,y,Fy2,Fx2)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax = cax,format='%.1f')
plt.show()
exit()

print(tot1)

div = divergence_calc(tot1)

print(div)
print(div.sum(axis=2))
# exit()
plt.imshow(div.sum(axis=2)/3000)

plt.colorbar()
plt.show()

ax = plt.figure(num='Divergence as reward from outward movement (normalised) 2 rewards rand').add_subplot(111)
P.plot_state_values(ax, env, div.sum(axis=2)/3000, **style)
plt.draw()

div = divergence_calc(tot)

ax = plt.figure(num='Divergence as reward from outward-in movement (normalised) 2 rewards rand').add_subplot(111)
P.plot_state_values(ax, env, div.sum(axis=2)/3000, **style)
plt.draw()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
x = np.arange(9)
y = np.arange(9)
X,Y = np.meshgrid(x,y)
surf = ax.plot_surface(X,Y,(div.sum(axis=2)/3000).reshape((9,9)))
plt.show()
exit()


div = divergence_calc(tot)

print(div)
print(div.sum(axis=2))
# exit()
plt.imshow(div.sum(axis=2)/3000)

plt.colorbar()
plt.show()
