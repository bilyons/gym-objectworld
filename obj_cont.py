import random
import numpy as np
import gym
from scipy.special import softmax
from scipy.stats import pearsonr
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.spatial.distance as dist
import pandas as pd

from gym_objectworld.envs import ObjectWorldEnv

from gym_objectworld.solvers import value_iteration_objectworld as V
from gym_objectworld.utilities import trajectory_continuous as T
from gym_objectworld import plot as P
random.seed(0)
np.random.seed(0)

size = 10

env = ObjectWorldEnv(size+2, 5, 4, 0.3, False)

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

ts= list(T.generate_trajectories_objectworld(1, env, POL))

states, transitions = T.vector_field(env, ts)

# rbfs_tuples = list(zip(rbfs, vectors))
raw_tuples = list(zip(states, transitions))

distance_array = dist.cdist(states, states)
size_of_neighbourhood = 1

idx_array = distance_array.argsort()[:,:size_of_neighbourhood+1]

entries = []

df = pd.DataFrame(raw_tuples, columns=['initial_state', 'vector'])


for i in range(size_of_neighbourhood):
	df[f"Neighbour_{i}"] = idx_array[:,i+1: i+2]
	df[f"Distance_to_Neighbour_{i}"] = pd.Series(dtype="float64")
	for j in range(len(df)):
		comp = df.iloc[df.iloc[j][f"Neighbour_{i}"]]['initial_state']
		vec = df.iloc[df.iloc[j][f"Neighbour_{i}"]]['vector']
		ori = df.iloc[j]['initial_state']
		dist = np.linalg.norm(ori- comp)
		df.loc[j, f"Distance_to_Neighbour_{i}"] = dist

df[f"Divergence"] = pd.Series(dtype="float64")
for k in range(len(states[0])):
	df[f"x_{k}"] = pd.Series(dtype="float64")


for j in range(len(states)):

	x0 = df.iloc[j]["initial_state"]
	y0 = df.iloc[j]["vector"]

	weighting = np.zeros(size_of_neighbourhood)
	grad = []
	for i in range(size_of_neighbourhood):

		# Calculate divergence
		# For each we have as many as i neighbours to consider
		# delta_pos = np.subtract(comparing_state, df.iloc[df.iloc[j][f"Neighbour_{i}"]]['initial_state'])
		# # print("delta", delta_pos)
		# delta_vec = np.subtract(comparing_vector, df.iloc[df.iloc[j][f"Neighbour_{i}"]]['vector'])
		# # print("delta_vec", delta_vec)
		# weighting[i] = df.iloc[j][f"Distance_to_Neighbour_{i}"]
		# # print("local_div", np.divide(delta_vec,delta_pos))
		# grad.append(np.divide(delta_vec,delta_pos))

		# Collect initial states and the vectors from them
		x1 = df.iloc[df.iloc[j][f"Neighbour_{i}"]]['initial_state']
		y1 = df.iloc[df.iloc[j][f"Neighbour_{i}"]]['vector']

		det = x0[0]*x1[1] - x0[1]*x1[0]
		div = (x0[0]*y0[0] - x0[1]*y1[1] - x1[0]*y1[1] + x1[1]*y1[1])/det




	# # Weighting normalize
	# weighting = softmax(weighting)
	# grad = np.vstack(grad)

	# # print("gradient", grad)
	# vector = np.matmul(weighting, grad)
	# div = np.sum(np.matmul(weighting, grad))

	# print("divergence", div)
		df.iloc[j, df.columns.get_loc("Divergence")] = div

	# for k in range(len(states[0])):
	# 	df.iloc[j, df.columns.get_loc(f"x_{k}")] = vector[k]

x = states[:,0]
y = states[:,1]
print(df['Divergence'])
v = df['Divergence']
# v = ( v - v.max())/(v.max()-v.min())
print(df.loc[df['Divergence'].idxmax()])

# xx, yy = np.meshgrid(x,y, sparse=True)

rows = 1
cols = 1
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x,y,v,s=1)
# ax.set_zlim(-5,5)
plt.show()

plt.pcolor(df)