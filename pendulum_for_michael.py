import gym, os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
import scipy.spatial.distance as dist
from itertools import count
from collections import namedtuple
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import math
from gym_objectworld.solvers import actorcritic_cont as AC
from gym_objectworld.utilities import trajectory_continuous as T
from gym_objectworld.utilities import rbf as R
import pandas as pd
from scipy.special import softmax
import scipy
import seaborn as sns
import matplotlib.colors as colors

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
pd.options.display.float_format = "{:,.2f}".format
env = gym.make("Pendulum-v0")

state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_'])
TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
# Train Expert AC


train = False
agent = AC.Agent(train)

# Already trained

if train == True:

	# Main loop
	window = 50
	reward_history = []

	for ep in count():

		state = env.reset()

		ep_reward = 0
		running_reward, running_q = -10000,0

		for t in range(1,1000):

			if ep%50 == 0:
				env.render()

			action = agent.select_action(state)

			state_, reward, done, _ = env.step(action)

			agent.store_transition(Transition(state, action, (reward+8)/8, state_))

			state = state_

			if agent.memory.isfull:
				q = agent.update()
				running_q = 0.99 * running_q + 0.01*q

			ep_reward += reward

			if done:
				break

		reward_history.append(ep_reward)

		# Result information
		if ep % 50 == 0:
			mean = np.mean(reward_history[-window:])
			print(f"Episode: {ep} Last Reward: {ep_reward} Rolling Mean: {mean}")

		if np.mean(reward_history[-100:])>-200:
			print(f"Environment solved at episode {ep}, average run length > 200")
			env.close()
			agent.save_param()
			with open('log/ddpg_training_records.pkl', 'wb') as f:
				pickle.dump(reward_history, f)
			break

	fig, ((ax1), (ax2)) = plt.subplots(2,1, sharey=True, figsize=[9,9])
	rolling_mean = pd.Series(reward_history).rolling(window).mean()
	std = pd.Series(reward_history).rolling(window).std()
	ax1.plot(rolling_mean)
	ax1.fill_between(range(len(reward_history)), rolling_mean-std, rolling_mean+std,
		color='orange', alpha=0.2)
	ax1.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
	ax1.set_xlabel('Episode')
	ax1.set_ylabel('Episode Length')
	ax2.plot(reward_history)
	ax2.set_title('Episode Rewards')
	ax2.set_xlabel('Episode')
	ax2.set_ylabel('Episode Length')
	plt.show()
	env.close()

# Redo?

# Trajectory numbers

ts = list(T.generate_trajectories(3, env, agent))

# Get the effective vectors of each actions and their initial state
states, transitions = T.vector_field(env, ts)

# Zip together for the data frame
raw_tuples = list(zip(states, transitions))

# Calculate all distances
distance_array = dist.cdist(states, states)

# Determine size of neighbourhood (cardinality)
size_of_neighbourhood = 3

# Sort the array and get the IDs for collating data
idx_array = distance_array.argsort()[:,:size_of_neighbourhood+1]

# Create data frame
df = pd.DataFrame(raw_tuples, columns=['initial_state', 'vector'])

# Make table of neighbours
for i in range(size_of_neighbourhood):
	df[f"Neighbour_{i}"] = idx_array[:,i+1: i+2]
	df[f"Distance_to_Neighbour_{i}"] = pd.Series(dtype="float64")
	for j in range(len(df)):
		comp = df.iloc[df.iloc[j][f"Neighbour_{i}"]]['initial_state']
		vec = df.iloc[df.iloc[j][f"Neighbour_{i}"]]['vector']
		ori = df.iloc[j]['initial_state']
		dist = np.linalg.norm(ori- comp)
		df.loc[j, f"Distance_to_Neighbour_{i}"] = dist

# Set up divergence and coordinate vector columns
df[f"Divergence"] = pd.Series(dtype="float64")
for k in range(len(states[0])):
	df[f"x_{k}"] = pd.Series(dtype="float64")

# Divergence calculations
for j in range(len(states)):

	comparing_state = df.iloc[j]["initial_state"]
	comparing_vector = df.iloc[j]["vector"]

	weighting = np.zeros(size_of_neighbourhood)
	grad = []
	for i in range(size_of_neighbourhood):

		# Calculate divergence
		# For each we have as many as i neighbours to consider
		delta_pos = np.subtract(comparing_state, df.iloc[df.iloc[j][f"Neighbour_{i}"]]['initial_state'])
		# print("delta", delta_pos)
		delta_vec = np.subtract(comparing_vector, df.iloc[df.iloc[j][f"Neighbour_{i}"]]['vector'])
		# print("delta_vec", delta_vec)
		weighting[i] = df.iloc[j][f"Distance_to_Neighbour_{i}"]
		# print("local_div", np.divide(delta_vec,delta_pos))
		grad.append(np.divide(delta_vec,delta_pos))

	# Weighting normalize
	weighting = softmax(weighting)
	grad = np.vstack(grad)

	vector = np.matmul(weighting, grad)
	div = np.sum(np.matmul(weighting, grad))

	df.iloc[j, df.columns.get_loc("Divergence")] = div

	for k in range(len(states[0])):
		df.iloc[j, df.columns.get_loc(f"x_{k}")] = vector[k]

x = states[:,0]
y = states[:,1]
v = df['Divergence']
print(df.loc[df['Divergence'].idxmax()])

rows = 1
cols = 1
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
plt.scatter(x,y,v)
cbar = plt.colorbar()

plt.show()
