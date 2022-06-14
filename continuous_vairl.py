import gym, os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gym_objectworld.solvers import catrpole as C
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from gym_objectworld.utilities import trajectory_continuous as T
import pickle
import pandas as pd
import scipy.spatial.distance as dist
env = gym.make("CartPole-v1")

train = False

if train:
	agent = C.create_agent()
	state = env.reset()
	done = False

	episode_reward = 0

	while not done:

		state = torch.FloatTensor(state).unsqueeze(0)

		action_pred = agent(state)
		
		action_prob = F.softmax(action_pred, dim = -1)

		dist = distributions.Categorical(action_prob)

		action = dist.sample()
		
		log_prob_action = dist.log_prob(action)

		env.render()
		
		state, reward, done, _ = env.step(action.item())

		episode_reward += reward

	ts = list(T.generate_trajectories(1, env, agent))

	filehandler = open(os.path.abspath(os.getcwd())+"/trajectories/t_set_{}".format(1), "wb")
	pickle.dump(ts, filehandler)

file = open(os.path.abspath(os.getcwd())+"/trajectories/t_set_{}".format(1),'rb')
ts = pickle.load(file)
file.close()

print(ts)

states, transitions = T.vector_field(env, ts)

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
		# Collect initial states and the vectors from them
		x1 = df.iloc[df.iloc[j][f"Neighbour_{i}"]]['initial_state']
		y1 = df.iloc[df.iloc[j][f"Neighbour_{i}"]]['vector']

		det = x0[0]*x1[1] - x0[1]*x1[0]
		div = (x0[0]*y0[0] - x0[1]*y1[1] - x1[0]*y1[1] + x1[1]*y1[1])/det

		df.iloc[j, df.columns.get_loc("Divergence")] = div
