import gym, os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from gym_objectworld.solvers import actorcritic as AC
from gym_objectworld.utilities import trajectory_continuous as T
np.set_printoptions(threshold=sys.maxsize)
env = gym.make("CartPole-v0")

state_space = env.observation_space.shape[0]
action_space = env.action_space.n

# Train Expert AC

model = AC.ActorCritic(state_space, action_space)

train = False

if train == True:

	# Main loop
	window = 50
	reward_history = []

	for ep in count():

		state = env.reset()

		ep_reward = 0

		for t in range(1,1000):

			if ep%50 == 0:
				env.render()

			action = model.select_action(state)

			state, reward, done, _ = env.step(action)

			model.rewards.append(reward)
			ep_reward += reward

			if done:
				break

		model.compute_returns(0.99)
		reward_history.append(ep_reward)

		# Result information
		if ep % 50 == 0:
			mean = np.mean(reward_history[-window:])
			print(f"Episode: {ep} Last Reward: {ep_reward} Rolling Mean: {mean}")

		if np.mean(reward_history[-100:])>199:
			print(f"Environment solved at episode {ep}, average run length > 200")
			break

	torch.save(model.state_dict(), os.getcwd()+'/model.pth')

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

else:
	model.load_state_dict(torch.load(os.getcwd()+'/model.pth'))

def divergence(f, h):
	num_dims = len(f)
	return np.ufunc.reduce(np.add, [np.gradient(f[i], h[i],axis=i) for i in range(num_dims)])

ts = list(T.generate_trajectories(1, env, model))


initial_array, div_array = T.vector_field(ts)

Nth = 100
Nth_dot = 100
Nx = Nth
Nx_dot = Nth_dot

xmin = env.observation_space.low[0]
xmax = env.observation_space.high[0]
x_dotmin = env.observation_space.low[0]
x_dotmax = env.observation_space.high[0]
thmax = env.observation_space.high[2]
thmin = env.observation_space.low[2]
th_dotmax = env.observation_space.high[2]
th_dotmin = env.observation_space.low[2]

dx = (xmax -xmin)/(Nx-1.)
dx_dot = (x_dotmax -x_dotmin)/(Nx_dot-1.)
dth = (thmax -thmin)/(Nth-1.)
dth_dot = (th_dotmax -th_dotmin)/(Nth_dot-1.)
h=[dx,dx_dot, dth, dth_dot]

x = initial_array[:,0]
x_dot = initial_array[:,1]
th = initial_array[:,2]
th_dot = initial_array[:,3]

xx, x_dotx_dot, thth, th_dotth_dot = np.meshgrid(x, x_dot, th, th_dot)

print(xx.shape)

print(initial_array[0])
print(xx[0])
print(x_dotx_dot[0])

Fx = np.array( [div_array[initial_array == np.array((xx[0,i,0,0], x_dotx_dot[i,0,0,0], thth[0,0,i,0], th_dotth_dot[0,0,0,i])), 0] for i in range(10)])

print(Fx)
exit()
Fx_dot = np.array([div_array[initial_array == np.array((xx[0,i,0,0], x_dotx_dot[i,0,0,0], thth[0,0,i,0], th_dotth_dot[0,0,0,i], 0))] for i in range(10)])
Fth = np.array([div_array[initial_array == np.array((xx[0,i,0,0], x_dotx_dot[i,0,0,0], thth[0,0,i,0], th_dotth_dot[0,0,0,i], 0))] for i in range(10)])
Fth_dot = np.array([div_array[initial_array == np.array((xx[0,i,0,0], x_dotx_dot[i,0,0,0], thth[0,0,i,0], th_dotth_dot[0,0,0,i],0))] for i in range(10)])
# Fx = np.array([div_array[initial_array[:,0] == xx[0,i,0,0], 0] for i in range(200) ])
# Fx_dot = np.array([div_array[initial_array[:,1] == x_dotx_dot[i,0,0,0], 1] for i in range(200) ])
# Fth = np.array([div_array[initial_array[:,2] == thth[0,0,i,0], 2] for i in range(200) ])
# Fth_dot = np.array([div_array[initial_array[:,3] == th_dotth_dot[0,0,0,i], 3] for i in range(200) ])

# Fx = np.array([div_array[]])

# print()
F = [Fx, Fx_dot, Fth, Fth_dot]
# print(F.shape)
g = divergence(F, h)

# print(g)