import gym, os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from gym_objectworld.solvers import actorcritic as AC
from gym_objectworld.utilities import trajectory_continuous as T
from gym_objectworld.utilities import rbf as R

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
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

ts = list(T.generate_trajectories(20, env, model))

x = R.RBFs(env, 20)
exit()

vector_array, h = T.vector_field(ts)

Fx = vector_array[:,:,:,:,0]
Fx_dot = vector_array[:,:,:,:,1]
Fth = vector_array[:,:,:,:,2]
Fth_dot = vector_array[:,:,:,:,3]

F = [Fx, Fx_dot, Fth, Fth_dot]

g = divergence(F, h)

print(g.shape)
# Reduce to 2D for visualisation
reduced_array = np.zeros((g.shape[0], g.shape[2]))

for j in range(g.shape[2]):
	for i in range(g.shape[0]):
		reduced_array[i,j] = np.max(g[i,:,j,:])


x = np.linspace(0, g.shape[0], g.shape[0], dtype=np.int64)
y = np.linspace(0, g.shape[2], g.shape[2], dtype=np.int64)

rows = 1
cols = 1
ax = plt.subplot(rows,cols,1,aspect='equal',title='div numerical outward moves')
#im=plt.pcolormesh(x, y, g)
im = plt.pcolormesh(x, y, reduced_array.T, shading='nearest', cmap=plt.cm.get_cmap('coolwarm'))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax = cax)
plt.show()