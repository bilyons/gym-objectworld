import gym
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
from solvers import value_iteration as V
from utilities import trajectory as T
import plot as P

env = gym.make('gym_objectworld:objectworld-gridworld-v0', size = 9, p_slip=0.000000001, n_rewards=1)
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

reward = np.zeros((env.observation_space.n))
reward[-1] = 10.0
# reward[72] = 10.0

POL = V.find_policy(env, reward, GAMMA)
style = {
	'border': {'color': 'red', 'linewidth': 0.5},
}
print(POL)
# ax = plt.figure(num='After training').add_subplot(111)
# P.plot_stochastic_policy(ax, env, POL, **style)
# plt.show()
ts= list(T.generate_trajectories(3, env, POL))
T.check_terminal_ratio(ts)
tot, tot1 = T.in_out_calc_it_all_about(env, ts)

x1 = np.arange(9)
x2 = np.arange(9)
u = tot1[:,:,0]
v = tot1[:,:,1]
# plt.quiver(x1,x2, u,v)
# plt.show()
print(tot1)
tot1=np.gradient(tot1, axis=0)+np.gradient(tot1, axis=1)
print(tot1)
print(tot1.sum(axis=2))
plt.imshow(tot1.sum(axis=2))
plt.colorbar()
plt.show()
exit()
u = tot1[:,:,0]
v = tot1[:,:,1]
plt.quiver(x1,x2, u,v)
plt.show()

