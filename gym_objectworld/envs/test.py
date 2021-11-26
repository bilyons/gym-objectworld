import gym
import numpy as np

env = gym.make('gym_objectworld:objectworld-gridworld-v0')

print()

ALPHA = 0.1
GAMMA = 0.95
EPISODE = 100000
EPSILON = 0.9
MIN_EPSILON = 0.01
DECAY_RATE = 0.9

Q = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state):
	action = 0
	if np.random.uniform(0,1) < EPSILON:
		action = env.action_space.sample()
	else:
		action = np.random.choice(env.action_space.n, 
			p=np.exp(Q[state,:])/np.sum(np.exp(Q[state,:])))
	return action

def learn(state, new_state, reward, action):

	p = Q[state, action]
	p_t = reward + GAMMA*np.max(Q[new_state,:])
	Q[state, action] = Q[state, action] + lr*(p_t - p)

for episode in range(EPISODE):
	obs = env.reset()

	if episode % 100 == 99:
		EPSILON *= DECAY_RATE
		EPSILON = max(EPSILON, MIN_EPSILON)

	while not done:
		