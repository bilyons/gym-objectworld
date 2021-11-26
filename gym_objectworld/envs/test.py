import gym
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('gym_objectworld:objectworld-gridworld-v0', size = 10, p_slip=0.01)

print()

ALPHA = 0.1
GAMMA = 0.95
EPISODE = 100000
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

episode_rewards = []
time = []
for episode in range(EPISODE):
	state = env.reset()
	done = False

	t = 0

	episode_reward = 0
	while not done:

		action = choose_action(state)

		new_state, reward, done, _ = env.step(action)
		reward*=10
		# print(reward)
		episode_reward += reward

		# print(Q)

		Q[state, action] = Q[state, action] + ALPHA*(reward + GAMMA*np.max(Q[new_state, :]) - Q[state, action])

		# print(Q)
		# exit()

		t+= 1

		if t == 100:
			done = True

		state = new_state

	if episode%10==0:
		print(f"Episode: {episode} Reward: {episode_reward} Steps: {t}")

	episode_rewards.append(episode_reward)
	time.append(t)

	EPSILON *= DECAY_RATE
	EPSILON = max(EPSILON, MIN_EPSILON)
	# if EPSILON == MIN_EPSILON:
	# 	print(episode)
	# 	exit()

print(Q)

env.close()

plt.figure(figsize=(12,5))
plt.title("steps/Episode Length")
plt.bar(range(len(time)), time, alpha=0.6, color='red', width=5)
plt.show()

plt.figure(figsize=(12,5))
plt.title("Episode Reward")
plt.bar(range(len(episode_rewards)), episode_rewards, alpha=0.6, color='blue', width=5)
plt.show()