import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from gym_objectworld.solvers import actorcritic as AC


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v0").unwrapped

state_space = env.observation_space.shape[0]
action_space = env.action_space.n
lr = 0.0001

# Train Expert AC

actor = AC.Actor(state_space, action_space).to(device)
critic = AC.Critic(state_space, action_space).to(device)

optimizerA = optim.Adam(actor.parameters())
optimizerC = optim.Adam(critic.parameters())

for ep in range(1000):

	state = env.reset()
	log_probs = []
	values = []
	rewards = []
	masks = []
	entropy = 0

	for i in count():

		if ep%50 == 0:
			env.render()

		state = torch.FloatTensor(state).to(device)

		dist, value = actor(state), critic(state)

		action = dist.sample()

		next_state, reward, done, _ = env.step(action.cpu().numpy())

		log_prob = dist.log_prob(action).unsqueeze(0)

		entropy += dist.entropy().mean()

		log_probs.append(log_prob)
		values.append(value)
		rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
		masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

		state = next_state

		if done:
			print(f'Episode: {ep} Length: {i}')
			break

	next_state = torch.FloatTensor(next_state).to(device)
	next_value = critic(next_state)
	returns = AC.compute_returns(next_value, rewards, masks)

	log_probs = torch.cat(log_probs)
	returns = torch.cat(returns).detach()
	values = torch.cat(values)

	advantage = returns - values

	actor_loss = -(log_probs*advantage.detach()).mean()
	critic_loss = advantage.pow(2).mean()

	optimizerA.zero_grad()
	optimizerC.zero_grad()

	actor_loss.backward()
	critic_loss.backward()

	optimizerA.step()
	optimizerC.step()

env.close()