import gym, os
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ActorCritic(nn.Module):
	"""
	Implementing both heads of the actor critic model
	"""
	def __init__(self, state_space, action_space):
		super(ActorCritic, self).__init__()

		self.state_space = state_space
		self.action_space = action_space
		
		self.linear1 = nn.Linear(self.state_space, 128)
		self.linear2 = nn.Linear(128, 256)

		self.critic_head = nn.Linear(256, 1)
		self.action_head = nn.Linear(256, self.action_space)

		self.saved_actions = []
		self.rewards = []

		self.optimizer = optim.Adam(self.parameters(), lr = 1e-3)
		self.eps = np.finfo(np.float32).eps.item()

	def forward(self, state):
		"""
		Forward pass for both actor and critic
		"""

		# State to Layer 1
		output = F.relu(self.linear1(state))

		# Layer 1 to Layer 2
		output = F.relu(self.linear2(output))

		# Layer 2 to Action
		action_prob = F.softmax(self.action_head(output), dim=-1)

		# Layer 2 to Value
		value_est = self.critic_head(output)

		return value_est, action_prob

	def select_action(self,state):
		state = torch.from_numpy(state).float()
		value_est, probs = self.forward(state)

		# Make prob categoric dist
		dist = Categorical(probs)

		action = dist.sample()

		self.saved_actions.append(SavedAction(dist.log_prob(action), value_est))

		return action.item()


	def compute_returns(self, gamma):
		"""
		Calculate losses and do backprop
		"""

		R = 0

		saved_actions = self.saved_actions

		policy_losses = []
		value_losses = []
		returns = []

		for r in self.rewards[::-1]:
			# Discount value
			R = r + gamma*R
			returns.insert(0,R)

		returns = torch.tensor(returns)
		returns = (returns - returns.mean())/(returns.std()+self.eps)

		for (log_prob, value), R in zip(saved_actions, returns):
			advantage = R - value.item()

			policy_losses.append(-log_prob*advantage)

			value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

		self.optimizer.zero_grad()

		loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

		loss.backward()
		self.optimizer.step()

		del self.rewards[:]
		del self.saved_actions[:]