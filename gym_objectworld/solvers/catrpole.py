import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import gym

train_env = gym.make('CartPole-v1')
test_env = gym.make('CartPole-v1')
INPUT_DIM = train_env.observation_space.shape[0]
HIDDEN_DIM = 128
OUTPUT_DIM = train_env.action_space.n
LEARNING_RATE = 0.01

MAX_EPISODES = 1000
DISCOUNT_FACTOR = 0.99
N_TRIALS = 50
REWARD_THRESHOLD = 475
PRINT_EVERY = 10

train_rewards = []
test_rewards = []

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

def train(env, policy, optimizer, discount_factor):
    
    policy.train()
    
    log_prob_actions = []
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:

        state = torch.FloatTensor(state).unsqueeze(0)

        action_pred = policy(state)
        
        action_prob = F.softmax(action_pred, dim = -1)
                
        dist = distributions.Categorical(action_prob)

        action = dist.sample()
        
        log_prob_action = dist.log_prob(action)
        
        state, reward, done, _ = env.step(action.item())

        log_prob_actions.append(log_prob_action)
        rewards.append(reward)

        episode_reward += reward

    log_prob_actions = torch.cat(log_prob_actions)
        
    returns = calculate_returns(rewards, discount_factor)
        
    loss = update_policy(returns, log_prob_actions, optimizer)

    return loss, episode_reward

def calculate_returns(rewards, discount_factor, normalize = True):
    
    returns = []
    R = 0
    
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
        
    returns = torch.tensor(returns)
    
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
        
    return returns

def update_policy(returns, log_prob_actions, optimizer):
    
    returns = returns.detach()
    
    loss = - (returns * log_prob_actions).sum()
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    return loss.item()

def evaluate(env, policy):
    
    policy.eval()
    
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:
        
        state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
        
            action_pred = policy(state)
        
            action_prob = F.softmax(action_pred, dim = -1)
                            
        action = torch.argmax(action_prob, dim = -1)
            
        state, reward, done, _ = env.step(action.item())

        episode_reward += reward
        
    return episode_reward

policy = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)

def create_agent():

	for episode in range(1, MAX_EPISODES+1):
	    
	    loss, train_reward = train(train_env, policy, optimizer, DISCOUNT_FACTOR)
	    
	    test_reward = evaluate(test_env, policy)
	    
	    train_rewards.append(train_reward)
	    test_rewards.append(test_reward)
	    
	    mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
	    mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
	    
	    if episode % PRINT_EVERY == 0:
	    
	        print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
	    
	    if mean_test_rewards >= REWARD_THRESHOLD:
	        
	        print(f'Reached reward threshold in {episode} episodes')
	        
	        break

	return policy