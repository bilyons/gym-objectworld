import gym, os
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import namedtuple

class DQN(nn.Module):

	super(DQN, self).__init__()
	self.fc = nn