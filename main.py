import random
import numpy as np
import gym
# Test specifically importing a specific environment
from gym_objectworld.envs import ObjectWorldEnv

env = ObjectWorldEnv(15, 50, 4, 0.3, False)

print(env.observation_space)