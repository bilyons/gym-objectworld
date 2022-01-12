# Import the envs module so that envs register themselves
import gym_objectworld.envs
from gym.envs.registration import register
register(
    id='objectworld-gridworld-v0',
    entry_point='gym_objectworld.envs:GridWorldEnv',
)
register(
    id='objectworld-v0',
    entry_point='gym_objectworld.envs:ObjectWorldEnv',
)