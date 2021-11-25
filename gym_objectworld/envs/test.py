import gym

env = gym.make('gym_objectworld:objectworld-gridworld-v0')
# env = gym.make("FrozenLake-v1", desc=None)

env.render()