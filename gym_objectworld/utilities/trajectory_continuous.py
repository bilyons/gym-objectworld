"""
Trajectories representing expert demonstrations and automated generation
thereof.
"""

import numpy as np
from itertools import chain
from .rbf import RBFs as R
import math

class Trajectory:
    """
    A trajectory consisting of states, corresponding actions, and outcomes.
    Args:
        transitions: The transitions of this trajectory as an array of
            tuples `(state_from, action, state_to)`. Note that `state_to` of
            an entry should always be equal to `state_from` of the next
            entry.
    """
    def __init__(self, transitions):
        self._t = transitions

    def transitions(self):
        """
        The transitions of this trajectory.
        Returns:
            All transitions in this trajectory as array of tuples
            `(state_from, action, state_to)`.
        """
        return self._t

    def states(self):
        """
        The states visited in this trajectory.
        Returns:
            All states visited in this trajectory as iterator in the order
            they are visited. If a state is being visited multiple times,
            the iterator will return the state multiple times according to
            when it is visited.
        """
        return map(lambda x: x[0], chain(self._t, [(self._t[-1][2], 0, 0)]))

    def states_actions(self):
        """
        The states visited in this trajectory.
        Returns:
            All states visited in this trajectory as iterator in the order
            they are visited. If a state is being visited multiple times,
            the iterator will return the state multiple times according to
            when it is visited.
        """
        return map(lambda x: x[0:2], chain(self._t, [(self._t[-1][2], 0, 0)]))

    def __repr__(self):
        return "Trajectory({})".format(repr(self._t))

    def __str__(self):
        return "{}".format(self._t)

    def __len__(self):
        length = 0
        for s in self.states():
            length+=1
        return length


def generate_trajectory(env, model):

    trajectory = []

    done = False
    state = env.reset()
    d_state = np.array([ (np.arctan2(state[1], state[0])), state[2] ] )
    t=0
    while not done:
        

        action = model.select_action(state)

        new_state, _, done, _ = env.step(action)

        n_d_state = np.array([np.arctan2(new_state[1], new_state[0]), new_state[2]])

        trajectory += [(d_state, action, n_d_state)]

        state = new_state

        d_state = n_d_state

        t+= 1

    return Trajectory(trajectory)

def generate_trajectories(n, env, model):

    def _generate_one():
        return generate_trajectory(env, model)

    return (_generate_one() for _ in range(n))


def generate_trajectory_objectworld(env, policy):

    trajectory = []

    done = False
    state = env.reset()
    check = 0
    check2 = 0
    while not done:
        
        conv_state = (env.agent_pos[0]-1)*(env.grid_size-2) + (env.agent_pos[1]-1)

        action = np.random.choice(range(env.action_space.n), p=policy[conv_state,:])

        new_state, _, done, _ = env.step(action)

        conv_new_state = (new_state[0]-1)*(env.grid_size-2) + (new_state[1]-1)

        trajectory += [(state, action, new_state)]

        state = new_state

    return Trajectory(trajectory)

def generate_trajectories_objectworld(n, world, policy):

    def _generate_one():
        return generate_trajectory_objectworld(world, policy)

    return (_generate_one() for _ in range(n))

def vector_field(env, trajectories):

    # Construct RBFs over space
    # g_rbfs = R(env, 100)

    # Stack vectors
    c = 0
    for t in trajectories:
        for i in range(len(t.transitions())):
            initial_state = t.transitions()[i][0]

            transition = t.transitions()[i][2] - t.transitions()[i][0]
            if c==0:
                state_list = initial_state
                transition_list = transition
            else:
                state_list = np.vstack((state_list, initial_state))
                transition_list = np.vstack((transition_list, transition))
            c+=1

    # centres = np.vstack(g_rbfs.centres)
    # vectors = np.matmul(g_rbfs._cal_activation(state_list).T, transition_list)
    return state_list, transition_list