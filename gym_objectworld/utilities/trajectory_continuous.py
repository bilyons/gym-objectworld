"""
Trajectories representing expert demonstrations and automated generation
thereof.
"""

import numpy as np
from itertools import chain


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

    t=0
    while not done:
        
        action = model.select_action(state)

        new_state, _, done, _ = env.step(action)

        trajectory += [(state, action, new_state)]

        state = new_state
        t+= 1

        if t==10:
            done = True

    return Trajectory(trajectory)

def generate_trajectories(n, env, model):

    def _generate_one():
        return generate_trajectory(env, model)

    return (_generate_one() for _ in range(n))

def vector_field(trajectories):
    div_array = []
    initial_array = []
    # For the vector field of continuous, loop over all trajectories
    for t in trajectories:
        for i in range(len(t.transitions())):
        # For each transition in the trajectory, action is +1 or 0
            div_array.append(t.transitions()[i][2] - t.transitions()[i][0])
            initial_array.append(t.transitions()[i][0])

    div_array = np.array(div_array)
    initial_array = np.array(initial_array)
    return initial_array, div_array 