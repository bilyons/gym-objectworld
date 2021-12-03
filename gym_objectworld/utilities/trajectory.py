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


def generate_trajectory(env, policy):
    trajectory = []

    done = False
    state = env.reset()
    check = 0
    check2 = 0
    while not done:

        action = np.random.choice(range(env.action_space.n), p=policy[state,:])

        new_state, _, done, _ = env.step(action)

        trajectory += [(state, action, new_state)]

        state = new_state

    return Trajectory(trajectory)

def generate_trajectories(n, world, policy):

    def _generate_one():
        return generate_trajectory(world, policy)

    return (_generate_one() for _ in range(n))
def check_terminal_ratio(trajectories):
    t1 = 0
    t2 = 0
    for t in trajectories:
        if t.transitions()[-1][2] == 24:
            t1+=1
        else:
            t2+=1
    print(t1, t2)

def in_out_calc_it_all_about(env, trajectories):
    in_array = np.zeros((env.observation_space.n))
    out_array = np.zeros((env.observation_space.n))

    size = np.int(np.sqrt(env.observation_space.n))
    other_in = np.zeros((size, size,2))
    other_out = np.zeros((size, size,2))
    for t in trajectories:
        for i in range(len(t.transitions())):
            in_array[t.transitions()[i][2]] += 1
            out_array[t.transitions()[i][0]] += 1
            # print(t.transitions()[i][2])
            # print(t.transitions()[i][2]%size, t.transitions()[i][2]//size)
            if np.abs(t.transitions()[i][2] - t.transitions()[i][0]) > 1:
                x, y = t.transitions()[i][2]%size, t.transitions()[i][2]//size
                if np.abs(t.transitions()[i][2] - t.transitions()[i][0]) < 0:
                    other_in[y, x,:] += [0,-1]
                else:
                    other_in[y, x,:] += [0,1]
            else:
                x, y = t.transitions()[i][2]%size, t.transitions()[i][2]//size
                if np.abs(t.transitions()[i][2] - t.transitions()[i][0]) < 0:
                    other_in[y, x,:] += [-1,0]
                else:
                    other_in[y, x,:] += [1,0]

            if np.abs(t.transitions()[i][0] - t.transitions()[i][2]) > 1:
                x, y = t.transitions()[i][0]%size, t.transitions()[i][0]//size
                if np.abs(t.transitions()[i][0] - t.transitions()[i][2]) < 0:
                    other_out[y, x,:] += [0,-1]
                else:
                    other_out[y, x,:] += [0,1]
            else:
                x, y = t.transitions()[i][0]%size, t.transitions()[i][0]//size
                if np.abs(t.transitions()[i][0] - t.transitions()[i][2]) < 0:
                    other_out[y, x,:] += [-1,0]
                else:
                    other_out[y, x,:] += [1,0]

    return in_array - out_array, other_in-other_out