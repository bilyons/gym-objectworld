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
        if t.transitions()[-1][2] == 80:
            t1+=1
        else:
            t2+=1
    print(t1, t2)

def movement_calc(env, trajectories):
    size = np.int(np.sqrt(env.observation_space.n))
    in_array = np.zeros((size, size, env.action_space.n))
    out_array = np.zeros((size, size, env.action_space.n))

    for t in trajectories:
        for i in range(len(t.transitions())):

            old_x, old_y = t.transitions()[i][0]%size, t.transitions()[i][0]//size
            new_x, new_y = t.transitions()[i][2]%size, t.transitions()[i][2]//size
            if t.transitions()[i][2] - t.transitions()[i][0] == size:
                # I went up
                out_array[old_y, old_x,:] +=[0,0,0,1]
                in_array[new_y, new_x,:] += [0,1,0,0]                

            elif t.transitions()[i][2] - t.transitions()[i][0] == -size:
                # I went down
                out_array[old_y, old_x,:] +=[0,1,0,0]
                in_array[new_y, new_x,:] += [0,0,0,1]
            elif t.transitions()[i][2] - t.transitions()[i][0] == 1:
                # I went right
                out_array[old_y, old_x,:] +=[0,0,1,0]
                in_array[new_y, new_x,:] += [1,0,0,0]
            elif t.transitions()[i][2] - t.transitions()[i][0] == -1:
                # I went left
                out_array[old_y, old_x,:] +=[1,0,0,0]
                in_array[new_y, new_x,:] += [0,0,1,0]
            elif t.transitions()[i][2] - t.transitions()[i][0] == 0:
                # I went nowhere
                out_array[old_y, old_x,:] +=[0,0,0,0]
                in_array[new_y, new_x,:] += [0,0,0,0]
            else:
                print("Movement error")
                exit()
    return in_array - out_array, out_array#-in_array
