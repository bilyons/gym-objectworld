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
    d_state = env.state
    t=0
    while not done:
        

        action = model.select_action(state)

        new_state, _, done, _ = env.step(action)

        n_d_state = env.state

        trajectory += [(d_state, action, n_d_state)]

        state = new_state

        d_state = n_d_state

        t+= 1

    return Trajectory(trajectory)

def generate_trajectories(n, env, model):

    def _generate_one():
        return generate_trajectory(env, model)

    return (_generate_one() for _ in range(n))

# def vector_field(trajectories):

#     c = 0
#     for t in trajectories:
#         for i in range(len(t.transitions())):
#             initial_state = t.transitions()[i][0]
#             if c == 0:
#                 state_list = initial_state
#             else:
#                 state_list = np.vstack((state_list, initial_state))
#             c+=1
#     x_abs = abs(max(state_list[:,0], key = abs))
#     x_dot_abs = abs(max(state_list[:,1], key = abs))
#     th_abs = abs(max(state_list[:,2], key = abs))
#     th_dot_abs = abs(max(state_list[:,3], key = abs))


#     h_range = np.array((x_abs, x_dot_abs, th_abs, th_dot_abs))
#     print(h_range)
#     l_range = -h_range
#     t_range = h_range - l_range
#     n_states = (t_range)*\
#                         np.array([20,20,200,20])

#     n_states = np.round(n_states, 0).astype(int)+2

#     boxes = np.zeros((n_states[0], n_states[1], n_states[2], n_states[3], 4))
#     counts = np.zeros((n_states[0], n_states[1], n_states[2], n_states[3], 1))

#     # For the vector field of continuous, loop over all trajectories
#     for t in trajectories:
#         for i in range(len(t.transitions())):
#             initial_state = t.transitions()[i][0]
#             disc_i_s = (initial_state-l_range)*\
#                             np.array([20,20,20,20])

#             disc_i_s = np.round(disc_i_s, 0).astype(int)+1

#             vector = t.transitions()[i][2] - t.transitions()[i][0]

#             boxes[disc_i_s[0], disc_i_s[1], disc_i_s[2], disc_i_s[3], :] += vector
#             counts[disc_i_s[0], disc_i_s[1], disc_i_s[2], disc_i_s[3], :] += 1

#     vector_array = np.divide(boxes, counts, out=np.zeros_like(boxes), where=counts!=0)

#     h = t_range/[20,20,200,20]
#     return vector_array, h

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