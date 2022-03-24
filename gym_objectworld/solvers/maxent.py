"""
Implements maximum entropy inverse reinforcement learning (Ziebart et al., 2008)

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

from itertools import product

import numpy as np
import numpy.random as rn
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

from gym_objectworld.solvers import value_iteration_objectworld as V
from gym_objectworld.solvers import optimizer as O

def normalize(vals):
  """
  normalize to (0, max_val)
  input:
    vals: 1d array
  """
  min_val = np.min(vals)
  max_val = np.max(vals)
  return (vals - min_val) / (max_val - min_val)

def irl(env, gamma, trajectories, learning_rate, eps=1e-4):
    """
    Find the reward function for the given trajectories.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    n_actions: Number of actions A. int.
    gamma: gamma factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    epochs: Number of gradient descent steps. int.
    learning_rate: Gradient descent learning rate. float.
    -> Reward vector with shape (N,).
    """

    feature_matrix = env._feature_matrix()
    n_states, d_states = feature_matrix.shape

    # Initialise weights.
    alpha = rn.uniform(size=(d_states,))
    # Calculate the feature expectations \tilde{phi}.
    feature_expectations = find_feature_expectations(env,
                                                     trajectories)
    # Gradient descent on alpha.
    delta = np.inf
    while delta > eps:
        alpha_old = alpha.copy()

        r = feature_matrix.dot(alpha)
        expected_svf = find_expected_svf(env, r, gamma, trajectories)

        grad = feature_expectations - feature_matrix.T.dot(expected_svf)

        alpha += (learning_rate*grad)
        
        delta = np.max(np.abs(alpha_old - alpha))

    return feature_matrix.dot(alpha).reshape((n_states,))

def find_svf(n_states, trajectories):
    """
    Find the state visitation frequency from trajectories.

    n_states: Number of states. int.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> State visitation frequencies vector with shape (N,).
    """

    svf = np.zeros(n_states)

    for trajectory in trajectories:
        for i in range(len(trajectory.transistions())):
            svf[t.transistions()[i][0]] += 1

    svf /= trajectories.shape[0]

    return svf

def find_feature_expectations(env, trajectories):
    """
    Find the feature expectations for the given trajectories. This is the
    average path feature vector.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Feature expectations vector with shape (D,).
    """

    _, feature_expectations_size = env._feature_matrix().shape
    feature_expectations = np.zeros(feature_expectations_size)
    for trajectory in trajectories:
        for i in range(len(trajectory.transitions())):
            feature_expectations += env._feature_matrix()[trajectory.transitions()[i][0]]

    feature_expectations /= len(trajectories)

    return feature_expectations

def find_expected_svf(env, r, gamma, trajectories):
    """
    Find the expected state visitation frequencies using algorithm 1 from
    Ziebart et al. 2008.

    n_states: Number of states N. int.
    alpha: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    gamma: gamma factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Expected state visitation frequencies vector with shape (N,).
    """

    n_trajectories = len(trajectories)
    trajectory_length = 8

    n_states, _ = env._feature_matrix().shape
    n_actions = env.action_space.n
    transition_probability = V.convert_transition_array(env)

    # policy = find_policy(n_states, r, n_actions, gamma,
    #                                 transition_probability)
    policy = V.find_policy(env, r, gamma)

    start_state_count = np.zeros(n_states)
    for trajectory in trajectories:
        start_state_count[trajectory.transitions()[0][0]] += 1
    p_start_state = start_state_count/n_trajectories

    expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[k, t] += (expected_svf[i, t-1] *
                                  policy[i, j] * # Stochastic policy
                                  transition_probability[i, j, k])

    return expected_svf.sum(axis=1)

def softmax(x1, x2):
    """
    Soft-maximum calculation, from algorithm 9.2 in Ziebart's PhD thesis.

    x1: float.
    x2: float.
    -> softmax(x1, x2)
    """

    max_x = max(x1, x2)
    min_x = min(x1, x2)
    return max_x + np.log(1 + np.exp(min_x - max_x))
