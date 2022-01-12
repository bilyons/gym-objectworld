import numpy as np
from scipy.special import logsumexp as sp_lse

def convert_transition_array(env):
    transition_array = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))

    for s_1 in range(env.observation_space.n):
        for a in range(env.action_space.n):
            for pos in range(len(env.P[s_1][a])):
                dest = env.P[s_1][a][pos][1]
                transition_array[s_1, a, dest] += env.P[s_1][a][pos][0]
    return transition_array

def value_iteration(p, reward, discount, eps=1e-3):
    t = convert_transition_array(p)
    n_states, n_actions, _ = t.shape
    v = np.zeros(n_states)
    q = np.zeros((n_states, n_actions))
    delta = np.inf

    while delta>eps:
        v_old = v.copy()
        for s in range(n_states):
            for a in range(n_actions):
                # print(reward+discount*v_old)
                q[s,a] = np.dot(t[s,a,:], reward+discount*v_old)
            v[s] = sp_lse(q[s,:])
        delta = np.max(np.abs(v_old - v))
    return v, t

def find_policy(p, reward, discount, eps=1e-3):

    v, t = value_iteration(p, reward, discount)

    n_states, n_actions, _ = t.shape

    Q = np.zeros((n_states, n_actions))

    y = [np.matrix(t[:,a,:]) for a in range(n_actions)]
    for a in range(n_actions):
        Q[:,a] = y[a].dot(reward + discount*v)
    Q -= Q.max(axis=1).reshape(n_states, 1)
    Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape(n_states, 1)

    return Q

def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-3):
    """
    Find the optimal value function.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """

    v = np.zeros(n_states)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            max_v = float("-inf")
            for a in range(n_actions):
                tp = transition_probabilities[s, a, :]
                max_v = max(max_v, np.dot(tp, reward + discount*v))

            new_diff = abs(v[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v

    return v