# grid_world.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
#
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

#########################################################################################
# Modified by: Anshuman Sharma (anshums@g.clemson.edu)
#########################################################################################

"""
The package `matplotlib` is needed for the program to run.

The Grid World environment has discrete state and action spaces
and allows for both model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
    env.trans_model             # the transition/dynamics model

In value_iteration and policy_iteration, you can access the transition model
at a given state s and action by calling
    t = env.trans_model[s][a]
where s is an integer in the range [0, env.observation_space.n),
      a is an integer in the range [0, env.action_space.n), and
      t is a list of four-element tuples in the form of
        (p, s_, r, terminal)
where s_ is a new state reachable from the state s by taking the action a,
      p is the probability to reach s_ from s by a, i.e. p(s_|s, a),
      r is the reward of reaching s_ from s by a, and
      terminal is a boolean flag to indicate if s_ is a terminal state.

In mc_control, sarsa, q_learning, and double q-learning once a terminal state is reached,
the environment should be (re)initialized by
    s = env.reset()
where s is the initial state.
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.


A Logger instance is provided for each function, through which you can
visualize the process of the algorithm.
You can visualize the value, v, and policy, pi, for the i-th iteration by
    logger.log(i, v, pi)
You can also only update the visualization of the v values by
    logger.log(i, v)
"""


# use random library if needed
import random
import numpy as np
from numpy import linalg as LA

def value_iteration(env, gamma, max_iterations, logger):
    """
    Implement value iteration to return a deterministic policy for all states.
    See lines 20-30 for details.

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of value iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint: The value iteration may converge before reaching max_iterations.
        In this case, you want to exit the algorithm earlier. A way to check
        if value iteration has already converged is to check whether
        the infinity norm between the values before and after an iteration is small enough.
        In the gridworld environments, 1e-4 (theta parameter in the pseudocode) is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process

    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model

    v = [0] * NUM_STATES
    # values = [v] * max_iterations
    pi = [0] * NUM_STATES
    # Visualize the value and policy
    logger.log(0, v, pi)
    # At each iteration, you may need to keep track of pi to perform logging

### Please finish the code below ##############################################
###############################################################################
    prev_v = [0] * NUM_STATES                                                   # Initializing the previous norm
    prev_norm = 0
    for k in range(1, max_iterations):
        for i in range(0, NUM_STATES):
            q_array = [0] * NUM_ACTIONS
            for j in range(0, NUM_ACTIONS):
                t = TRANSITION_MODEL[i][j]
                q = 0                                                             # Initializing local variable to store the value at given state for each action
                for l in range(len(t)):
                    if not t[l][3]:
                        q += t[l][0] * (t[l][2] + gamma*prev_v[t[l][1]])
                    else:
                        q += t[l][0] * t[l][2]
                q_array[j] = q

            v[i] = np.max(q_array)                                                  # Updating the maximum value
            pi[i] = np.argmax(q_array)                                              # Updating the corresponding action
        prev_v = v
        logger.log(k, v, pi)
        if abs(LA.norm(v) - prev_norm) <= 10**-4:
            break
        prev_norm = abs(LA.norm(v))
###############################################################################
    return pi


def policy_iteration(env, gamma, max_iterations, logger):
    """
    Optional: Implement policy iteration to return a deterministic policy for all states.
    See lines 20-30 for details.

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of policy iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint 1: Policy iteration may converge before reaching max_iterations.
        In this case, you should exit the algorithm. A simple way to check
        if the algorithm has already converged is by simply checking whether
        the policy at each state hasn't changed from the previous iteration.
        Hint 2: The value iteration during policy evaluation usually converges
        very fast and policy evaluation should end upon convergence. A way to check
        if policy evaluation has converged is to check whether the infinity norm
        norm between the values before and after an iteration is small enough.
        In the gridworld environments, 1e-4 is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to record and visualize the iteration process.
        During policy evaluation, the V-values will be updated without changing the current policy;
        here you can update the visualization of values by simply calling logger.log(i, v).

    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model

    v = [0.0] * NUM_STATES
    pi = [random.randint(0, NUM_ACTIONS-1)] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

### Please finish the code below ##############################################
###############################################################################
# policy evaluation

    policy_stable = 0                                                           # Initialization

    i=0

    while policy_stable != 1 and i<max_iterations:
        v_prev = [0.0] * NUM_STATES
        prev_norm = 0
        while True:                                                             # Policy Evaluation
            for st in range(NUM_STATES):
                t = TRANSITION_MODEL[st][pi[st]]
                v[st] = 0
                for j in range(len(t)):
                    if not t[j][3]:
                        v[st] += t[j][0] * (t[j][2] + gamma*v_prev[t[j][1]])
                    else:
                        v[st] += t[j][0] * t[j][2]

            if abs(LA.norm(v) - prev_norm) <= 10**-4:
                break
            prev_norm = abs(LA.norm(v))
            v_prev = v
        policy_stable = 1
        for st in range(NUM_STATES):                                              # Policy Improvement
            old_action = pi[st]
            q_values = [0]*NUM_ACTIONS
            for action in range(NUM_ACTIONS):
                t = TRANSITION_MODEL[st][action]
                q = 0
                for l in range(len(t)):
                    if not t[l][3]:
                        q += t[l][0] * (t[l][2] + gamma*v[t[l][1]])
                    else:
                        q += t[l][0] * t[l][2]
                q_values[action] = q
            pi[st] = np.argmax(q_values)
            if old_action != pi[st]:
                policy_stable = 0
        i+=1
        logger.log(i, v, pi)
###############################################################################
    return pi

def on_policy_mc_control(env, gamma, max_iterations, logger):
    """
    Implement on-policy first visiti Monte Carlo control to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model
    and the reward function, i.e. you cannot call env.trans_model.
    Instead you need to learn policies by collecting samples using env.step
    See lines 32-42 for more details.

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (either training episodes or total steps) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.

    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n

    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    #########################

### Please finish the code below ##############################################
###############################################################################
# Note: Use max_iterations = 200
    epsilon_saturation_factor = 0.85
    min_eps = 0.05
    max_eps = eps

    q = [[0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]           # initialize q table
    for k in range(1, max_iterations):
        s = env.reset()                                                        # Resetting the environemnt for every new episode
        trajectory = []

        while True:                                                            # trajectory collection
            rand_num = random.uniform(0, 1)                                    # epsilon greedy action selection
            if rand_num > eps:
                pi[s] = np.argmax(q[s])
            else:
                pi[s] = random.randint(0, NUM_ACTIONS-1)

            s_, r, terminal, info = env.step(pi[s])
            trajectory.append((s, pi[s], r))
            s = s_
            if terminal:
                break

        for t in range(len(trajectory)):                                         # Iterating through every step of the collected trajectory
            visited = 0
            for l in range(t):                                                   # Checking for visited state action pairs
                if trajectory[t][0] == trajectory[l][0] and trajectory[t][1] == trajectory[l][1]:
                    visited = 1
                    break
            if visited == 0:
                Gt = 0
                power = 0
                for y in range(t, len(trajectory)):                               # Calculation of Gt and updating q(St, At)
                    Gt += trajectory[y][2] * gamma**power
                    power += 1

                q[trajectory[t][0]][trajectory[t][1]] = ((1-alpha)  * q[trajectory[t][0]][trajectory[t][1]]) + (alpha * Gt)
                v[trajectory[t][0]] = q[trajectory[t][0]][trajectory[t][1]]
        # eps = 0.01 + (1 - 0.01) * np.exp(-0.001*k)
        # eps = 1/k
        if (k < max_iterations * epsilon_saturation_factor):
            eps = 1 - ((max_eps - min_eps)/max_iterations)*k                        # Epsilon Decay
        print('epsilon: {}'.format(eps))
        logger.log(k, v, pi)

###############################################################################
    return pi


def sarsa(env, gamma, max_iterations, logger):
    """
    Implement SARSA to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model
    and the reward function, i.e. you cannot call env.trans_model.
    Instead you need to learn policies by collecting samples using env.step
    See lines 32-42 for more details.

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (either training episodes or total steps) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.

    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n

    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    #########################



### Please finish the code below ##############################################
###############################################################################
# Note: Use max_iterations = 200
    epsilon_saturation_factor = 0.8
    min_eps = 0.05
    max_eps = eps
    q = [[0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]
    for k in range(250):                                             # Initialize
        s = env.reset()
        rand_num = random.uniform(0, 1)                                         # Epsilon Greedy for inital state
        if(rand_num > eps):
            a = np.argmax(q[s])
        else:
            a = random.randint(0, NUM_ACTIONS-1)

        while True:
            s_, r, terminal, info = env.step(a)                                 # Taking the action

            rand_num = random.uniform(0, 1)                                     # Choosing action at the next state with epsilon-greedy
            if(rand_num > eps):
                a_next = np.argmax(q[s_])
            else:
                a_next = random.randint(0, NUM_ACTIONS-1)

            if terminal:
                q[s][a] = q[s][a] + alpha*(r - q[s][a])                         # Updating q value
                break
            q[s][a] = q[s][a] + alpha*(r + gamma*q[s_][a_next] - q[s][a])
            pi[s] = np.argmax(q[s])
            v[s] = np.max(q[s])
            # pi[s] = a
            s = s_
            a = a_next

        # eps = 0.01 + (1 - 0.01) * np.exp(-0.001*k)
        if (k < max_iterations * epsilon_saturation_factor):
            eps = 1 - ((max_eps - min_eps)/max_iterations)*k                        # Epsilon Decay
        print('epsilon: {}'.format(eps))
        logger.log(k, v, pi)


###############################################################################
    return pi


def q_learning(env, gamma, max_iterations, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model
    and the reward function, i.e. you cannot call env.trans_model.
    Instead you need to learn policies by collecting samples using env.step
    See lines 32-42 for more details.

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (either training episodes or total steps) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.

    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n

    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    #########################



### Please finish the code below ##############################################
###############################################################################
# Note: Use max_iterations = 200
    epsilon_saturation_factor = 0.9
    min_eps = 0.05
    max_eps = eps
    q = [[0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]
    for k in range(200):
        s = env.reset()
        while True:
            rand_num = random.uniform(0, 1)
            if rand_num > eps:                                                  # epsilon greedy action selection
                a = np.argmax(q[s])
            else:
                a = random.randint(0, NUM_ACTIONS-1)

            s_, r, terminal, info = env.step(a)                                 # Taking action a
            if terminal:
                q[s][a] = q[s][a] * (1-alpha) + alpha * r
                break

            q[s][a] = q[s][a] * (1-alpha) + alpha *(r + gamma*q[s_][np.argmax(q[s_])])  # Updating the q values
            v[s] = q[s][a]
            pi[s] = np.argmax(q[s])
            v[s] = np.max(q[s])                                           # Updating the greedy policy
            s = s_

        if (k < max_iterations * epsilon_saturation_factor):
            eps = 1 - ((max_eps - min_eps)/max_iterations)*k
        print('epsilon: {}'.format(eps))
        logger.log(k, v, pi)
###############################################################################
    return pi

def double_q_learning(env, gamma, max_iterations, logger):
    """
    Implement double Q-learning to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model
    and the reward function, i.e. you cannot call env.trans_model.
    Instead you need to learn policies by collecting samples using env.step
    See lines 32-42 for more details.

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (either training episodes or total steps) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.

    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n

    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    #########################



### Please finish the code below ##############################################
###############################################################################
# Note: Use max_iterations = 200
    epsilon_saturation_factor = 0.8
    min_eps = 0.05
    max_eps = eps
    q_a = [[0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]              # Initialize Qa and Qb
    q_b = [[0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]
    for k in range(200):                                                 # For every episode
        s = env.reset()

        while True:                                                                 # For every step
            rand_num = random.uniform(0, 1)
            if(rand_num > eps):                                                     # epsilon-greedy
                a = np.argmax(np.add(q_a[s], q_b[s])/2)
            else:
                a = random.randint(0, NUM_ACTIONS-1)
            s_, r, terminal, info = env.step(a)

            rand_coin = random.uniform(0, 1)                                         # Random selection between A and B for update
            if rand_coin > 0.5:
                if terminal:
                    q_a[s][a] = q_a[s][a] + alpha * (r - q_a[s][a])
                    break
                a_star = np.argmax(q_a[s_])
                q_a[s][a] = q_a[s][a] + alpha * (r + gamma*q_b[s_][a_star] - q_a[s][a])

            else:
                if terminal:
                    q_b[s][a] = q_b[s][a] + alpha * (r - q_b[s][a])
                    break
                b_star = np.argmax(q_b[s_])
                q_b[s][a] = q_b[s][a] + alpha * (r + gamma*q_a[s_][b_star] - q_b[s][a])

            q = np.add(q_a[s], q_b[s])/2
            pi[s] = np.argmax(q)                                                        # Returning the greedy policy
            v[s] = np.max(q)
            s = s_

        if (k < max_iterations * epsilon_saturation_factor):
            eps = 1 - ((max_eps - min_eps)/max_iterations)*k
        print('epsilon: {}'.format(eps))
        logger.log(k, v, pi)
###############################################################################
    return pi


if __name__ == "__main__":
    from app.grid_world import App
    import tkinter as tk

    algs = {
        "Value Iteration": value_iteration,
        "Policy Iteration": policy_iteration,
        "On-policy MC Control": on_policy_mc_control,
        "SARSA": sarsa,
        "Q-Learning": q_learning,
        "Double Q-Learning": double_q_learning
   }
    worlds = {
        # o for obstacle
        # s for start cell
        "world1": App.DEFAULT_WORLD,
        "world2": lambda : [
            ["_", "_", "_", "_", "_"],
            ["s", "_", "_", "_", 1],
            [-100, -100, -100, -100, -100],
        ],
        #"world2": lambda : [
        #    [10, "s", "s", "s", 1],
        #    [-10, -10, -10, -10, -10],
        #],
        "world3": lambda : [
            ["_", "_", "_", "_", "_"],
            ["_", "o", "_", "_", "_"],
            ["_", "o",   1, "_",  10],
            ["s", "_", "_", "_", "_"],
            [-10, -10, -10, -10, -10]
        ]
    }

    root = tk.Tk()
    App(algs, worlds, root)
    tk.mainloop()
