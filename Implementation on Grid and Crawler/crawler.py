# crawler.py
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
In this file, you should test your Q-learning implementation on the robot crawler environment 
that we saw in class. It is suggested to test your code in the grid world environments before this one.

The package `matplotlib` is needed for the program to run.


The Crawler environment has discrete state and action spaces
and provides both of model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
  

Once a terminal state is reached the environment should be (re)initialized by
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
"""


# use random library if needed
import random
import numpy as np

def sarsa(env, logger):
    """
    Implement SARSA to return a deterministic policy for all states.

    Parameters
    ----------
    env: CrawlerEnv
        the environment
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
    gamma = 0.95   

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    # maximum number of training iterations
    max_iterations = 5000
    #########################

### Please finish the code below ##############################################
###############################################################################
    v = [0 for _ in range(NUM_STATES)] 
    pi = [0] * NUM_STATES
    epsilon_saturation_factor = 0.8
    min_eps = 0.01
    max_eps = eps
    q = [[0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]
    max_iterations = 10000                                                 # max_iterations is just used as a threshold for steps, the program does not terminate at this

    # Initialize
    s = env.reset()
    rand_num = random.uniform(0, 1)                                         # Epsilon Greedy for inital state
    if(rand_num > eps):
        a = np.argmax(q[s])
    else:
        a = random.randint(0, NUM_ACTIONS-1)
    step = 0
    while True:
        s_, r, terminal, info = env.step(a)                                 # Taking the action
        
        rand_num = random.uniform(0, 1)                                     # Choosing action at the next state with epsilon-greedy
        if(rand_num > eps):
            a_next = np.argmax(q[s_])
        else:
            a_next = random.randint(0, NUM_ACTIONS-1)
            
        q[s][a] = q[s][a] + alpha*(r + gamma*q[s_][a_next] - q[s][a])
        
        pi[s] = np.argmax(q[s])
        v[s] = np.max(q[s])
        s = s_
        a = a_next
        step+=1

        if (step < max_iterations * epsilon_saturation_factor):
            eps = 1 - ((max_eps - min_eps)/max_iterations)*step                       # Epsilon Decay
        else:
            eps = 0.01
        
        print('epsilon: {}'.format(eps))
        logger.log(step, v, pi)
###############################################################################
    return pi

def q_learning(env, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.

    Parameters
    ----------
    env: CrawlerEnv
        the environment
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
    gamma = 0.95   

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    # maximum number of training iterations
    max_iterations = 5000
    #########################

### Please finish the code below ##############################################
###############################################################################
    epsilon_saturation_factor = 0.9
    min_eps = 0.05
    max_eps = eps
    v = [0] * NUM_STATES
    q = [[0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]
    pi = [0] * NUM_STATES
    s = env.reset()
    step = 0
    max_iterations = 10000                                                 # max_iterations is just used as a threshold for steps, the program does not terminate at this
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
    
        if (step < max_iterations * epsilon_saturation_factor):
            eps = 1 - ((max_eps - min_eps)/max_iterations)*step                       # Epsilon Decay
        else:
            eps = 0.01
        
        step += 1
        print('epsilon: {}'.format(eps))
        logger.log(step, v, pi)
###############################################################################
    return pi


if __name__ == "__main__":
    from app.crawler import App
    import tkinter as tk
    
    algs = {
        "Q Learning": q_learning,
         "SARSA": sarsa
    }

    root = tk.Tk()
    App(algs, root)
    root.mainloop()