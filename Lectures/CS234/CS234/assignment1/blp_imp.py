### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
import time
from lake_envs import *
from gym.envs.toy_text import FrozenLakeEnv

np.set_printoptions(precision=3)


def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
    V = np.zeros(nS)
    old_V=np.zeros(nS)
    Policy = np.zeros(nS, dtype=int)#actually the policy should be a distibution, here to make things easy for myself
    idx = 1
    V_Action=np.zeros(nA)
    while idx<=max_iteration or np.sum(np.sqrt(np.square(old_V-V)))>tol:
        idx += 1
        old_V=V.copy()
        for state in range(nS):
            for action in range(nA):
                prob,new_state,reward,termination=P[state][action][0]
                V_Action[action]=reward+gamma*prob*old_V[new_state]
                    
            V[state] = max(V_Action)
            Policy[state] = np.argmax(V_Action)
    return V, Policy






def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=200, tol=1e-3):


    def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=100, tol=1e-3):
        V = np.zeros(nS)
        old_V = np.zeros(nS)
        i=1
            while i<=max_iteration or np.sum(np.sqrt(np.square(V-old_V)))>tol:
            i += 1
            old_V=V.copy()
            for state in range(nS):
                prob,new_state,reward,termination = P[state][policy[state]][0]
                V[state]=reward+old_V[new_state]*prob*gamma
        return V
    

    def policy_improvement(P, nS, nA, policy, gamma=0.9):
        q_function = np.zeros([nS,nA])
        value_from_policy=policy_evaluation(P, nS, nA, policy)
        for state in range(nS):
            for action in range(nA):
                (prob, new_state, reward, terminal) = P[state][action][0]
                q_function[state][action] = reward+(gamma*prob*value_from_policy[new_state])
        new_policy = np.argmax(q_function, axis=1)
        return new_policy

    i = 0 
    old_policy= policy.copy()
    while i<=max_iteration or np.sum(np.sqrt(np.square(old_policy-policy)))>tol:
        i += 1
        old_policy=policy
        policy = policy_improvement(P, nS, nA, old_policy)
    V=policy_evaluation(P,nS,nA,policy)
    return V, policy



def example(env):
    """Show an example of gym
    Parameters
    	----------
    env: gym.core.Environment
        Environment to play on. Must have nS, nA, and P as
        attributes.
    """
    env.seed(0); 
    from gym.spaces import prng; prng.seed(10) # for print the location
    # Generate the episode
    ob = env.reset()
    for t in range(100):
        env.render()
        a = env.action_space.sample()
        ob, rew, done, _ = env.step(a)
        if done:
            break
    assert done
    env.render();

def render_single(env, policy):
    """Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
        Environment to play on. Must have nS, nA, and P as
        attributes.
    Policy: np.array of shape [env.nS]
        The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for ep in range(5000):
        for t in range(200):
        #env.render()
        #time.sleep(0.5) # Seconds between frames. Modify as you wish.
            a = policy[ob]
            ob, rew, done, _ = env.step(a)
            episode_reward += rew
            if done:
                print("Episode{:.5f} steps{},reward{}".format(ep,t,episode_reward))
                ob=env.reset()
                episode_reward=0
                break
        #assert done
        #env.render()
        #env.render();


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
    #env = gym.make("Stochastic-4x4-FrozenLake-v0")
    env=FrozenLakeEnv(None,"4x4",True)
    env._max_episode_steps = 100000
    print(env.__doc__)
    env.render()
    V_vi, P_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=200, tol=1e-3)
    print(V_vi)
    print(P_vi)
    V_pi,P_pi=policy_iteration(env.P,env.nS, env.nA, gamma=0.9, max_iteration=200, tol=1e-3)
    print(V_pi)
    print(P_pi)
	
