import numpy as np
import MDP
import RL2
import matplotlib.pyplot as plt

def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli
    
    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean: return 1
    else: return 0


# Multi-arm bandit problems (3 arms with probabilities 0.3, 0.5 and 0.7)
T = np.array([[[1]],[[1]],[[1]]])
R = np.array([[0.3],[0.5],[0.7]])
discount = 0.999
mdp = MDP.MDP(T,R,discount)
banditProblem = RL2.RL2(mdp,sampleBernoulli)

trial = 1000

# Test epsilon greedy strategy
avg_r_history = np.zeros((3,200))
for n in range(0, trial):
    
    if n % 100 == 0:
        print("trial ", n)
    
    empiricalMeans, r_history = banditProblem.epsilonGreedyBandit(nIterations=200)
    avg_r_history[0,:] = (avg_r_history[0,:]*n + r_history)/(n+1)
    #print("\nepsilonGreedyBandit results")
    #print(empiricalMeans)
    
    # Test Thompson sampling strategy
    empiricalMeans, r_history = banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions,2]),nIterations=200)
    avg_r_history[1,:] = (avg_r_history[1,:]*n + r_history)/(n+1)
    #print("\nthompsonSamplingBandit results")
    #print(empiricalMeans)


    # Test UCB strategy
    empiricalMeans, r_history = banditProblem.UCBbandit(nIterations=200)
    avg_r_history[2,:] = (avg_r_history[2,:]*n + r_history)/(n+1)
#    print("\nUCBbandit results")
#    print(empiricalMeans)

x = np.linspace(1,200,200)
fig, ax = plt.subplots()
ax.plot(x,avg_r_history[0,:], label = "EpsilonGreedy")
ax.plot(x, avg_r_history[1,:], label = "ThompsonSampling")
ax.plot(x, avg_r_history[2,:], label = "UCB")
legend = ax.legend(loc='lower right', shadow=True)
plt.xlabel("Episode")
plt.ylabel("Avg Reward")
plt.show()