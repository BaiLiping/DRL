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

N = 1000
nIterations = 200
nActions = mdp.nActions
avgRewardsEarned = np.zeros((3, nIterations))
# Test epsilon greedy strategy
sumRewardsEarned = np.zeros(nIterations)
sumEmpiricalMeans = np.zeros(nActions)
for i in range(N):
    [empiricalMeans, rewardsEarned] = banditProblem.epsilonGreedyBandit(nIterations=nIterations)
    sumRewardsEarned = sumRewardsEarned + rewardsEarned
    sumEmpiricalMeans = sumEmpiricalMeans + empiricalMeans
avgRewardsEarned[0] = sumRewardsEarned / N
avgEmpiricalMeans = sumEmpiricalMeans / N
print "\nepsilonGreedyBandit results"
print avgEmpiricalMeans


# Test Thompson sampling strategy
sumRewardsEarned = np.zeros(nIterations)
sumEmpiricalMeans = np.zeros(nActions)
for i in range(N):
    [empiricalMeans, rewardsEarned] = banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions,2]),nIterations=nIterations)
    sumRewardsEarned = sumRewardsEarned + rewardsEarned
    sumEmpiricalMeans = sumEmpiricalMeans + empiricalMeans
avgRewardsEarned[1] = sumRewardsEarned / N
avgEmpiricalMeans = sumEmpiricalMeans / N
print "\nthompsonSamplingBandit results"
print avgEmpiricalMeans

# Test UCB strategy
sumRewardsEarned = np.zeros(nIterations)
sumEmpiricalMeans = np.zeros(nActions)
for i in range(N):
    [empiricalMeans, rewardsEarned] = banditProblem.UCBbandit(nIterations=200)
    sumRewardsEarned = sumRewardsEarned + rewardsEarned
    sumEmpiricalMeans = sumEmpiricalMeans + empiricalMeans
avgRewardsEarned[2] = sumRewardsEarned / N
avgEmpiricalMeans = sumEmpiricalMeans / N
print "\nUCBbandit results"
print avgEmpiricalMeans

plt.plot(avgRewardsEarned[0], label='Epsilon-greedy')
plt.plot(avgRewardsEarned[1], label='Thompson sampling')
plt.plot(avgRewardsEarned[2], label='UCB', color='y')
plt.title('Multi-armed Bandit Problem')
plt.xlabel('iteration #')
plt.ylabel('average (based on 1000 trials) \n of the reward earned at each iteration')
plt.legend(loc='lower right')
plt.show()





