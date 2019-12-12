from MDP import *

''' Construct simple MDP as described in Lecture 2a Slides 13-14'''
# Transition function: |A| x |S| x |S'| array
# 0 is A | 1 is S
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
# Reward function: |A| x |S| array
R = np.array([[0,0,10,10],[0,0,10,10]])
# Discount factor: scalar in [0,1)
discount = 0.9        
# MDP object
mdp = MDP(T,R,discount)
#print ([T, R, discount])

'''Test each procedure'''
[V,iterId,epsilon] = mdp.valueIteration(initialV=(np.zeros(mdp.nStates)+10), nIterations=np.inf)
print (V)
print (iterId,epsilon)

policy = mdp.extractPolicy(V)
print (policy)

V = mdp.evaluatePolicy(np.array([0,1,1,1]))
print (V)

[policy,V,iterId] = mdp.policyIteration(np.array([1,0,0,0]), nIterations=10)
print (V)
print (policy,iterId)

[V,iterId,epsilon] = mdp.evaluatePolicyPartially(np.array([0,1,1,1]),np.array([0,10,0,13]))
print (V)
print (iterId,epsilon)

[policy,V,iterId,epsilon] = mdp.modifiedPolicyIteration(np.array([1,0,1,0]),np.array([0,10,0,13]), nEvalIterations=5,nIterations=np.inf,tolerance=0.01)
print (V)
print (policy,iterId,epsilon)


