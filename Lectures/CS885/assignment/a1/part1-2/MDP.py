import numpy as np

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
    
    
    
    def actionValue(self, V):
        Q = np.zeros((self.nActions, self.nStates))
        for a in range(self.nActions):
            #Q[a] = (self.T[a]).dot(self.R[a] + V * self.discount) # reward in next state
            #Q[a] = self.R[a] + (self.T[a]).dot(V * self.discount) # reward in current state
            for s in range(self.nStates):
                Q[a,s] = (self.T[a,s]).dot(self.R[a,s] + V * self.discount) # R[a,s] | R[a]
        return Q
    
    
    
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        V = initialV
        iterId = 0
        epsilon = np.inf
        
        while (epsilon > tolerance and iterId < nIterations):
            V_old = np.copy(V)
            Q = self.actionValue(V)
            V = Q.max(axis=0)
            epsilon = (np.fabs(V-V_old)).max()
            iterId = iterId + 1
            #print (V)
            #print epsilon
        return [V,iterId,epsilon]
    
    
    
    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''
        
        Q = self.actionValue(V)
        policy = np.argmax(Q, axis=0)
        #print (Q)
        return policy
    
    
    def policySelect(self, T, policy):
        Tpi = np.copy(T[0])
        for s in range(self.nStates):
	        Tpi[s] = T[policy[s], s]
        return Tpi
    
    
    def policyT(self, policy):
        return self.policySelect(self.T, policy)
    
    
    def policyR(self, policy):
        return self.policySelect(self.R, policy)
        
        
    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        #V = (I - T*d)^-1 * R
        Tpi = self.policyT(policy)
        Rpi = self.policyR(policy)
        I = np.eye(self.nStates)
        V = np.linalg.inv(I - Tpi * self.discount).dot(Rpi)
        return V
    
    
    
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        policy = initialPolicy
        V = np.zeros(self.nStates)
        iterId = 0
        stable = False
        
        while ((not stable) and iterId < nIterations):
            policyOld = np.copy(policy)
            V = self.evaluatePolicy(policy)
            #print (V)
            policy = self.extractPolicy(V)
            #print (policy)
            
            iterId = iterId + 1
            stable = np.array_equal(policyOld,policy)
        return [policy,V,iterId]
    
    
    
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        V = initialV
        iterId = 0
        epsilon = np.inf
        
        while (epsilon > tolerance and iterId < nIterations):
            V_old = np.copy(V)
            Q = self.actionValue(V) # not efficient. slow when action space is large. should use policyselect(T) and policyselect(R). #TODO
            V = self.policySelect(Q, policy) # same as value iteration except use policy instead of max
            epsilon = (np.fabs(V-V_old)).max()
            iterId = iterId + 1
        return [V,iterId,epsilon]
    
    
    # increase nEvalIterations if action space is large
    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        policy = initialPolicy
        V = np.zeros(self.nStates)
        iterId = 0
        epsilon = np.inf
        stable = False
        
        while ((not stable) and iterId < nIterations and epsilon > tolerance):
            policyOld = np.copy(policy)
            # same as policy iteration except evaluate partially
            [V,_,_] = self.evaluatePolicyPartially(policy,V,nEvalIterations,tolerance)
            #print (V)
            policy = self.extractPolicy(V)
            #print (policy)
            
            iterId = iterId + 1
            # eith stop when policy is stable or when V converges
            # stop when stable only with nEvalIterations = inf and tolerance is small
            #stable = np.array_equal(policyOld,policy) 
            Q = self.actionValue(V)
            V2 = Q.max(axis=0)
            epsilon = (np.fabs(V2-V)).max()
        
        [V,_,_] = self.evaluatePolicyPartially(policy,V,np.inf,tolerance)
        return [policy,V,iterId,epsilon]
    
    
    
    
    
    
