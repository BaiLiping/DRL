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
        
        assert initialV.shape == (self.nStates,), "Invalid initial values"
        V = initialV
        iterId = 0
        epsilon = 0
        
        V = np.amax(self.R, axis=0)
        iterId = iterId + 1
        
        epsilon = max(np.absolute(V))
        while epsilon > tolerance:
            V_all_actions = self.R + self.discount*self.T.dot(V)
            V_new = np.amax(V_all_actions, axis=0)
            epsilon = max(np.absolute(V_new - V))
            V = V_new
#            print(V)
            iterId = iterId + 1
            if iterId > nIterations:
                break;
        
        return [V,iterId,epsilon]

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''
        
        policy = np.argmax(self.R + self.discount*self.T.dot(V), axis=0)
            
        return policy 

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''
        
        V = np.zeros(self.nStates)

        T_pi = np.zeros((self.nStates,self.nStates))
        R_pi = np.zeros(self.nStates)
        
        for i in range(0,self.nStates):
            T_pi[i] = self.T[policy[i]][i]
            R_pi[i] = self.R[policy[i]][i]
        
        delta_V = np.ones(self.nStates)
        while max(delta_V) > 0.0001:
            V_new = R_pi + self.discount*T_pi.dot(V)
            delta_V = np.absolute(V_new-V)
            V = V_new
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

        V = np.zeros(self.nStates)
        
        assert initialPolicy.shape == (self.nStates,), "invalid initial policy"
        policy = initialPolicy
        
        iterId = 0
        
        
        while iterId < nIterations:
            # Get new T and R according to policy pi
            V = self.evaluatePolicy(policy)
            policy_new = np.argmax(self.R + self.discount*self.T.dot(V), axis=0)
            iterId = iterId + 1
            if np.array_equal(policy_new, policy):
                break
            policy = policy_new

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
        T_pi = np.zeros((self.nStates,self.nStates))
        R_pi = np.zeros(self.nStates)
        
        for i in range(0,self.nStates):
            T_pi[i] = self.T[policy[i]][i]
            R_pi[i] = self.R[policy[i]][i]
            
        V = R_pi + self.discount*T_pi.dot(V)
        
        iterId = 0
        epsilon = 0

        return [V,iterId,epsilon]

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

        assert initialPolicy.shape == (self.nStates,), "Invalid initial policy"
        policy = initialPolicy
        assert initialV.shape == (self.nStates,), "Invalid initial values"
        V = initialV
        iterId = 0
        epsilon = 1

        while epsilon > tolerance :
            
            for _ in range(0,nEvalIterations):
                V, iId, e =self.evaluatePolicyPartially(policy,V)
                
            policy = np.argmax(self.R + self.discount*self.T.dot(V), axis=0)
            V_new = np.amax(self.R + self.discount*self.T.dot(V), axis=0)
            epsilon = max(np.absolute(V_new - V))
            V = V_new
            iterId = iterId + 1
            
        return [policy,V,iterId,epsilon]
        