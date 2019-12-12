from agent import Agent
from env import PhasedArrayEnv
from policy_estimator import Policy_Estimator
from value_estimator import Value_Estimator
#import plot

def main():
    num_episodes=10000
    num_steps=9
    
    #set global variables that would be passed to methods
    global state
    global td_error
    global action
    
    #set hyperparameters
    gamma=0.9
    alpha=0.01
    beta=0.01
    
    #setup environment
    env=PhasedArrayEnv()
    state=env.reset() #make state a global variable such that class method can access
    
    #show the behaviour of random policy
    print('demostrate random policy: ')
    env.reset()
    env.render()
    for t in range(num_steps):
        action=np.random.choice(np.arange(nA))
        state,reward=env.step(action)
        env.render()
        print(reward)

    
    #traning the policy
    agent=Agent(env,num_episodes,num_steps,gamma,alpha,beta)
    error_log,policy_estimator,value_estimator=agent.ac_traing()
    '''
    #for future implementation
    agent.td_traing()
    agent.mc_traing()
    '''
    #plot(error_log)

    #show the behaviour of trained policy
    print('demostrate trained policy: ')
    env.reset()
    env.render()#render should include the target location
    for t in range(num_steps):
        prob=ploicy_estimator.predict(state)
        action=argmax(prob)
        state,reward=env.step()
        env.render()
        print(reward)


if __name__=='__main__':
    main()
