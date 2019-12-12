import numpy as np
import gym
import matplotlib.pyplot as plt

pos_space=np.linspace(-1.2,0.6,20)
vel_space=np.linspace(-0.07,0.07,20)

def get_state(observation):
    pos,vel=observation
    pos_binary=np.digitize(pos,pos_space)
    vel_binary=np.digitize(vel,vel_space)
    return (pos_binary,vel_binary)

def max_action(Q,state,actions=[0,1,2]):
    values=np.array([Q[state,a] for a in actions])
    action=np.argmax(values)
    return action


if __name__=='__main__':
    env=gym.make('MountainCar-v0')
    env.max_episode_steps=50000
    n_game=5000
    alpha=0.1
    gamma=0.9
    epsilon=1.0

    states=[]
    for i in range(21):
        for j in range(21):
            states.append((i,j))
    Q={}
    for state in states:
        for action in [0,1,2]:
            Q[state,action]=0


    score=0
    total_rewards=np.zeros(n_game)
    for i in range(n_game):
        done=False
        obs=env.reset()
        state=get_state(obs)
        if i%10==0:
            print('episode: {},score {:.5f}, epsilon {:.5f}'.format(i,score,epsilon))
        score=0

        while not done:
            if np.random.random()<epsilon:
                action=np.random.choice([0,1,2])
            else:
                max_action(Q,state)
            obs_,reward,done,info=env.step(action)
            state_=get_state(obs_)
            score+=reward
            action_=max_action(Q,state_)
            Q[state,action]=Q[state,action]+alpha*(reward+gamma*Q[state,action]-Q[state,action])
            state=state_
            
        total_rewards[i]=score
        epsilon=epsilon-2/n_game if epsilon >0.01 else 0.01

    mean_rewards=np.zeros(n_game)
    for r in range(n_game):
        mean_rewards[i]=np.mean(total_rewards[max(0,t-50):(t+1)])
    plt.plot(mean_rewards)
    plt.savefig('plot.png')

            



