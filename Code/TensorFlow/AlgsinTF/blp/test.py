from blackjack import BlackjackEnv
import numpy as np

env=BlackjackEnv()

def print_observation(observation):
    score,dealer_score,usable_ace=observation
    print('player score: {}, player first card: {}, usable ace {}'.format(score, dealer_score, usable_ace))

def strategy(observation):
    score, dealer_score, usable_ace=observation
    return 0 if score >=20 else 1
    #stick (action 0) if the socre is >20, hit (action 1) if the socre smaller than 20

    

if __name__=='__main__':
    env=BlackjackEnv()
    print(env.reset())
    
    for i in range(20):
         observation=env.reset()
         for t in range(100):
             print_observation(observation)
             action=strategy(observation)
             print("taking action: {}".format(['stick','hit'][action]))
             observation, reward, done, _ =env.step(action)
             if done:
                 print_observation(observation)
                 print('game over. Reward {}'.format(reward))
                 break
   

