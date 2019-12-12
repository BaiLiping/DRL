import numpy as np
import math
import sys
wave_length=0.7
phase=180/5

def generate_target():
    index=np.random.choice(range(48))
    position=np.unravel_index(index,(8,6))
    return position
    
def location_info(target_location):
    distance=math.sqrt(pow((3*(target_location[0])),2)+pow(3*(target_location[1]),2))
    impact=(1/(distance+1))*30*math.cos(distance/wave_length)
    return impact
        
def compute_reward(x,y,p,target_location):
    distance=math.sqrt(pow((3*(x-target_location[0])),2)+pow(3*(y-target_location[1]),2))
    reward=(1/(distance+1))*30*math.cos(distance/wave_length+p*phase)
    return reward

class PhasedArrayEnv(object):
    def __init__(self):
        self.wave_length=0.7
        self.phase=180/5
        self.nA=47*5
        #self.nS=48*47*6
        self.state=np.zeros(48)
        self.target_location=generate_target()
    
    def reset(self):
        self.target_location=generate_target()
        self.state=np.zeros(48)
        self.state[0]=location_info(self.target_location)
    
    def get_nA(self):
        return self.nA

    def step(self,action):
        x=np.unravel_index(action,(8,6,5))[0]
        y=np.unravel_index(action,(8,6,5))[1]
        p=np.unravel_index(action,(8,6,5))[2]+1
        state_index=np.ravel_multi_index((y,x),(8,6))
        self.state[state_index]=p
        reward=compute_reward(x,y,p,self.target_location)
        return self.state,reward
    
    def render(self):
        outfile=sys.stdout
        for i in range(len(self.state)):
            position=np.unravel_index(i,(8,6))
            if self.state[i]!=0:
                output='  '
                output+=str(int(self.state[i]))
                output+='  '
            else:
                output='  _  '
    
            if position[1]==0:
                output=output.lstrip()
            if position[1]==(8,6)[1]-1:
                output=output.rstrip()
                output+='\n'
            outfile.write(output)
        outfile.write('\n')
