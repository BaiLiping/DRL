import numpy as np

class PhasedArrayEnv(object):
    def __init__(self):
        self.shape=(3,8)
        nS=23*6*50#need to think a bit more about this
        nA=np.array(23,6) #keep the action space as a 2D array, flat it before training

        init_state=np.zeros(nS)
        init_state[np.ravel_multi_index((1,0),self.shape)]=1.0

    def step():
    def reset():
    def render():
        outfile=sys.stdout
        for s in range(self.nS):
            position=np.unravel_index(s,self.shape)
            if self.s == s:
                output 'X'
            elif position ==(#terminal position):
                    output='T'
            else:
                output='O'

            outfile.write(output)
        outfile.write('\n')





        
