import numpy as np

class testglobal(object):
    global state
    def __init__(self,state):
        self.state=state

    def class_method(self):
        self.location=np.random.choice(10)
        self.state[self.location]=1
        print(self.location)

