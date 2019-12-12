import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('vanilla_data.pickle','rb') as rfile:
    vanilla_reward=pickle.load(rfile)
with open('acceleration_data.pickle','rb') as rfile:
    acceleration_reward=pickle.load(rfile)
with open('weight.pickle','rb') as rfile:
    weight_reward=pickle.load(rfile)

x=np.arange(400)
plt.plot(x,vanilla_reward,label='original')
plt.plot(x,acceleration_reward,label='acceleration')
plt.plot(x,weight_reward,label='weight')
plt.title('result')
plt.xlabel('expriments')
plt.ylabel('reward')
plt.legend()
plt.savefig('result')
plt.show()
