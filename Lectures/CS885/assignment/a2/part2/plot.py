import numpy as np
import matplotlib.pyplot as plt

def plot(fileName):
    A = np.load(fileName+".npy")
    plt.plot(A, label=fileName)
	
def main():
    plot('unmodified')
    plot('shared')
    #plt.xlim((0, 500))
    plt.title('DDPG in the Pendulum-w0 environment')
    plt.xlabel('episode #')
    plt.ylabel('average reward of each episode')
    plt.legend(loc='lower right')
    plt.show()

main()
