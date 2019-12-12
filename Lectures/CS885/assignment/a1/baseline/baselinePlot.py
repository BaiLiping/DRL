import numpy as np
import matplotlib.pyplot as plt


def plot(episode_rewards):
	xAxis = range(1,len(episode_rewards)+1)
	plt.plot(xAxis, episode_rewards, label='')
	plt.xlim((0, 1000))
	plt.xlabel('episode #')
	plt.ylabel('cumulative reward of each episode')
	plt.legend(loc='lower right')
	plt.show()


def updatePlot():
	update1000 = np.load('reward1000.npy')
	update250 = np.load('reward250.npy')
	update50 = np.load('reward50.npy')
	updates1 = np.load('reward1.npy')
	
	plt.plot(update1000, label='1000')
	plt.plot(updates250, label='250')
	plt.plot(update50, label='50')
	plt.plot(update1, label='1')
	
	plt.xlim((0, 1000))
	plt.xlabel('episode #')
	plt.ylabel('cumulative reward of each episode')
	plt.legend(loc='right')
	plt.show()

def samplePlot():
	sample32 = np.load('sample32.npy')
	sample15 = np.load('sample15.npy')
	sample5 = np.load('sample5.npy')
	sample1 = np.load('sample1.npy')
	
	plt.plot(sample1, label='1')
	plt.plot(sample5, label='5')
	plt.plot(sample15, label='15')
	plt.plot(sample32, label='32')
	
	plt.xlim((0, 1000))
	plt.xlabel('episode #')
	plt.ylabel('cumulative reward of each episode')
	plt.legend(loc='lower right')
	plt.show()

samplePlot()
