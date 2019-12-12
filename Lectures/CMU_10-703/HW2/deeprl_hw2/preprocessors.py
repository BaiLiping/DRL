
"""Suggested Preprocessors."""

import numpy as np
from PIL import Image
from collections import deque
import time

from deeprl_hw2 import utils
from deeprl_hw2.core import Preprocessor


class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, history_length=1):
        self.history_length = history_length
        self.history = deque([], maxlen = history_length)

    def process_state_for_network(self, state):
        """You only want history when you're deciding the current action to take.
        return a 4d numpy array, first dimension is for batch
        """
        assert state.dtype == 'uint8'
        assert state.shape == (84, 84)

        if (len(self.history) < self.history_length - 1):
            for i in xrange(len(self.history), self.history_length - 1):
                self.history.append(np.zeros(state.shape, dtype = 'uint8'))

        assert len(self.history) >= self.history_length - 1

        self.history.append(state)
        history_state = np.zeros((1, state.shape[0], state.shape[1], self.history_length), dtype = 'uint8')
        for i in xrange(len(self.history)):
            history_state[:, :, :, i] = self.history[i]

        return history_state

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self.history.clear()

    def get_config(self):
        return {'history_length': self.history_length}


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, new_size):
        self.new_size = new_size

    def process_state_for_memory(self, observation):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        assert observation.ndim == 3        # (height, width, channel)
        state_unit8 = np.zeros(self.new_size)
        image_tmp = Image.fromarray(observation)
        state_unit8 = np.asarray(image_tmp.resize(self.new_size).convert('L'))
        state_unit8.astype('uint8')      
        assert state_unit8.shape == self.new_size

        return state_unit8


    def process_state_for_network(self, observation):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """

        # process observation from raw image to 84x84x1 with floating type
        return self.process_state_for_memory(observation).astype('float32') / 255.      

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        return np.clip(reward, -1.0, 1.0)


class PreprocessorSequence(Preprocessor):
    """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).

    You can easily do this by just having a class that calls each preprocessor in succession.

    For example, if you call the process_state_for_network and you
    have a sequence of AtariPreproccessor followed by
    HistoryPreprocessor. This this class could implement a
    process_state_for_network that does something like the following:

    state = atari.process_state_for_network(state)
    return history.process_state_for_network(state)
    """
    def __init__(self, preprocessors):
        self.Atari = preprocessors['Atari']
        self.History = preprocessors['History']

    def process_state_for_network(self, observation):
        '''
        observation: 84x84 uint8

        return: 84x84 float32
        '''

        assert observation.dtype == 'uint8', 'observation in forward is not correct'
        assert observation.shape == (84, 84)
        tmp = self.History.process_state_for_network(observation)
        processed_state = tmp.astype('float32') / 255. 
        
        random_index = np.random.randint(84, size=1)[0]
        # assert processed_state.shape == (1, 84, 84, 4)
        assert processed_state[0, random_index, random_index, 0] <=1. and processed_state[0, random_index, random_index, 0] >= 0., 'processed state is not correct while forward'
        return processed_state


    def process_state_from_memory_batch(self, batch_state_from_memory):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.

        batch_state_from_memory is a list which has length of batch_size, each item
        is a state with shape 84x84x4

        return a numpy array with shape 32x84x84x4
        """
        batch_num = len(batch_state_from_memory)
        random_batch = np.random.randint(batch_num, size=1)[0]

        assert batch_state_from_memory[random_batch].shape == (84, 84, 4)
        assert batch_state_from_memory[random_batch].dtype == 'uint8'
        batch_state_processed = np.array(batch_state_from_memory).astype('float32') / 255.

        # assert batch_state_processed.shape == (batch_num, 84, 84, 4)
        random_index = np.random.randint(84, size=1)[0]
        assert batch_state_processed[random_batch, random_index, random_index, 0] >= 0 and batch_state_processed[random_batch, random_index, random_index, 0] <= 1

        return batch_state_processed
