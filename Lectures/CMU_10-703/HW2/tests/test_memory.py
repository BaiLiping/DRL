from __future__ import division
import pytest
import numpy as np

import __init__paths__
from numpy.testing import assert_allclose
from deeprl_hw2.core import ReplayMemory, RingBuffer


def test_ring_buffer():
    def assert_elements(b, ref):
        assert len(b) == len(ref)
        for idx in range(b.maxlen):
            if idx >= len(ref):
                with pytest.raises(KeyError):
                    b[idx]
            else:
                assert b[idx] == ref[idx]

    b = RingBuffer(5)

    # Fill buffer.
    assert_elements(b, [])
    b.append(1)
    assert_elements(b, [1])
    b.append(2)
    assert_elements(b, [1, 2])
    b.append(3)
    assert_elements(b, [1, 2, 3])
    b.append(4)
    assert_elements(b, [1, 2, 3, 4])
    b.append(5)
    assert_elements(b, [1, 2, 3, 4, 5])

    # Add couple more items with buffer at limit.
    b.append(6)
    assert_elements(b, [2, 3, 4, 5, 6])
    b.append(7)
    assert_elements(b, [3, 4, 5, 6, 7])
    b.append(8)
    assert_elements(b, [4, 5, 6, 7, 8])

def test_sampling():
    obs_size = (3, 4)
    test = np.random.random(obs_size)
    memory = ReplayMemory(max_size=100, input_shape=obs_size, window_length=2, datatype=test.dtype)
    actions = range(5)
    
    obs0 = np.random.random(obs_size)
    terminal0 = False
    action0 = np.random.choice(actions)
    reward0 = np.random.random()
    
    obs1 = np.random.random(obs_size)
    terminal1 = False
    action1 = np.random.choice(actions)
    reward1 = np.random.random()
    
    obs2 = np.random.random(obs_size)
    terminal2 = False
    action2 = np.random.choice(actions)
    reward2 = np.random.random()
    
    obs3 = np.random.random(obs_size)
    terminal3 = True
    action3 = np.random.choice(actions)
    reward3 = np.random.random()

    obs4 = np.random.random(obs_size)
    terminal4 = False
    action4 = np.random.choice(actions)
    reward4 = np.random.random()

    obs5 = np.random.random(obs_size)
    terminal5 = False
    action5 = np.random.choice(actions)
    reward5 = np.random.random()

    obs6 = np.random.random(obs_size)
    terminal6 = False
    action6 = np.random.choice(actions)
    reward6 = np.random.random()
    
    memory.append(obs0, action0, reward0, terminal1)
    memory.append(obs1, action1, reward1, terminal2)
    memory.append(obs2, action2, reward2, terminal3)
    memory.append(obs3, action3, reward3, terminal4)
    memory.append(obs4, action4, reward4, terminal5)
    memory.append(obs5, action5, reward5, terminal6)
    assert memory.nb_entries == 6

    experiences = memory.sample(batch_size=5, indexes=[0, 1, 2, 3, 4])
    assert len(experiences) == 5
    
    assert_allclose(experiences[0].state, np.dstack((np.zeros(obs_size), obs0)))
    assert_allclose(experiences[0].next_state, np.dstack((obs0, obs1)))
    assert experiences[0].action == action0
    assert experiences[0].reward == reward0
    assert experiences[0].is_terminal is False

    assert_allclose(experiences[1].state, np.dstack((obs0, obs1)))
    assert_allclose(experiences[1].next_state, np.dstack((obs1, obs2)))
    assert experiences[1].action == action1
    assert experiences[1].reward == reward1
    assert experiences[1].is_terminal is False

    assert_allclose(experiences[2].state, np.dstack((obs1, obs2)))
    assert_allclose(experiences[2].next_state, np.dstack((obs2, obs3)))
    assert experiences[2].action == action2
    assert experiences[2].reward == reward2
    assert experiences[2].is_terminal is True

    assert not np.all(experiences[3].state == np.dstack((obs2, obs3)))

    assert_allclose(experiences[4].state, np.dstack((np.zeros(obs_size), obs4)))
    assert_allclose(experiences[4].next_state, np.dstack((obs4, obs5)))
    assert experiences[4].action == action4
    assert experiences[4].reward == reward4
    assert experiences[4].is_terminal is False
    

if __name__ == '__main__':
    pytest.main([__file__])
    # test_sampling()
