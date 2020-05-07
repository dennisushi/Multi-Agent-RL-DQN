from collections import namedtuple, deque, defaultdict
from functools import partial
import numpy as np
import random

class ReplayMemory(object):
  # Defines the states that are held in the replay memories
  def __init__(self, state, action, reward, next_state,done):
    self.state = state
    self.action = action
    self.reward = reward
    self.next_state = next_state
    self.done = done

class MemoryBuffer(object):
  def __init__(self, action_size, buffer_size, batch_size, seed):
    self.action_size = action_size
    self.memory_buffer = deque(maxlen=buffer_size)
    self.batch_size = batch_size
    self.seed = np.random.seed(seed)
    self.size = len(self.memory_buffer)
      
  def remember(self,state, action, reward, next_state,done):
    m = ReplayMemory(state,action,reward,next_state,done)
    self.memory_buffer.append(m)
    self.size = len(self.memory_buffer)
  
  def sample(self):
    memories = random.sample(self.memory_buffer,k=self.batch_size)
    
    states      = np.vstack([m.state      for m in memories if m is not None])
    actions     = np.vstack([m.action     for m in memories if m is not None])
    rewards     = np.vstack([m.reward     for m in memories if m is not None])
    next_states = np.vstack([m.next_state for m in memories if m is not None])
    dones       = np.vstack([m.done       for m in memories if m is not None]).astype(np.uint8)
    return (states,actions,rewards,next_states,dones)
