import gym
import ma_gym
from ma_gym.wrappers import Monitor
import matplotlib.pyplot as plt
import glob
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay

from pyvirtualdisplay import Display
display = Display(visible=0, size=(400, 300))
display.start()

# Original Code:
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple, deque, defaultdict
from functools import partial
import numpy as np
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import ReplayBuffer
import dqn_agent

"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""

def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 300px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")
    

def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env


def rollout_all_agent(agents, n_episodes= 2000, iterations_per_episode = 50, 
                      eps_start=1.0, eps_end = 0.01,
                      eps_decay=0.99, verbose = False, name=""):
    scores = [] # list containing score from each episode
    last_N_scores = deque(maxlen=20) # rollign window of last N scores
    eps = eps_start
    max_score=-20-1
    for ep_iter in range(1, n_episodes+1):
      state_n = env.reset()
      score = 0
      for t in range(iterations_per_episode):
        actions = np.array([agent_i.act(np.array([state_i[0],state_i[1],t*1.0/50.0]),eps) for state_i, agent_i in zip(state_n, agents)])
        next_state_n,reward_n,done_n,_ = env.step(actions)

        for j, (state, reward, next_state, done, agent_j) in enumerate(zip(state_n, reward_n, next_state_n, done_n, agents)):
          state_i = np.array([state[0], state[1],t/50.0])
          next_state_i = np.array([next_state[0], next_state[1],(t+1)/50.0])
          # Reward cooperation:
          reward_i = sum(reward_n)
          # Reward No-operation behaviour after 
          # goal is reached
          if done and actions[j] == 4:
            reward_i += 0.2 
          agent_j.step(state_i, actions[j], reward_i, next_state_i, done)
          
        state_n = next_state_n
        score += np.asarray(reward_n).sum()

        if np.all(done_n):
          last_N_scores.append(score) ## save the most recent score
          scores.append(np.mean(last_N_scores)) ## sae the most recent score
          eps = max(eps*eps_decay,eps_end)      # decrease the epsilon
          if (ep_iter % 150 == 0):
            print('Episode {}, average reward is {:.3f}, epsilon is {:.3f}'.format(ep_iter,np.mean(last_N_scores),eps))
          if score > max_score and eps == eps_end:
            max_score = score
            for i,a_i in enumerate(agents):
              a_i.save_model(name+"agent"+str(i))
          break
        
      if (ep_iter==n_episodes):
        for i, a_i in enumerate(agents):
          agents[i].load_best_model()
          if verbose:
            print("Agent "+str(i)+" strategy per timestep")
            agents[i].evaluate_step_policy()
        
    return scores

def MARL_evaluate(MARL_agents):
  state_n = env.reset()
  reward = 0
  print("t, state, action")
  for t in range (50):
    actions = np.array([agent_i.act(np.array([state_i[0],state_i[1],t/50.0]),eps=0) for state_i, agent_i in zip(state_n, MARL_agents)])
    next_state_n,reward_n,done_n,_ = env.step(actions)
    print(t,state_n,actions)
    state_n = next_state_n
    reward += np.asarray(reward_n).sum()
    env.render()
    if np.all(done_n):
      if (t!=49):
        print("Goals reached before timeout! In {} steps. Total reward was {:.2f}".format(t,reward))
      else:
        print ("Not all goals reached. Timeout. Total reward was {:.2f}".format(reward))
      break
  env.close()
  show_video()
  return



# Environment setup
env = wrap_env(gym.make("Switch2-v0"))
agent_num = env.n_agents
action_num = len(env.get_action_meanings(0))

# Env Setup 
env = wrap_env(gym.make("Switch2-v0"))
MARL_agents = []
for i in range(agent_num):
  MARL_agent = MARL_DQN_Agent(state_size=3,action_size=action_num,seed=0, lr=0.0008)
  MARL_agents.append(MARL_agent)

# Training
scores= rollout_all_agent(MARL_agents)
env.close()

# EVALUATION
env = wrap_env(gym.make("Switch2-v0"))
MARL_evaluate(MARL_agents)

# Learning curve
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)),scores)
plt.title("Learning curve for Switch2")
plt.ylabel('Score')
plt.xlabel('Epsiode #')
plt.grid()
plt.show()
