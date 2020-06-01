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

class MARL_QNetwork(nn.Module):
  # Defines the Q Network used for local and target networks
  def __init__(self, state_size, action_size, seed, fc1_unit=64, fc2_unit = 64):
    super(MARL_QNetwork, self).__init__()
    self.seed = torch.manual_seed(seed)
    self.fc1 = nn.Linear(state_size, fc1_unit)
    self.fc2 = nn.Linear(fc1_unit, fc2_unit)
    self.head = nn.Linear(fc2_unit, action_size)

    self.init_weights(self.fc1)
    self.init_weights(self.fc2)
    self.init_weights(self.head)

  def init_weights(self,m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
      
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.head(x)
    return x.view(x.size(0), -1)

class MARL_DQN_Agent(object):
  def __init__(self, state_size, action_size, seed,
               update_every=5, lr=1e-3,
               buffer_size = 1000, batch_size=5,
               gamma = 0.99, tau = 1e-3):
    super(MARL_DQN_Agent, self).__init__()
    # Model settings
    self.state_size = state_size
    self.action_size = action_size
    self.actions = np.arange(self.action_size)
    # Model parameters
    self.batch_size = batch_size
    self.buffer_size = buffer_size
    self.gamma=gamma
    self.tau = tau
    self.update_step = update_every
    # Model seed
    self.seed = random.seed(seed)

    # Debugging variables
    self.policy = defaultdict(partial(np.zeros, self.action_size))
    self.step_policy = defaultdict(partial(np.zeros, self.action_size))
      
    #Q- Network
    self.QNet_policy = MARL_QNetwork(state_size, action_size, seed).to(device)
    self.QNet_policy.train()
    self.QNet_target = MARL_QNetwork(state_size, action_size, seed).to(device)
    self.QNet_target.eval()
    
    # Training setup
    self.optimizer = optim.Adam(self.QNet_policy.parameters(),lr=lr)
    self.memory = MemoryBuffer(action_size, self.buffer_size,self.batch_size,seed)
    self.lstep = 0

  def step(self, state, action, reward, next_state, done):
    # Save experience in memory
    self.memory.remember(state, action, reward, next_state, done)
    # Learn every X time steps.
    self.lstep = (self.lstep+1)% self.update_step
    if self.lstep == 0 and self.memory.size>self.batch_size:
      self.learn(self.memory.sample())

  def act(self, state, eps = 0):
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    # Put network in evaluation mode
    self.QNet_policy.eval()
    with torch.no_grad():
      action_values = self.QNet_policy(state_tensor)
    # Put network back into training mode
    self.QNet_policy.train()
    
    # Debugging parameters
    self.policy[(state[0],state[1])]= action_values.cpu().data.numpy()
    self.step_policy[(0,state[2])]= action_values.cpu().data.numpy()

    #Epsilon-greedy action selction
    if random.random() > eps:
      return np.argmax(action_values.cpu().data.numpy())
    else:
      return random.choice(self.actions)

  def evaluate_policy(self):
    # Print policy per observation
    print("\n")
    for line in self.policy:
      print (line, self.policy[line][0])  

  def evaluate_step_policy(self):
    # Print policy per time step
    print("\n")
    for line in self.step_policy:
      print (line, np.argmax(self.step_policy[line][0]))        
  
  def convert_to_tensor(data, type="float"):
    if type=="long":
      return torch.from_numpy(data).long().to(device)    
    else:
      return torch.from_numpy(data).float().to(device)


  def learn(self, experiences):
    states, actions, rewards, next_states, dones = experiences

    # Convert to tensors
    states = convert_to_tensor(states)
    actions = convert_to_tensor(actions,"long")
    rewards = convert_to_tensor(rewards)
    next_states = convert_to_tensor(next_states)
    dones = convert_to_tensor(dones)

    # Compute gradient and optimize
    criterion = torch.nn.MSELoss()
    Q_expected = self.QNet_policy(states).gather(1,actions)
    with torch.no_grad():
      Q_targets_next = self.QNet_target.forward(next_states).detach().max(1)[0].unsqueeze(1)
      Q_targets_current = rewards + (self.gamma* Q_targets_next*(1-dones))
    loss = criterion(Q_expected,Q_targets_current).to(device)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    # Update network parameters
    for policy_p, target_p in zip (self.QNet_policy.parameters(), self.QNet_target.parameters()):
      target_p.data.copy_(self.tau*policy_p.data + (1-self.tau)*target_p.data)

  def save_model(self, name):
    self.best_policy_net = self.QNet_policy.parameters()
    self.best_target_net = self.QNet_target.parameters()
  def load_best_model(self):
    for target_p_best, target_p in zip(self.best_target_net, self.QNet_target.parameters()):
      target_p.data.copy_(target_p_best)
    for policy_p_best, policy_p in zip(self.best_policy_net, self.QNet_policy.parameters()):
      policy_p.data.copy_(policy_p_best)
