# Multi-Agent-RL-DQN
Implements a Deep Q Network for Multi-agent RL. This was done as part of a UCL COMP0124 Multi-Agent Artificial Intelligence Coursework. 

## Environment
The environment can be found [here](https://github.com/koulanurag/ma-gym). It is set to a four-agent switch game in which the agents have to share a tunnel in order to get to their goals. 

## Instructions
To run, just download the notebook and run it in Colab. Currently, this version has not been adapted to direct use. 

Make sure that you have all the dependencies. In google CoLab, this can be done by running the following lines before the rest of the code:

```
  !rm -rf /content/ma-gym  
  !git clone https://github.com/koulanurag/ma-gym.git 
  %cd /content/ma-gym 
  !pip install -q -e . 
  !apt-get install -y xvfb python-opengl x11-utils > /dev/null 2>&1
  !pip install pyvirtualdisplay > /dev/null 2>&1
  !apt-get install x11-utils
  !apt-get update > /dev/null 2>&1
  !apt-get install cmake > /dev/null 2>&1
  !pip install --upgrade setuptools 2>&1
  !pip install ez_setup > /dev/null 2>&1
  !pip install -U gym[atari] > /dev/null 2>&1
```
  
Then import ReplayBuffer.py and dqn_agent.py and run the main.py code.
