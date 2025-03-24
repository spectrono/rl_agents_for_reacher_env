# Project Details

This project aims to solve the so called Reacher environment with a single arm, which is  part of the Unity ML-Agents environment via Deep Reinforcement Learning.

## Reacher environment

The environemnt realizes a double-jointed arm which can be move to a target's locations by executing actions in this environment. Each action is a vector with four values, corresponding to torque applicable to two joints. Every entry in the action vector must be a number between -1 and +1. The target location is a sphere which moves within the robot arm's reach. Ther environment's observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the robot arm.

A reward of +0.1 is provided for each step that the agent's hand is in the target location. Thus, the RL-agent's goal is to maintain its position at the target location for as many time steps as possible.

## How to solve the environment

The environment is considered as solved when the agent achieves an average score of at least +30.0 over a period of 100 episodes.

## Approach taken

To solve the environment a variant of Deep Deterministic Policy Gradient (DDPG) is implemented (see REPORT.md for details). This is an algorithm used in deep reinforcement learning for continuous action spaces. It combines elements from both policy gradient methods and Q-learning.

## Getting started

The code is tested on Linux (Pop!_OS 22.04 LTS), only. All code is developed and tested with python 3.9.21, pytorch 2.6.0 and standard libraries such as numpy and matplotlib.

For further details on dependent packages see the provided requirenement_*.txt files:

- [packages installed via conda](./requirements_conda.txt)
- [packages installed via pip](./requirements_pip.txt)

Additionally, download the single arm reacher environment from this location: [Headless Single Arm Reacher environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) and unzip it in the main root of your repository. The code to train the agent uses the headless version of the environment, only! It speeds up the training a little bit.

## Instructions (for training)

All code is provided within the **code/** folder. The main training file is the **ddpg_train.py** file which setups and trains the agent. It contains a set of hyperparameters which have been successfuly used to solve the reacher environment. Just call
`python code/ddpg_train.py`
and it will execute the training with the preselected hyperparameter. Results, like the achieved scores and the agent's network weights, will be placed in the **results/** folder.

Feel free to modify the hyperparameter to see if you can reach an even better performance. For details on the training results and the used hyperparameter see the [REPORT.md](./REPORT.md) file.
