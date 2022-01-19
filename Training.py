from GridWorlds import GridWorld
import torch
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math
import torch.nn as nn
import torch.optim as optim
import DataProcessing as dp
import seaborn as sns
from QModels2 import DQNAgent
""""Settings for the experiment:
seed = Environment seed (1-6)
number_actions = Number of actions possible
names = [target network's name,policy network's name]
layers_shape = (mXn) shape of the fully connected linear layer
sample_size = How many samples of real returned reward to be evaluated.
file_name = Name of the data-file containing (average_true_returned_rewards,average_overestimation,average_loss)
"""

""" A network for debugging"""
class NeuralNetwork(torch.nn.Module):
    def __init__(self,
                 layers_shape,
                 output_width,
                 input_width,
                 activation_function,
                 device,
                 env = None,
                 availableActions = None):
        super(NeuralNetwork,self).__init__()

        # Model parameters
        self.width = layers_shape[0]
        self.depth = layers_shape[1]
        self.output_width = output_width
        self.input_width = input_width
        self.activation_function = activation_function
        self.makeModel()
        self.actions_available = availableActions
        self.env = env
        # CUDA vs GPU
        self.device = device

    def makeModel(self):
        networkStack = []
        networkStack += [(torch.nn.Linear(self.input_width, self.width))]
        hiddenLayers = [torch.nn.Linear(self.width,
                                        self.width)] * self.depth  # There is no non-linear function between the last hidden layer and the output
        networkStack = networkStack + hiddenLayers + [self.activation_function]
        networkStack += [(torch.nn.Linear(self.width, self.output_width))]
        self.stack = torch.nn.Sequential(*networkStack)

    def forward(self, input):
        input = input.to(self.device)
        return self.stack(input)

    def random_forward(self):
        return torch.tensor(np.random.randint(0,100,size=(self.output_width)))

"""Will take the name of a GPU network, load its weights and save them as the original name with "DICT" attached at the end

    For debugging networks trained on the HPC's GPU on my PC's CPU
"""
def load_GPU_network(PATH,Net,save = False):
    device = "cpu"
    model = torch.load(PATH,map_location=torch.device("cpu"))
    torch.save(model.state_dict(),f"{PATH}DICT")
    Net.load_state_dict(torch.load(f"{PATH}DICT",map_location=device))
    return Net

"""
Starts a loop for manual debugging:
Direction = [0:left,1:right,2:up,3:down]
Times = Number of times to take the chosen direction

Code word to end the loop: 69420 in Direction
"""

def manual_Debugging(env,dqn):
    Dir = None
    while Dir != 69420:
        Dir = int(input("Direction"))
        Times = int(input("Times"))
        actions = ["left", "right", "up", "down"]
        action = [actions[Dir], Times]
        step = env.step(action)
        print(step[1])
        print(env.board.transpose())
        if step[2]:
            env.reset()

def double_DDQN_TrainingLoop(dqn_model,n,noDone = False): #We have two seperate functions in order to avoid if statements, that can be costly when running millions of iterations

    for _ in range(100*1000): #Fill up the EMR before the network gets stuck in a self-reinforcing loop (Outer walls)
        dqn_model.Training_step()
    dqn.model.epsilon = 1  # Reset epsilon
    dqn_model.optimizeDDQN(no_done=noDone)
    for _ in range((n-dqn_model.batch_size)):
        dqn_model.Training_step()
        if dqn_model.t % dqn_model.T_train == 0:
            dqn_model.optimizeDDQN(no_done=noDone)
        if dqn_model.t % dqn_model.T_target == 0:
            dqn_model.synchronize()
        if dqn_model.t%dqn_model.T_evaluate == 0:
            dqn_model.follow_Optimal_Average()

def DQN_TrainingLoop(dqn_model,n,noDone = False): #We have two seperate functions in order to avoid if statements, that can be costly when running millions of iterations

    for _ in range(100*1000): #Fill up the EMR before the network gets stuck in a self-refinforcing loop (Outer walls)
        dqn_model.Training_step()
    dqn.model.epsilon = 1 #Reset epsilon
    dqn_model.optimizeDDQN(no_done=noDone)
    for _ in range((n-dqn_model.batch_size)):
        dqn_model.Training_step()
        if dqn_model.t % dqn_model.T_train == 0:
            dqn_model.optimizeDQN(no_done=noDone)
        if dqn_model.t % dqn_model.T_target == 0:
            dqn_model.synchronize()
        if dqn_model.t%dqn_model.T_evaluate == 0:
            dqn_model.follow_Optimal_Average()


def createActionSpace(n): #It actually matters what you set it to, after I updated the QModel's output to be the size of the action list
    original_Action_Space = ["left", "right", "up", "down"]
    new_Action_Space = []
    for x in range(1,(n+1)):
        for action in original_Action_Space:
            new_Action_Space.append([action, x])

    return new_Action_Space

#The standard format I use to evaluate the networks
def createCSV(csv_Name):
    dict = {"rewards": np.array([]), "values": np.array([]), "estimates": np.array([]), "losses": np.array([])}
    df = pd.DataFrame.from_dict(dict)
    df.to_csv(f"{csv_Name}", index=False)


