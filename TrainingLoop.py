from QModels2 import DQNAgent
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
import Training as tn

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
        networkStack = networkStack + [(torch.nn.Linear(self.input_width, self.width))]  # Input layer
        hiddenLayers = [torch.nn.Linear(self.width,
                                        self.width)] * self.depth  # There is no non-linear function between the last hidden layer and the output
        networkStack = networkStack + hiddenLayers + [self.activation_function]
        networkStack = networkStack + [(torch.nn.Linear(self.width, self.output_width))]  # Output layer
        self.stack = torch.nn.Sequential(*networkStack)

    def forward(self, input):
        input = input.to(self.device)
        output = self.stack(input)
        return output

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

def manual_Debugging(env):
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

def double_DDQN_TrainingLoop(dqn_model,n,no_done = True): #We have two seperate functions in order to avoid if statements, that can be costly when running millions of iterations

    for y in range(dqn_model.batch_size): #Instead of running two if loops for x * million steps, we start off by getting enough experience for the first training
        dqn_model.Training_step()
    dqn_model.optimizeDDQN(no_done)
    for x in range((n-dqn_model.batch_size)):
        dqn_model.Training_step()
        if dqn_model.t % dqn_model.T_train == 0:
            dqn_model.optimizeDDQN(no_done)
        if dqn_model.t % dqn_model.T_target == 0:
            dqn_model.synchronize()
        if dqn_model.t%dqn_model.T_evaluate == 0:
            dqn_model.follow_Optimal_Average()

#in case we don't need to evaluate
def DDQN_TrainingONLY(dqn_model,n,no_done = False):
    for y in range(
            dqn_model.batch_size):  # Instead of running two if loops for x * million steps, we start off by getting enough experience for the first training
        dqn_model.Training_step()
    dqn_model.optimizeDDQN(no_done=no_done)
    for x in range((n-dqn_model.batch_size)):
        dqn_model.Training_step()
        if dqn_model.t % dqn_model.T_train == 0:
            dqn_model.optimizeDDQN(no_done=no_done)
        if dqn_model.t % dqn_model.T_target == 0:
            dqn_model.synchronize()

def DQN_TrainingLoop(dqn_model,n,no_Done = True): #We have two seperate functions in order to avoid if statements, that can be costly when running millions of iterations

    for y in range(dqn_model.batch_size): #Instead of running two if loops for x * million steps, we start off by getting enough experience for the first training
        dqn_model.Training_step()
    dqn_model.optimizeDDQN(no_Done)
    for x in range((n-dqn_model.batch_size)):
        dqn_model.Training_step()
        if dqn_model.t % dqn_model.T_train == 0:
            dqn_model.optimizeDQN(no_Done)
        if dqn_model.t % dqn_model.T_target == 0:
            dqn_model.synchronize()
        if dqn_model.t%dqn_model.T_evaluate == 0:
            dqn_model.follow_Optimal_Average()


def createActionSpace(n):
    original_Action_Space = ["left", "right", "up", "down"]
    new_Action_Space = []
    for x in range(1,(n+1)):
        for action in original_Action_Space:
            new_Action_Space.append([action, x])


    return new_Action_Space


def createCSV(csv_Name):
    dict = {"rewards": np.array([]), "values": np.array([]), "estimates": np.array([]), "losses": np.array([])}
    df = pd.DataFrame.from_dict(dict)
    df.to_csv(f"{csv_Name}", index=False)

experiment_Names_list = ["en_done_DDQN_4","en_done_DQN_4"]
extra_moves = [1,2,4,6,8,9]
epsilon_constants = [10,10]
no_done = True


action_states = createActionSpace(4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
"""Net = NeuralNetwork(input_width=3, output_width=(4 * 8), layers_shape=(520, 1), activation_function=torch.nn.ReLU(), #Used for debugging networks
                    device=device, env=env, availableActions=action_states)"""
for idx,exp_Nr in enumerate(experiment_Names_list):
        n = (875 * 1000)*2
        env = GridWorld(seed=1)
        state = torch.tensor((env.x, env.y, env.has_key), dtype=torch.float32)  # board + has_key
        network_type = "DDQN"
        csv_Name = fr"C:\Users\benja\PycharmProjects\IISProject\Final_Study\Reward Density\Data:{exp_Nr}"
        createCSV(csv_Name)
        a = 4
        actions = createActionSpace(a)
        names1 = [fr"C:\Users\benja\PycharmProjects\IISProject\Final_Study\Revisited 7 + 7 mil\No_done\Target{exp_Nr}", fr"C:\Users\benja\PycharmProjects\IISProject\Final_Study\Revisited 7 + 7 mil\No_done\Policy{exp_Nr}"]
        Network_Architecture = (520, 1)
        epsilon_decay = (0.09) / (n /50)
        save_Cutoff = 100000
        T_eval = 125*1000
        sample_Size = 50000
        dqn = DQNAgent(state_representation=state, actions_List=actions, layers_shape=Network_Architecture, env=env,
                       load_Name=(names1), save_Name=(names1),
                       epsilon_start=1, T_evaluate=T_eval, eval_Sample_Sizes=sample_Size, save_Cutoff=save_Cutoff,
                       epsilon_decay=epsilon_decay, feature_type="XY", csv_Name=f"{csv_Name}")
        if idx != 2:
            tn.double_DDQN_TrainingLoop(dqn,n,no_done=no_done)
        else:
            tn.DQN_TrainingLoop(dqn,n,no_done=no_done)
