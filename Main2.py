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

experiment_names = "Fixed_DQN_Test"


actions = tn.createActionSpace(4)
env = GridWorld(noStay_Rewards=False,seed=1)
state_Representation = torch.tensor((env.x,env.y,env.has_key),dtype=torch.float32)

dqn = DQNAgent(layers_shape=(520,1),state_representation=state_Representation,env=env,actions_List=actions,load_Name=[None,None],save_Name=[None,None])
dqn.pNet = tn.load_GPU_network(fr"C:\Users\benja\PycharmProjects\IISProject\Final_Study\Fixed Epsilon\4 times 7\Networks\Run0\Policyrun(0)Fixed14milDDQN_4",dqn.pNet)

heatMap = dp.get_Heat_Map(dqn,500*1000,1,path=fr"{experiment_names}")
heat_Map = sns.heatmap(heatMap)
plt.show()