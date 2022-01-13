from QModels2 import DQNAgentFullSpace,DQNAgentXY
from GridWorlds import GridWorld
import torch
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
import pandas
import sys
import math
""""Settings for the experiment:
seed = Environment seed (1-6)
number_actions = Number of actions possible
names = [target network's name,policy network's name]
layers_shape = (mXn) shape of the fully connected linear layer
sample_size = How many samples of real returned reward to be evaluated.
file_name = Name of the data-file containing (average_true_returned_rewards,average_overestimation,average_loss)
"""


env = GridWorld(seed=1)
n = 100*1000
state1 = torch.tensor((env.x,env.y,env.has_key),dtype=torch.float32) #board + has_key
state2 = torch.tensor(env.board,dtype=torch.float32).flatten()
names1 = ["XYTarget", "XYSpacePolicy"]
names2 = ["FullSpaceTarget","FullSpacePolicy"]
dqnXY = DQNAgentXY(state_representation=state1,actions_List=[["left",1],["right",1],["up",1],["down",1]],layers_shape=(200,4), env=env, load_Name=(names1), save_Name=(names1),
               epsilon_start=1,eval_Sample_Sizes=n,T_evaluate=n)

dqnFULL = DQNAgentFullSpace(state_representation=state2,actions_List=[["left",1],["right",1],["up",1],["down",1]],layers_shape=(200,4), env=env, load_Name=(names2), save_Name=(names2),
               epsilon_start=1,eval_Sample_Sizes=n,T_evaluate=n)
timestart = time.perf_counter()

def double_DDQN_TrainingLoop(dqn_model,n): #We have two seperate functions in order to avoid if statements, that can be costly when running millions of iterations
    time_stamps = []
    for y in range(dqn_model.batch_size): #Instead of running two if loops for x * million steps, we start off by getting enough experience for the first training
        dqn_model.Training_step()
    dqn_model.optimizeDDQN()
    for x in range(n): 
        dqn_model.Training_step()
        if dqn_model.t % dqn_model.T_train == 0:
            time_start = time.perf_counter()
            dqn_model.optimizeDDQN()
            time_stamps = time_stamps + [time.perf_counter() - time_start]
        if dqn_model.t % dqn_model.T_target == 0:
            dqn_model.synchronize()
        if dqn_model.t%dqn_model.T_evaluate == 0:
            dqn_model.follow_Optimal()
    return time_stamps
env.reset()




def calculate_confi(list):
	list = np.array(list)
	std = np.std(list)
	mean = np.average(list)
	confi = [(mean - 1.96* std/math.sqrt(n)),(mean - 1.96* std/math.sqrt(n))]
	return confi
	
time_stamps_full = np.array(double_DDQN_TrainingLoop(dqnFULL,(n+1)))
time_mean_full = np.average(time_stamps_full)
time_end_full = calculate_confi(time_stamps_full)
time_eval_full = calculate_confi(dqnFULL.eval_times)
score_eval_full = dqnFULL.true_Score_Averages
print(f"FULLSPACE: \n Time from beginning to end 100000 fullspace: {time_end_full} + {time_mean_full}, Time to evaluate 100000 steps: {time_eval_full}\n True average score gained: {score_eval_full}")



time_stamps_xy = double_DDQN_TrainingLoop(dqnXY,(n+1))
time_mean_xy = np.average(time_stamps_xy)
time_end_xy = calculate_confi(time_stamps_xy)
time_eval_xy = calculate_confi(dqnXY.eval_times)
score_eval_xy = dqnXY.true_Score_Averages
print(f"XY: \n Time from beginning to end 100000 fullspace: {time_end_xy} + {time_mean_xy}, Time to evaluate 100000 steps: {time_eval_xy}\n True average score gained: {score_eval_xy}")



