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
""""Settings for the experiment:
seed = Environment seed (1-6)
number_actions = Number of actions possible
names = [target network's name,policy network's name]
layers_shape = (mXn) shape of the fully connected linear layer
sample_size = How many samples of real returned reward to be evaluated.
file_name = Name of the data-file containing (average_true_returned_rewards,average_overestimation,average_loss)
"""

def calculate_confi(list):
	list = np.array(list)
	std = np.std(list)
	mean = np.average(list)
	confi = [(mean - 1.96* std/math.sqrt(n)),mean,(mean - 1.96* std/math.sqrt(n))]
	return confi

def createActionSpace(n):
    original_Action_Space = ["left", "right", "up", "down"]
    new_Action_Space = []
    for x in range(1,n):
        for action in original_Action_Space:
            new_Action_Space.append([action, x])


    return new_Action_Space


experiment_Names_list = ["XY_n1","XY_n2","XY_n3","Fullspace_n1","Fullspace_n2","Fullspace_n3"]
env = GridWorld(seed=1)
for idx,exp_Nr in enumerate(experiment_Names_list): #Experiment main loop
    n = 10
    a = (idx%3 +1 *2) #2,4,6
    actions = createActionSpace(a)
    state1 = torch.tensor((env.x,env.y,env.has_key),dtype=torch.float32) #board + has_key
    state2 = torch.tensor((env.board + [env.has_key]),dtype=torch.float32).flatten()
    states = [state1,state2]
    names1 = [None,None]
    path = f"PilotNetwork{exp_Nr}"
    Network_Architecture = [(520,1)]
    if idx<3:
        dqnXY = DQNAgent(state_representation=state1, actions_List=actions, layers_shape=(520,1), env=env, load_Name=(names1), save_Name=(names1),
                         epsilon_start=1, eval_Sample_Sizes=n, T_evaluate=n,feature_type="XY")
    else:
        dqnXY = DQNAgent(state_representation=state2, actions_List=actions, layers_shape=(520,1), env=env, load_Name=(names1), save_Name=(names1),
                        epsilon_start=1, eval_Sample_Sizes=n, T_evaluate=n)



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



    time_train = double_DDQN_TrainingLoop(dqnXY, n)
    time_train_CI = calculate_confi(time_train)
    true_value = dqnXY.true_Values
    true_Value_CI = calculate_confi(dqnXY.true_Values)
    eval_times = dqnXY.greedy_Estimates
    eval_CI = calculate_confi(dqnXY.greedy_Estimates)
    losses = dqnXY.losses
    losses_CI = calculate_confi(dqnXY.losses)
    raw_scores = dqnXY.rewards_List
    raw_scores_CI = calculate_confi(dqnXY.rewards_List)

    print(f"Experiment {exp_Nr}:\n Average time per training: {time_train_CI} \n Average rewards with discount returned per (s-a) pair: {true_Value_CI}\n Average time per evaluation: {eval_CI} \n Average scores without discount per (s-a) pair: {raw_scores_CI} \n Average losses {losses_CI}")
    print(dqnXY.get_State())
    list_o_headers = [f"Experiment Nr","Time per training","Value per (s-a)","Time per evaluation","Score without discount","training loss"]
    list_o_data = [exp_Nr,time_train,true_value,eval_times,raw_scores,losses]


    with open(path,"w") as f:
        writer = csv.writer(f)
        writer.writerow(list_o_headers)
        writer.writerow(list_o_data)