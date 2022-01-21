import pandas as pd
import numpy as np
import torch
import time
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

#Creates a pd.DataFrame object based on a CSV file, which it then turns into a list of numpy arrays containing only the numerical values of each column
def columns_To_Arrays(fileName):
    df = pd.read_csv(fileName,index_col=[0])
    arrays_list = []
    for column in df.columns:
        arr = pd.to_numeric(df[column], errors="coerce").dropna()
        arr = arr.to_numpy()
        arrays_list = arrays_list + [arr]
    return arrays_list

#Essentially just an iterator placeholder
def calculator(arrays_list,function):
    results_List = []
    for arr in arrays_list:
        if function == "mean":
                result = np.mean(arr)
                results_List = results_List + [result]
        elif function == "std":
                result = np.std(arr)
                results_List = results_List + [result]
        elif function == "lengths":
            result = len(arr)
            results_List = results_List + [result]
    return np.array(results_List)

#Calculates the 95% confidence intervals
def confidence_From_Arrays(arrays_list,isSingle = False):
    confidence_Intervals = []

    if isSingle:
        mean = np.mean(arrays_list)
        std = np.std(arrays_list)
        n = np.shape(arrays_list)[0]
        print(np.shape(arrays_list))
    else:
        means = calculator(arrays_list,"mean")
        stds = calculator(arrays_list,"std") #stds = Standard deviations
        n = calculator(arrays_list,"lengths")



    lower_CI = means - 1.96 * stds/np.sqrt(n)
    upper_CI = means + 1.96 * stds/np.sqrt(n)

    for idx,CI in enumerate(lower_CI): #Put the mean in the middle
        confidence_Intervals.append([CI,means[idx],upper_CI[idx]])

    return confidence_Intervals

def dataFrame_to_Confidence(csv):
    CI = confidence_From_Arrays(columns_To_Arrays(csv))
    return CI

#Returns an interval for a single attribute for every dataset
def compare_CI_Tabular(csv_files,index):
    confLists = []
    comparative_CI_List = []
    for idx, csv in enumerate(csv_files):
        CI = dataFrame_to_Confidence(csv)
        comparison = CI[index]
        confLists = confLists + [CI]
        comparative_CI_List = comparative_CI_List + [comparison]
    return comparative_CI_List


def normalize_Heat_Map(heatMap):
    new_heatMap = (heatMap - heatMap.mean()) / heatMap.std()
    return new_heatMap
"""Takes a DQNAgent object and gathers a heatmap

n = Number of steps
max_same_State = Number of times the agent can remain in the same state without repsawning
path = Where to save it
"""
def get_Heat_Map(dqn,n,max_same_state,path,normalized = False):
    k = 0
    heat_map = np.zeros((10,10))
    nr_same = 0
    state = torch.tensor((dqn.env.x,dqn.env.y,dqn.env.has_key),dtype=torch.float32)
    for x in range(n):
        max_a = torch.argmax(dqn.pNet.forward(state)).item() #Get the optimal policy
        move = dqn.actions_available[max_a] #Get the optimal action
        stepTuple = dqn.env.step(move) #Step in the environment
        stateNew = torch.tensor(stepTuple[0], dtype=torch.float32)
        dqn.env.render()
        time.sleep(10)
        heat_map[dqn.env.y][dqn.env.x] = heat_map[dqn.env.y][dqn.env.x] + 1
        if torch.all(state.eq(stateNew)):
            nr_same += 1
        state = stateNew
        if stepTuple[2] or nr_same>max_same_state:
            dqn.env.reset()
            state = torch.tensor((dqn.env.x,dqn.env.y,dqn.env.has_key),dtype=torch.float32)
            nr_same = 0
    heatMap = pd.DataFrame(heat_map,columns=["a","b","c","d","e","f","g","h","i","j"])
    if normalized == True:
        heatMap = normalize_Heat_Map(heatMap)
    heatMap.to_csv(path)
    return heatMap

"""
Will make a Catplot based on the standard DataFrame genered with get_Catplot_DF
"""

def make_Catplot(dataFrame):

    sns.set_theme(style="whitegrid")

    # Draw a pointplot to show pulse as a function of three categorical factors
    CatPole = sns.catplot(x="action_size", y="value", hue="value_type",col="DQN_type",
                    capsize=.2, palette="YlGnBu_d", height=6, aspect=.75,
                    kind="point", data=dataFrame)
    print("Catpole initialized")
    CatPole.despine(left=True)
    print("Catpole returned")
    return CatPole


def make_Value_dict(value_array,value_type,action_space,DQN_type):
    value_length = len(value_array)
    dict = {"value": value_array,"value_type":np.full(value_length,value_type), "action_size": np.full(value_length,action_space),"DQN_type": np.full(value_length,DQN_type)}

    return dict
"""

Takes a csv file in the format (R|V|E|L)

And outputs a dataframe in the format (value|value_type|action_size|DQN_type)

"""
def get_Catplot_DF(csv,action_space = None,DQN_type = None):
    arrays = columns_To_Arrays(csv)

    reward_array = arrays[0]
    true_value_array = arrays[1]
    estimate_array = arrays[2]
    if action_space == None and DQN_type == None:
        action_space,DQN_type = auto_read(csv)

    reward_dict = make_Value_dict(reward_array,"Reward",action_space,DQN_type)
    true_value_dict = make_Value_dict(true_value_array,"True Value",action_space,DQN_type)
    estimate_dict = make_Value_dict(estimate_array,"Estimate",action_space,DQN_type)

    reward_DF = pd.DataFrame.from_dict(reward_dict)
    true_value_DF = pd.DataFrame.from_dict(true_value_dict)
    estimate_DF = pd.DataFrame.from_dict(estimate_dict)

    catplot_DF = pd.concat([reward_DF, true_value_DF, estimate_DF])

    return catplot_DF


""" Tries to extrapolate the type and action space of a path.

Only recognizes paths in the format of EXPERIMENTNAME_TYPE_ACTION SPACE(.csv)
"""


def auto_read(path):
    path = str(path) #Otherwise we'll get a "path" object
    path_list = path.split("_")
    print(path_list)
    action_space = int(path_list[-1][0]) #might contain csv

    type_list = path_list[-2]
    if "DDQN" in type_list: #DQN is a part of DDQN, therefore we check for DDQN first
        DQN_type = "DDQN"
    elif "DQN" in type_list:
        DQN_type = "DQN"
    else:
        print("UNABLE TO READ DQN OR DDQN")
        return

    return action_space,DQN_type


""" 
Takes a folder of run data
"""
def Big_Data_CatPlot(folder_path,action_spaces = None, DQN_types = None):
    file_paths = Path(folder_path).glob("*")
    dataFrame_appendix = []
    for file in file_paths:
        if file.is_file():
            if action_spaces == None and DQN_types == None:
                action,DQN = auto_read(file) #Get action spaces and DQN types
                temporary_DF = get_Catplot_DF(file, action, DQN) #Create a catplot dataframe
                temporary_DF = temporary_DF.groupby(["value_type","DQN_type"],as_index=False).mean()
                dataFrame_appendix = dataFrame_appendix + [temporary_DF] #add to the appendices

    big_Frame = pd.concat(dataFrame_appendix) #Turn into one big CatPlot :)
    print("Done with appending DataFrames")
    return  big_Frame


