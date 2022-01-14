import pandas as pd
import numpy as np

#Creates a pd.DataFrame object based on a CSV file, which it then turns into a list of numpy arrays containing only the numerical values of each column
def columns_To_Arrays(fileName):
    df = pd.read_csv(fileName)
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