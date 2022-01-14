import pandas as pd
import numpy as np
import DataProcessing as dp

time_names = [r"C:\Users\benja\PycharmProjects\IISProject\ErrorPilotStudy\Pilot_Study_ErrorsDDQN_2",r"C:\Users\benja\PycharmProjects\IISProject\ErrorPilotStudy\Pilot_Study_ErrorsDQN_2",r"C:\Users\benja\PycharmProjects\IISProject\ErrorPilotStudy\Pilot_Study_ErrorsDDQN_4",r"C:\Users\benja\PycharmProjects\IISProject\ErrorPilotStudy\Pilot_Study_ErrorsDQN_4",r"C:\Users\benja\PycharmProjects\IISProject\ErrorPilotStudy\Pilot_Study_ErrorsDDQN_6",r"C:\Users\benja\PycharmProjects\IISProject\ErrorPilotStudy\Pilot_Study_ErrorsDQN_6"]#The files are next to eachother for a better comparison


over_ests = []
std_over_ests = []
for name in time_names:
    col_arr = dp.columns_To_Arrays(name)
    value = col_arr[1]
    estimate = col_arr[2]
    over_estimation = estimate - value
    over_ests = over_ests+ [over_estimation]

print(dp.confidence_From_Arrays(over_ests))