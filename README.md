Purpose:

This is the reposotory containing the scripts and (the most important) data used in my 3 week project in Introduction to Intelligent Systems.

Descriptions:
Main.py: Contains the actual training- and evaluation loops.
InferentialStatistics.py: Contains the tools neccessary for doing inferential- but also descriptive statistics on the data.
QModels.py: Contains the actual DDQN and DQN model-classes.
GridWorlds: Contains a modified version of the GridWorld game. In this version actions are given in the form of [STR_direction,number_of_times_to_repeat] rather than just [STR_direction]
Eg. ["left",2] -> Go "left" 2 times.
board(N).txt: A text-file with the board for the given seed.


Conditions:

In order to run the GridWorlds.py script all images and board-text-files must be in the same folder as GridWorlds.py.
