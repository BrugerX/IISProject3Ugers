Purpose:

This is the reposotory containing the scripts and (the most important) data used in my 3 week project in Introduction to Intelligent Systems.

Descriptions:

TrainingLoop: Contains an example of how to run a trainingloop.

HeatMapEvaluation: Contains an example of how to load a network trained on the GPU and then evaluate a heatmap on a CPU

QModels2.py: Contains the actual DDQN and DQN model-classes.

GridWorlds: Contains a modified version of the GridWorld game. In this version actions are given in the form of [STR_direction,number_of_times_to_repeat] rather than just [STR_direction].
Eg. ["left",2] -> Go "left" 2 times.
board(N).txt: A text-file with the board for the seed N.

DataProcessing: Contains all the necessary tools to extract data from a CSV saved in the format (rewards|true_values|estimates|losses) with a name of the format EXPERIMENTNAME(D)DQN_ACTIONTYPE and convert it into a catplot, sorted by value types,DQN_type and action space size.

Training: Contains various files for training a (D)DQN Network

Conditions:

In order to run the GridWorlds.py script all images and board-text-files must be in the same folder as GridWorlds.py.
