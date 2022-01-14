import numpy as np
import torch
import random
import copy
import time
import os


class OUTDATED:
    """
    This is the main DQN class.

    Methods:
        InitializeNetwork() = Builds the policy network and deep copies it unto the target network.

    Parameters:

    actions_num = Amount of possible actions, also the size of the output layer.
    actions_Lists = All possible moves (used for selecting greedy policy and stepping)
    state_representation = A tensor representation of the environment state, will be used to initialize the size of the input layer of the NN
    feature_type = "XY" or else fullspace by default
    layers_shape = Shape of the hidden layers, in the following format:
    (w,d), w = width, d = depth.
    memory_size = Number of experience tuples held in the experience memory
    batch_size = Size of each minibatch
    learning_rate = Learning rate used for backpropagation
    T_train = Timesteps befor training the policy network
    T_target = Timesteps befor synchronising the target network, only used in vanilla DQN
    T_evaluate = Timesteps before evaluating
    eval_Sample_Sizes = How many samples we need per evaluation
    gamma = The discount factor for calculating the target in the bellman equation
    epsilon_start = For use in the epsilon-greedy exploration/exploitation strategy
    epsilon_decay = Decay rate for epsilon per 1000 steps
    epsilon_final = Final value of epsilon
    env = An object containing the game environment
    load_Name = A tuple containing the name of the models' state dictionary in the format (tNetPath,pNetPath) - it must be stored in the same folder as the qTest.py file
    save_Name = A string containing the name of the models' state dictionary in the format (tNetPath,pNetPath) - it will be save in the same folder as q.Test.py


    """

    def __init__(self,
                 actions_num=4,
                 actions_List = None,
                 state_representation=None,
                 feature_type = None,
                 layers_shape=(10, 10),
                 activation_function=torch.nn.ReLU(),
                 memory_size=100000,
                 batch_size=32,
                 T_train=4,
                 T_target=10000,
                 T_evaluate = 10000,
                 eval_Sample_Sizes = 1000,
                 gamma=0.99,
                 epsilon_start=1,
                 epsilon_decay=(0.9) / (1000),
                 epsilon_final=0.1,  # Based on the Van Hasselt DDQN article
                 reward_System=(-10, 10 * 9, 0, 0),
                 learning_rate=0.00025,
                 env=None,
                 load_Name=None,
                 save_Name=None):

        # Network features
        self.actions_num = actions_num
        self.actions_available = actions_List
        self.state_representation = state_representation
        self.layers_shape = layers_shape
        self.activation_function = activation_function

        # Network Setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_NameTarget, self.load_NamePolicy = load_Name
        self.save_NameTarget, self.save_NamePolicy = save_Name
        self.initializeNeuralNetworks()

        # Training
        self.learning_rate = learning_rate
        self.T_train = T_train
        self.T_target = T_target
        self.optimizerP = torch.optim.RMSprop(lr=self.learning_rate, params=self.tNet.parameters(),
                                              momentum=0.95)  # This is the optimizer mentioned in the papers
        self.lossFn = torch.nn.MSELoss(reduction="sum")

        # Experience Replay Memory
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.EMR = ExperienceMemoryReplay(memory_size=self.memory_size,
                                          batch_size=self.batch_size)  # EMR, it's short for ExperienceMemoryReplay ;)

        # Policy and Value Evaluation
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_final = epsilon_final

        # Reward system
        self.die_Reward = reward_System[0]
        self.win_Reward = reward_System[1]
        self.stay_Reward = reward_System[2]
        self.survive_Reward = reward_System[3]

        # Incremental values
        self.t = 0
        self.epsilon = self.epsilon_start

        # Environment
        self.env = env
        self.feature_type = feature_type

        # For evaluation of the networks
        self.losses = []
        self.T_evaluate = T_evaluate
        self.eval_Sample_Size = eval_Sample_Sizes
        self.k = 0 #Steps since terminal discount, used for calculating the discount.
        self.true_Scores_Averages = [] #For storing the average after each evaluation, idx * eval_Sample_Sizes = timestep at which we evaluated.
        self.true_Scores = [] #For storing each reward with discount - only used for pilot studies
        self.raw_Scores = [] #For the actual game score
        self.eval_Times = []


    """Initializes the target network and deep copies the policy network unto this.
    
    Target network architecture = [input_layers]-[activation_function][layers_shape*]-[actions_num] 
    *Each layer in the connected linear layer is seperated by the activation function
    
    """
    def initializeNeuralNetworks(self):
        input_width = len(self.state_representation.flatten())  # Width of the input layer

        self.tNet = NeuralNetwork(layers_shape=self.layers_shape, output_width=self.actions_num,
                                  input_width=input_width, activation_function=self.activation_function,
                                  device=self.device)  # Target network
        self.pNet = copy.deepcopy(self.tNet)  # Policy network

        if self.load_NamePolicy != None or self.load_NameTarget != None:
            self.loadNeuralNetowrks()

        self.tNet.stack.to(self.device)
        self.pNet.stack.to(self.device)

    def loadNeuralNetowrks(self):
        if self.load_NameTarget != None:
            try:  # In case we wish to automate, but we haven't trained under that specific model name yet
                self.tNet = torch.load(self.load_NameTarget, map_location=self.device)
            except:
                print(f"Model named {self.load_NameTarget} could not be loaded; Will proceed with randomized weights")
        if self.load_NamePolicy != None:
            try:
                self.pNet = torch.load(self.load_NamePolicy, map_location=self.device)
            except:
                print(f"Model named {self.load_NamePolicy} could not be loaded; Will proceed with randomized weights")

    """For use with the frozenLake environment

    Arguments:
    env.desc = a NxN np.array
    env.s = A single integer representing the index at which the agent is

    Output: A one dimensional tensor representing the ASCII-char codes for each letter in the grid.
    This way we get a numerical representation of the board, which we can use to train the DQN.
    """

    """
    Rewards:
    - Dying: -10 * NxN * gamma
    - Winning: 10* NxN * gamma
    - Surviving: 1

    We add the winning and dying bonus based on the tile size, as this will ensure, that the optimal policy will be to win the game in case we have big boards.

    """

    def evaluateReward(self, tuple):

        if tuple[2]:
            if tuple[1] > 0:  # We win
                return self.win_Reward  # The input depth is the number of tiles on the board i.e NxN
            else:  # We lose
                return self.die_Reward
        if self.S_t[-1] == tuple[0] or self.S_t[-1]:
            return self.stay_Reward
        else:
            return self.survive_Reward

    # Chooses between a greedy or exploratory action
    def greedVsExp(self):

        if self.epsilon > random.random():  # Choose exploratory
            return random.randint(0, self.actions_num - 1)
        else:  # Else choose greedy
            a = torch.argmax(self.pNet(self.S_t)).item()
            return a

    """This is the primary method for the agent to interact with its environment and collect experience.

    """

    def Training_step(self):
        # 1. Decay Epsilon
        if self.epsilon > self.epsilon_final:
            self.epsilon = self.epsilon - self.epsilon_decay

        # 2. Choose between exploitation or exploration (Greedy or exploratory action)
        self.S_t = torch.tensor((self.env.x,self.env.y,self.env.has_key),dtype=torch.float32)
        a = self.greedVsExp()


        # 3. Step in environment - we get a tuple in the format: ((x,y,has_key),Reward,Done?)
        stepTuple = self.env.step(self.actions_available[a])
        reward = stepTuple[1]
        done = stepTuple[2]
        S_t1 = torch.tensor(stepTuple[0],dtype=torch.float32)


        # 4. Save transition in the EMR
        exp = [self.S_t,a,reward,S_t1,done]
        self.EMR.storeExp(exp)

        # 5. Increment timestep
        self.t = self.t + 1

        # 6. If the game is done reset
        if stepTuple[2]:
            self.env.reset()

    # Follows the greedy policy in the environment, calculates the average true returned reward and stores it.
    def follow_Optimal(self):
        accumulated_score = 0
        score = 0
        for m in range(self.eval_Sample_Size):
            start_time = time.perf_counter()
            state = torch.tensor((self.env.x,self.env.y,self.env.has_key),dtype=torch.float32)  # Get the state
            max_a = torch.argmax(self.pNet.forward(state)).item()  # Get the greedy policy
            stepTuple = self.env.step(self.actions_available[max_a])  # Step in the environment
            value = stepTuple[1] * self.gamma ** self.k  # Calculate the value - discount
            score = score + stepTuple[1]
            self.k += 1  # Increment the current non-terminal counter
            accumulated_score = accumulated_score + value
            if stepTuple[2]:
                self.raw_Scores = self.raw_Scores + [score] #The accumulated score per game
                score = 0
                self.env.reset()
                self.k = 0
            self.eval_Times = self.eval_Times + [time.perf_counter() - start_time]
            self.true_Scores = self.true_Scores + [value]
        self.raw_Scores = self.raw_Scores + [score] #If we don't die on the last run




    def synchronize(self):
        self.tNet = copy.deepcopy(self.pNet)
        if self.save_NameTarget != None:
            torch.save(self.tNet, self.save_NameTarget)

    def optimizeDQN(self):
        # 1. Get sample batch and reset optimizer
        batch = self.EMR.sample()  # Returns a list of tuples in the format (S_t,a,R_t+1,S_t+1,Done?,Probability)
        self.optimizerP.zero_grad()

        # 2. Calculate the targets and quality predictions
        target_Tensor = []
        prediction_Tensor = []

        for exp in batch:
            maxTerm = self.gamma * max(self.pNet.forward(exp[3]))
            target_Tensor = target_Tensor + [exp[2] + maxTerm]
            prediction_Tensor = prediction_Tensor + [self.pNet.forward(exp[0])[exp[1]]]

        target_Tensor = torch.tensor(target_Tensor, dtype=torch.float32, requires_grad=True)
        prediction_Tensor = torch.tensor(prediction_Tensor, dtype=torch.float32, requires_grad=True)

        # 3. Calculate loss and backpropagate
        loss = self.lossFn(prediction_Tensor, target_Tensor)
        loss.backward()
        self.optimizerP.step()
        self.losses = self.losses + [loss.item()]

        # 4. Save the model
        if self.save_NameTarget != None:
            torch.save(self.pNet, self.save_NamePolicy)

    def optimizeDDQN(self):
        time_start = time.perf_counter()
        # 1. Get sample batch, reset optimizer and decide how many transitions to train the policy network with
        batch = self.EMR.sample()
        self.optimizerP.zero_grad()

        target_Tensor = []
        policy_Tensor = []

        # 2. Calculate policy and target
        for exp in batch:  # exp = (S_t,a,R_t+1,S_t+1,Done?,Prob)
            max_a = torch.argmax(self.pNet.forward(exp[3])).item()
            target_Tensor = target_Tensor + [exp[2] + self.gamma * self.tNet.forward(exp[3])[max_a]]
            policy_Tensor = policy_Tensor + [self.pNet.forward(exp[0])[exp[1]]]

        target_Tensor = torch.tensor(target_Tensor, dtype=torch.float32, requires_grad=True)
        policy_Tensor = torch.tensor(policy_Tensor, dtype=torch.float32, requires_grad=True)

        # 3. Calculate the losses and backpropagate
        lossP = self.lossFn(policy_Tensor, target_Tensor)
        lossP.backward()
        self.optimizerP.step()
        self.losses = self.losses + [lossP.item()]

        # 4. Save the models
        if self.save_NamePolicy != None:
            torch.save(self.pNet, self.save_NamePolicy)


class ExperienceMemoryReplay:

    def __init__(self,
                 memory_size,
                 batch_size
                 ):
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.memory = []  # Although the transitions are tuples, the memory is a list, as tuples are immutable

    """
    Stores experience tuples (transitions) in the EMR.

    If the EMR memory is full, it deletes the first tuple and appends the latest one to the end. 

    Arguments:

    transition: Tuple consisting of (S_t,a,R_t+1,S_t+1,Done?,P(S_t+1|a,S_t)
    """

    def storeExp(self, transition):
        if len(self.memory) > self.memory_size:
            del self.memory[0]

        self.memory = self.memory + [transition]

    def sample(self):
        return random.sample(self.memory, self.batch_size)


class NeuralNetwork(torch.nn.Module):
    def __init__(self,
                 layers_shape,
                 output_width,
                 input_width,
                 activation_function,
                 device):
        super().__init__()

        # Model parameters
        self.width = layers_shape[0]
        self.depth = layers_shape[1]
        self.output_width = output_width
        self.input_width = input_width
        self.activation_function = activation_function
        self.makeModel()

        # CUDA vs GPU
        self.device = device

    def makeModel(self):
        networkStack = []
        networkStack = networkStack + [(torch.nn.Linear(self.input_width, self.width))]  # Input layer
        hiddenLayers = [torch.nn.Linear(self.width,self.width)] * self.depth  # There is no non-linear function between the last hidden layer and the output
        networkStack = networkStack + hiddenLayers + [self.activation_function]
        networkStack = networkStack + [(torch.nn.Linear(self.width, self.output_width))]  # Output layer
        self.stack = torch.nn.Sequential(*networkStack)

    def forward(self, input):
        input = input.to(self.device)
        output = self.stack(input)
        return output


import numpy as np
import torch
import random
import copy
import time
import os


class DQNAgent:
    """
    This is the main DQN class.

    Methods:
        InitializeNetwork() = Builds the policy network and deep copies it unto the target network.

    Parameters:

    actions_num = Amount of possible actions, also the size of the output layer.
    actions_Lists = All possible moves (used for selecting greedy policy and stepping)
    state_representation = A tensor representation of the environment state, will be used to initialize the size of the input layer of the NN
    feature_type = XY or fullspace by default
    layers_shape = Shape of the hidden layers, in the following format:
    (w,d), w = width, d = depth.
    memory_size = Number of experience tuples held in the experience memory
    batch_size = Size of each minibatch
    learning_rate = Learning rate used for backpropagation
    T_train = Timesteps befor training the policy network
    T_target = Timesteps befor synchronising the target network, only used in vanilla DQN
    T_evaluate = Timesteps before evaluating
    eval_Sample_Sizes = How many samples we need per evaluation
    gamma = The discount factor for calculating the target in the bellman equation
    epsilon_start = For use in the epsilon-greedy exploration/exploitation strategy
    epsilon_decay = Decay rate for epsilon per 1000 steps
    epsilon_final = Final value of epsilon
    env = An object containing the game environment
    load_Name = A tuple containing the name of the models' state dictionary in the format (tNetPath,pNetPath) - it must be stored in the same folder as the qTest.py file
    save_Name = A string containing the name of the models' state dictionary in the format (tNetPath,pNetPath) - it will be save in the same folder as q.Test.py


    """

    def __init__(self,
                 actions_num=4,
                 actions_List=None,
                 state_representation=None,
                 feature_type = None,
                 layers_shape=(10, 10),
                 activation_function=torch.nn.ReLU(),
                 memory_size=100000,
                 batch_size=32,
                 T_train=4,
                 T_target=10000,
                 T_evaluate=10000,
                 eval_Sample_Sizes=1000,
                 gamma=0.99,
                 epsilon_start=1,
                 epsilon_decay=(0.9) / (1000),
                 epsilon_final=0.1,  # Based on the Van Hasselt DDQN article
                 reward_System=(-10, 10 * 9, 0, 0),
                 learning_rate=0.00025,
                 env=None,
                 load_Name=None,
                 save_Name=None):

        # Network features
        self.actions_num = actions_num
        self.actions_available = actions_List
        self.state_representation = state_representation
        self.layers_shape = layers_shape
        self.activation_function = activation_function

        # Network Setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_NameTarget, self.load_NamePolicy = load_Name
        self.save_NameTarget, self.save_NamePolicy = save_Name
        self.initializeNeuralNetworks()

        # Training
        self.learning_rate = learning_rate
        self.T_train = T_train
        self.T_target = T_target
        self.optimizerP = torch.optim.RMSprop(lr=self.learning_rate, params=self.tNet.parameters(),
                                              momentum=0.95)  # This is the optimizer mentioned in the papers
        self.lossFn = torch.nn.MSELoss(reduction="sum")

        # Experience Replay Memory
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.EMR = ExperienceMemoryReplay(memory_size=self.memory_size,
                                          batch_size=self.batch_size)  # EMR, it's short for ExperienceMemoryReplay ;)

        # Policy and Value Evaluation
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_final = epsilon_final

        # Reward system
        self.die_Reward = reward_System[0]
        self.win_Reward = reward_System[1]
        self.stay_Reward = reward_System[2]
        self.survive_Reward = reward_System[3]

        # Incremental values
        self.t = 0
        self.epsilon = self.epsilon_start

        # Environment
        self.env = env
        self.feature_Type = feature_type

        # For evaluation of the networks
        self.losses = []
        self.T_evaluate = T_evaluate
        self.eval_Sample_Size = eval_Sample_Sizes
        self.true_Scores_Averages = []  # For storing the average after each evaluation, idx * eval_Sample_Sizes = timestep at which we evaluated.
        self.true_Values = np.array([])  # For storing each reward with discount - only used for pilot studies
        self.rewards_List = np.array([])  # For the actual game score
        self.greedy_Estimates = np.array([])

    """Initializes the target network and deep copies the policy network unto this.

    Target network architecture = [input_layers]-[activation_function][layers_shape*]-[actions_num] 
    *Each layer in the connected linear layer is seperated by the activation function

    """

    def initializeNeuralNetworks(self):
        input_width = len(self.state_representation.flatten())  # Width of the input layer

        self.tNet = NeuralNetwork(layers_shape=self.layers_shape, output_width=self.actions_num,
                                  input_width=input_width, activation_function=self.activation_function,
                                  device=self.device)  # Target network
        self.pNet = copy.deepcopy(self.tNet)  # Policy network

        if self.load_NamePolicy != None or self.load_NameTarget != None:
            self.loadNeuralNetowrks()

        self.tNet.stack.to(self.device)
        self.pNet.stack.to(self.device)

    def loadNeuralNetowrks(self):
        if self.load_NameTarget != None:
            try:  # In case we wish to automate, but we haven't trained under that specific model name yet
                self.tNet = torch.load(self.load_NameTarget, map_location=self.device)
            except:
                print(f"Model named {self.load_NameTarget} could not be loaded; Will proceed with randomized weights")
        if self.load_NamePolicy != None:
            try:
                self.pNet = torch.load(self.load_NamePolicy, map_location=self.device)
            except:
                print(f"Model named {self.load_NamePolicy} could not be loaded; Will proceed with randomized weights")

    """For use with the frozenLake environment

    Arguments:
    env.desc = a NxN np.array
    env.s = A single integer representing the index at which the agent is

    Output: A one dimensional tensor representing the ASCII-char codes for each letter in the grid.
    This way we get a numerical representation of the board, which we can use to train the DQN.
    """

    """
    Rewards:
    - Dying: -10 * NxN * gamma
    - Winning: 10* NxN * gamma
    - Surviving: 1

    We add the winning and dying bonus based on the tile size, as this will ensure, that the optimal policy will be to win the game in case we have big boards.

    """

    def evaluateReward(self, tuple):

        if tuple[2]:
            if tuple[1] > 0:  # We win
                return self.win_Reward  # The input depth is the number of tiles on the board i.e NxN
            else:  # We lose
                return self.die_Reward
        if self.S_t[-1] == tuple[0] or self.S_t[-1]:
            return self.stay_Reward
        else:
            return self.survive_Reward

    # Chooses between a greedy or exploratory action
    def greedVsExp(self):

        if self.epsilon > random.random():  # Choose exploratory
            return random.randint(0, self.actions_num - 1)
        else:  # Else choose greedy
            a = torch.argmax(self.pNet(self.S_t)).item()
            return a

    """This is the primary method for the agent to interact with its environment and collect experience.

    """
    def get_State(self):
        if self.feature_Type == "XY":
            state = torch.tensor((self.env.x,self.env.y,self.env.has_key),dtype=torch.float32)
        else:
            state = torch.tensor((self.env.board+[self.env.has_key]),dtype=torch.float32).flatten()
        return state

    def Training_step(self):
        # 1. Decay Epsilon
        if self.epsilon > self.epsilon_final:
            self.epsilon = self.epsilon - self.epsilon_decay

        # 2. Choose between exploitation or exploration (Greedy or exploratory action)
        self.S_t = self.get_State()
        a = self.greedVsExp()

        # 3. Step in environment - we get a tuple in the format: ((x,y,has_key),Reward,Done?)
        stepTuple = self.env.step(self.actions_available[a])
        reward = stepTuple[1]
        done = stepTuple[2]
        S_t1 = self.get_State()

        # 4. Save transition in the EMR
        exp = [self.S_t, a, reward, S_t1, done]
        self.EMR.storeExp(exp)

        # 5. Increment timestep
        self.t = self.t + 1

        # 6. If the game is done reset
        if stepTuple[2]:
            self.env.reset()

    #Collects one sample per step (Read the pseudocode)
    def follow_Optimal_Average(self):
        score = 0
        Greedy_estimate = 0
        k = 0 #Steps since terminal
        for m in range(self.eval_Sample_Size):
            Q_tuple = self.pNet.forward(self.get_State())
            greedy_Estimate = torch.max(Q_tuple).item()
            greedy_Action = torch.argmax(Q_tuple)
            stepTuple = self.env.step(self.actions_available[greedy_Action])
            reward = stepTuple[1]
            true_Value = reward * self.gamma ** k
            k += 1
            self.rewards_List.append(reward)
            self.true_Values.append(true_Value)
            self.greedy_Estimates.append(greedy_Estimate)
            if stepTuple[2]:
                k = 0
                self.env.reset()

    def synchronize(self):
        self.tNet = copy.deepcopy(self.pNet)
        if self.save_NameTarget != None:
            torch.save(self.tNet, self.save_NameTarget)

    def optimizeDQN(self):
        # 1. Get sample batch and reset optimizer
        batch = self.EMR.sample()  # Returns a list of tuples in the format (S_t,a,R_t+1,S_t+1,Done?,Probability)
        self.optimizerP.zero_grad()

        # 2. Calculate the targets and quality predictions
        target_Tensor = []
        prediction_Tensor = []

        for exp in batch:
            maxTerm = self.gamma * max(self.pNet.forward(exp[3]))
            target_Tensor = target_Tensor + [exp[2] + maxTerm]
            prediction_Tensor = prediction_Tensor + [self.pNet.forward(exp[0])[exp[1]]]

        target_Tensor = torch.tensor(target_Tensor, dtype=torch.float32, requires_grad=True)
        prediction_Tensor = torch.tensor(prediction_Tensor, dtype=torch.float32, requires_grad=True)

        # 3. Calculate loss and backpropagate
        loss = self.lossFn(prediction_Tensor, target_Tensor)
        loss.backward()
        self.optimizerP.step()
        self.losses = self.losses + [loss.item()]

        # 4. Save the model
        if self.save_NameTarget != None:
            torch.save(self.pNet, self.save_NamePolicy)

    def optimizeDDQN(self):
        time_start = time.perf_counter()
        # 1. Get sample batch, reset optimizer and decide how many transitions to train the policy network with
        batch = self.EMR.sample()
        self.optimizerP.zero_grad()

        target_Tensor = []
        policy_Tensor = []

        # 2. Calculate policy and target
        for exp in batch:  # exp = (S_t,a,R_t+1,S_t+1,Done?,Prob)
            max_a = torch.argmax(self.pNet.forward(exp[3])).item()
            target_Tensor = target_Tensor + [exp[2] + self.gamma * self.tNet.forward(exp[3])[max_a]]
            policy_Tensor = policy_Tensor + [self.pNet.forward(exp[0])[exp[1]]]

        target_Tensor = torch.tensor(target_Tensor, dtype=torch.float32, requires_grad=True)
        policy_Tensor = torch.tensor(policy_Tensor, dtype=torch.float32, requires_grad=True)

        # 3. Calculate the losses and backpropagate
        lossP = self.lossFn(policy_Tensor, target_Tensor)
        lossP.backward()
        self.optimizerP.step()
        self.losses = self.losses + [lossP.item()]

        # 4. Save the models
        if self.save_NamePolicy != None:
            torch.save(self.pNet, self.save_NamePolicy)


class ExperienceMemoryReplay:

    def __init__(self,
                 memory_size,
                 batch_size
                 ):
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.memory = []  # Although the transitions are tuples, the memory is a list, as tuples are immutable

    """
    Stores experience tuples (transitions) in the EMR.

    If the EMR memory is full, it deletes the first tuple and appends the latest one to the end. 

    Arguments:

    transition: Tuple consisting of (S_t,a,R_t+1,S_t+1,Done?,P(S_t+1|a,S_t)
    """

    def storeExp(self, transition):
        if len(self.memory) > self.memory_size:
            del self.memory[0]

        self.memory = self.memory + [transition]

    def sample(self):
        return random.sample(self.memory, self.batch_size)


class NeuralNetwork(torch.nn.Module):
    def __init__(self,
                 layers_shape,
                 output_width,
                 input_width,
                 activation_function,
                 device):
        super().__init__()

        # Model parameters
        self.width = layers_shape[0]
        self.depth = layers_shape[1]
        self.output_width = output_width
        self.input_width = input_width
        self.activation_function = activation_function
        self.makeModel()

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


