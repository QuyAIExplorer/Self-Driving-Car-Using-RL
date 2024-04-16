from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf

"""The ReplayBuffer class is used to store and sample experiences for training the DDQN agent."""
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        ## max_size: Maximum size of the replay buffer.
        ## input_shape: Shape of the input state.
        ## n_actions: Number of possible actions.
        ## discrete: Boolean value indicating whether the action space is discrete or continuous.
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        
    ## Stores a transition in the replay buffer.
    def store_transition(self, state, action, reward, state_, done):
        ## state: Current state
        ## action: Action taken.
        ## reward: Reward received
        ## state_: Next state
        ## done: Boolean value indicating whether the episode is done.
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        ## store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        ## store value of actions , but not one hot encoding form
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1
    """ Take a transition batch from replay buffer. """
    def sample_buffer(self, batch_size):
        ## batch_size: size of batch
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class DDQNAgent(object): ## Q(s, a) = Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.999995,  epsilon_end=0.10,
                 mem_size=25000, fname='ddqn_model.h5', replace_target=25):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
       
        self.brain_eval = Brain(input_dims, n_actions, batch_size)
        self.brain_target = Brain(input_dims, n_actions, batch_size)

    ## Stores a transition in the replay buffer.
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    ## Chooses an action based on the epsilon-greedy policy.
    def choose_action(self, state):

        state = np.array(state)
        state = state[np.newaxis, :]

        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.brain_eval.predict(state)
            action = np.argmax(actions)

        return action
    """ Performs the Q-learning update using a batch of samples from the replay buffer."""
    def learn(self):
        ## checks if the replay buffer has enough data to perform the learning process. If not, the method stops without doing anything.
        if self.memory.mem_cntr > self.batch_size: 
            ##  randomly samples a batch of data from the replay buffer
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
            
            ##  creates an array containing the possible action values.
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)
            
            ## predicts the Q-values for the new states using the target network (brain_target).
            q_next = self.brain_target.predict(new_state)
            
            ##  predicts the Q-values for the new states using the evaluation network (brain_eval).
            q_eval = self.brain_eval.predict(new_state)
            
            ##  predicts the Q-values for the current states using the evaluation network (brain_eval).
            q_pred = self.brain_eval.predict(state)
            
            ##  finds the indices of the actions with the highest Q-values in the predicted Q-values for the new states.
            max_actions = np.argmax(q_eval, axis=1)
            
            ## copies the predicted Q-values for the current states to use as the target for the update.
            q_target = q_pred
            
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            
            ## updates the target Q-values (q_target) for the selected state-action pairs. 
            # Q(s, a) = r + γ * max_a' Q(s', a')
            q_target[batch_index, action_indices] = reward + self.gamma*q_next[batch_index, max_actions.astype(int)]*done
            ## trains the evaluation network (brain_eval) using the current states and the updated target Q-values.
            _ = self.brain_eval.train(state, q_target)

            """ updates the exploration rate (epsilon) by decreasing it gradually over time, 
                ensuring a balance between exploration and exploitation during training."""
            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    ## Updates the target network with the evaluation network's weights.
    def update_network_parameters(self):
        self.brain_target.copy_weights(self.brain_eval)
    ## Saves the trained model 
    def save_model(self):
        self.brain_eval.model.save(self.model_file)
    ## Loads a trained model from a file
    def load_model(self):
        self.brain_eval.model = load_model(self.model_file)
        self.brain_target.model = load_model(self.model_file)
       
        if self.epsilon == 0.0:
            self.update_network_parameters()

""" Representing the neural network model used in the DQN agent."""
class Brain:
    def __init__(self, NbrStates, NbrActions, batch_size = 256):
        self.NbrStates = NbrStates
        self.NbrActions = NbrActions
        self.batch_size = batch_size
        self.model = self.createModel()
        
    ## model i have designed for train agent 
    def createModel(self): 
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu)) #prev 256 
        model.add(tf.keras.layers.Dense(self.NbrActions, activation=tf.nn.softmax))
        model.compile(loss = "mse", optimizer="adam")

        return model
    
    ## x is agent's states and y is predicted Q-values 
    def train(self, x, y, epoch = 1, verbose = 0):
        self.model.fit(x, y, batch_size = self.batch_size , verbose = verbose)
        
    """ This method takes a state (s) as input and returns the predicted Q-values for all possible actions. 
        It uses the neural network model to perform the prediction."""
    def predict(self, s):
        return self.model.predict(s)

    """ This method is similar to 'predict', but it takes a single state (s) as input and returns the predicted Q-values as a flattened array. 
        It reshapes the input state to match the expected shape of the model input."""
    def predictOne(self, s):
        return self.model.predict(tf.reshape(s, [1, self.NbrStates])).flatten()
    
    """This method copies the weights of the trainable variables from another 'Brain' object (TrainNet) to the current 'Brain' object. 
It iterates over the trainable variables of both models and assigns the values from TrainNet to the corresponding variables in the current model. 
    This is typically used to update the target network with the weights of the evaluation network in the DQN algorithm."""
    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())