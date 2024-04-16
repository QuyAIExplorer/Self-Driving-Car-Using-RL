## Nguyễn Công Quý - 20134022 (Group 8)
## Nguyễn Tấn Hoàng - 20134015

""" Reference Resources: 
    Self-Driving AI Car Simulation in Python : https://www.youtube.com/watch?v=Cy155O5R1Oo&list=LL&index=2&t=33s 
    Self Driving Car Neural Network - with Python and NEAT : https://www.youtube.com/watch?v=cFjYinc465M&list=LL&index=3
    Deep Q Learning Explained - Making a Self-Driving Car (Part 1) : https://www.youtube.com/watch?v=WQ1bQAzBjPM&list=LL&index=5
    Deep Q Learning in Python - Making a Self-Driving Car (Part 2) : https://www.youtube.com/watch?v=khxoUDAkF98&t=2s
    Deep Q Learning in Python - Making a Self-Driving Car (Part 3) : https://www.youtube.com/watch?v=-XKZ9dERKM8
    Creating a Self Driving Car in Python : https://www.youtube.com/watch?v=uJKpCfX88A8&list=LL&index=4
"""

## Import needed libraries
import GameEnv
import pygame
import numpy as np
from ddqn_keras import DDQNAgent
from collections import deque
import random, math

## Set up parameters
TOTAL_GAMETIME = 10000 # Max game time for one episode
N_EPISODES = 10000 # Number of training episodes
REPLACE_TARGET = 50  # Number of episodes to update target model

game = GameEnv.RacingEnv() # Initialize game environment
game.fps = 60  # Set frame rate

GameTime = 0  # Game time counter
GameHistory = [] # Store game state history
renderFlag = False # # Flag to render the game

## ## Initializes a DDQNAgent object. The arguments of DDQNAgent determine important training parameters
ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=8, epsilon=1.00, epsilon_end=0.10, epsilon_dec=0.9995, replace_target= REPLACE_TARGET, batch_size=512, input_dims=19)


ddqn_scores = [] # A list to store the scores achieved in each simulation episode.
eps_history = [] # A list to store the epsilon (exploration rate) values during 

def run():

    for e in range(N_EPISODES):
        
        game.reset() #reset environment to start a new episode

        done = False
        score = 0
        counter = 0
        
        # initialize observation list
        observation_, reward, done = game.step(0)
        observation = np.array(observation_)

        gtime = 0 # set game time back to 0
        
        renderFlag = True # if you want to render every episode set to true

        if e % 10 == 0 and e > 0: # render every 10 episodes
            renderFlag = True

        while not done:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    return

            action = ddqn_agent.choose_action(observation) # Uses the DQN model to choose an action based on the current state (observation).
            ##  Performs the chosen action in the game environment and receives the next state (observation_), reward, and done flag.
            observation_, reward, done = game.step(action)
            observation_ = np.array(observation_)

            # This is a countdown if no reward is collected the car will be done within 100 ticks
            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward ## update reward

            ddqn_agent.remember(observation, action, reward, observation_, int(done)) ## Stores a transition in the replay buffer.
            observation = observation_
            ddqn_agent.learn() ## Update the neural network for predicting Q-values 
            
            # Update simulation time if TOTAL_GAMETIME = 10000 we end simlation
            # Increase TOTAL_GAMETIME if we want to increase training time
            gtime += 1

            if gtime >= TOTAL_GAMETIME:
                done = True

            if renderFlag:
                game.render(action)

        eps_history.append(ddqn_agent.epsilon) ## Append the current 'epsilon' value (exploration rate) to the 'eps_history' list. 
        ddqn_scores.append(score) ## Append the current episode's score to the 'ddqn_scores' list, which stores the scores achieved during training.
        """ Calculate the average score over the last 100 episodes using the ddqn_scores list. 
            This provides an indication of the agent's overall performance."""
        avg_score = np.mean(ddqn_scores[max(0, e-100):(e+1)]) 
        
        """ Check if the current episode number is divisible by REPLACE_TARGET and greater than REPLACE_TARGET. 
            This condition is used to determine when to update the target network. 
            The target network is a separate copy of the main network that is periodically updated to stabilize the training process."""
        if e % REPLACE_TARGET == 0 and e > REPLACE_TARGET:
            ddqn_agent.update_network_parameters()
        """ After every 10 episodes save model once 
            The model is saved periodically to capture the progress and allow for resuming training or inference later."""
        if e % 10 == 0 and e > 10:
            ddqn_agent.save_model()
            print("save model")
        ##  Print the training progress information
        print('episode: ', e,'score: %.2f' % score,
              ' average score %.2f' % avg_score,
              ' epsolon: ', ddqn_agent.epsilon,
              ' memory size', ddqn_agent.memory.mem_cntr % ddqn_agent.memory.mem_size)   

run() 