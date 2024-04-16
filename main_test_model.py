## import needed libraries 
import GameEnv
import pygame
import numpy as np

from ddqn_keras import DDQNAgent

from collections import deque
import random, math


## The maximum play time for each game.
TOTAL_GAMETIME = 100000
## The maximum number of simulation episodes.
N_EPISODES = 10000
## The number of simulation episodes before updating the target model.
REPLACE_TARGET = 10

## Creates an instance of the game environment.
game = GameEnv.RacingEnv()
game.fps = 60

GameTime = 0 
GameHistory = []
renderFlag = True

## Initializes a DDQNAgent object
ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=8, epsilon=0.02, epsilon_end=0.01, epsilon_dec=0.999, replace_target= REPLACE_TARGET, batch_size=64, input_dims=19,fname='ddqn_model.h5')

##  Loads the trained DQN model.
ddqn_agent.load_model()

##  Updates the evaluation network and target network parameters in the model.
ddqn_agent.update_network_parameters()

ddqn_scores = [] ##  A list to store the scores achieved in each simulation episode.
eps_history = [] ##  A list to store the epsilon (exploration rate) values during 

""" utilizing the trained DQN model to control the car in the gaming environment. """
def run():
    ## save 100 the most recent score 
    #scores = deque(maxlen=100) (optional)

    for e in range(N_EPISODES):
        #reset env to start new episode
        game.reset()

        done = False
        score = 0
        counter = 0

        gtime = 0 # set game time back to 0

        #first step
        observation_, reward, done = game.step(0)
        observation = np.array(observation_)

        while not done:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    run = False
                    return

            #new
            action = ddqn_agent.choose_action(observation) # Uses the DQN model to choose an action based on the current state (observation).
            ##  Performs the chosen action in the game environment and receives the next state (observation_), reward, and done flag.
            observation_, reward, done = game.step(action)
            observation_ = np.array(observation_)

            if reward == 0: ## This is a countdown if no reward is collected the car will be done within 100 ticks
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward ## update reward

            observation = observation_ ## update observation list 

            gtime += 1 ## updata simulation time if TOTAL_GAMETIME = 10000 we end simlation

            if gtime >= TOTAL_GAMETIME:
                done = True

            if renderFlag:
                game.render(action)



run()