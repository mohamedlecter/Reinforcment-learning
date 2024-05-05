import logging
import random
from datetime import datetime
import numpy as np
from environment import GameStatus 
from models import AbstractModel


class QTable(AbstractModel):
    default_check_iterations_every = 5

    def __init__(self, game, **kwargs):
        super().__init__(game, name="QTableModel", **kwargs) # Create a new prediction model for 'game'.
        self.Q = dict()  # table with values for state and action 
        self.epsilon_values = []  # List to store epsilon values
        self.reward_history = []  # Store reward history

    def train(self, stop_at_convergence=False, **kwargs): # Train the model.
        gamma = kwargs.get("gamma ", 0.95) # gamma (discount rate) of 0.90
        epsilon = kwargs.get("epsilon", 0.1) # epsilon (exploration rate) preference for exploring of 0.10 
        exploration_decay = kwargs.get("exploration_decay", 0.995)  # exploration rate reduction after each random step of 0.995
        learning_rate = kwargs.get("learning_rate", 0.7) # learning rate (alpha) preference for using new knowledge of 0.7
        n_eval_episodes = max(kwargs.get("n_eval_episodes ", 100), 1) # number of training games to play
        check_iterations_every = kwargs.get("check_iterations_every", self.default_check_iterations_every) # check for convergence every # episodes
        
        # variables that are used for recording the training process
        total_reward = 0 # cumulative reward
        win_history = [] # history of win rate

        starting_points = list() # list of starting points
        start_time = datetime.now() # start time of training
        
        # this is where the training starts 
        for iteration in range(1, n_eval_episodes  + 1): 
            # if there are no starting points, copy the empty cells to the starting points list and choose a random starting point and then remove it from the list of starting points
            if not starting_points: 
                starting_points = self.environment.empty_cells.copy()
            starting_cell = random.choice(starting_points) 
            starting_points.remove(starting_cell) 

            state = self.environment.reset(starting_cell)
            state = tuple(state.flatten())

            while True:
                # choosing the action using the epsilon greedy method,
                # if the random number is less than the exploration rate, then the action is chosen randomly
                if np.random.random() < epsilon:
                    action = random.choice(self.environment.available_actions)
                else:
                    action = self.predict_value(state)
                    
                # get the next state, reward and status of the environment, and flatten the next state to a tuple    
                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())

                # calculate the maximum next Q value
                total_reward += reward
                
                # if the state and action are not in the Q table, then set the value to 0.0
                if (state, action) not in self.Q.keys(): 
                    self.Q[(state, action)] = 0.0
                    
                # calculate the Q value for the state and action using the formula
                #Q[s,a] = Q[s,a] + eta*(r + gamma*np.nanmax(Q[s_next,:]) - Q[s,a]) where eta is learning rate, gammma is discount factor
                max_next_Q = max([self.Q.get((next_state, a), 0.0) for a in self.environment.available_actions])
                
                # update the Q value for the state and action using the formula
                self.Q[(state, action)] += learning_rate * (reward + gamma  * max_next_Q - self.Q[(state, action)])
                
                # if the status is win or lose, then break the loop
                if status in (GameStatus.WIN, GameStatus.LOSE):  
                    break

                state = next_state

                self.environment.render_q_values(self)
                
            # append the total reward to the reward history    
            self.reward_history.append(total_reward)

            logging.info("iteration: {:d}/{:d} | status: {:4s} | e: {:.5f}"
                         .format(iteration, n_eval_episodes , status.name, epsilon))

            if iteration % check_iterations_every == 0: # checks for the win rate and if the model has won from all starting cells then stop learning
                all_wins, win_rate = self.environment.check_win_all(self)
                win_history.append((iteration, win_rate))
                if all_wins and stop_at_convergence:
                    logging.info("won from all starting cells, stop learning")
                    break

            epsilon *= exploration_decay  # explore less as training progresses
            self.epsilon_values.append(epsilon)  # Store epsilon value

        logging.info("iterations: {:d} | time spent: {}".format(iteration, datetime.now() - start_time))

        return self.reward_history, win_history, iteration, datetime.now() - start_time

    # get the Q values for all actions for a certain state to then predict the action to take based on the Q values 
    def q_values(self, state):
        if type(state) == np.ndarray:
            state = tuple(state.flatten())

        return np.array([self.Q.get((state, action), 0.0) for action in self.environment.available_actions])
    # predict the action to take based on the Q values 
    def predict_value(self, state):
        q_values = self.q_values(state)

        logging.debug("q[] = {}".format(q_values))

        actions = np.nonzero(q_values == np.max(q_values))[0]
        return random.choice(actions)
    
    # Method to get epsilon values
    def get_epsilon_values(self):
        return self.epsilon_values
    
    def get_reward_history(self):
        return self.reward_history