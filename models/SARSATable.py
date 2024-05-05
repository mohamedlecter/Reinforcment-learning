import logging
import random
from datetime import datetime
import numpy as np
from environment import GameStatus 
from models import AbstractModel


class SARSATable(AbstractModel):
    default_check_iterations_every = 5

    def __init__(self, game, **kwargs):
        super().__init__(game, name="SARSATable", **kwargs)  # Create a new prediction model for 'game'.
        self.Q = dict()  # table with values for state and action 
        self.epsilon_values = []  # List to store epsilon values
        self.reward_history = []  # Store reward history

    def train(self, stop_at_convergence=False, **kwargs):
        gamma = kwargs.get("gamma", 0.95)  # gamma (discount rate) of 0.95
        epsilon = kwargs.get("epsilon", 0.10)  # epsilon (exploration rate) preference for exploring of 0.10
        exploration_decay = kwargs.get("exploration_decay", 0.995)  # exploration rate reduction after each random step
        learning_rate = kwargs.get("learning_rate", 0.7)  # learning rate (alpha) preference for using new knowledge of 0.7
        n_eval_episodes = max(kwargs.get("n_eval_episodes", 1000), 1)  # number of training games to play
        check_iterations_every = kwargs.get("check_iterations_every", self.default_check_iterations_every)

        total_reward = 0  # cumulative reward
        win_history = []  # history of win rate

        starting_points = list()  # list of starting points
        start_time = datetime.now()  # start time of training

        # Loop over training episodes
        for iteration in range(1, n_eval_episodes + 1):
            # Choose a starting cell
            if not starting_points:
                starting_points = self.environment.empty_cells.copy()
            starting_cell = random.choice(starting_points)
            starting_points.remove(starting_cell)

            # Reset the environment to the starting cell
            state = self.environment.reset(starting_cell)
            state = tuple(state.flatten())

            # Choose initial action
            action = self.choose_action(state, epsilon)

            # Loop over steps in the episode
            while True:
                # Take the chosen action and observe the next state and reward
                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())

                # Update total reward
                total_reward += reward

                # Choose next action using epsilon-greedy policy
                next_action = self.choose_action(next_state, epsilon)

                # Update Q-value of the current state-action pair
                if (state, action) not in self.Q.keys():
                    self.Q[(state, action)] = 0.0

                # Update Q-value using SARSA update rule
                next_Q = self.Q.get((next_state, next_action), 0.0)
                self.Q[(state, action)] += learning_rate * (reward + gamma * next_Q - self.Q[(state, action)])

                # If episode ends, break the loop
                if status in (GameStatus.WIN, GameStatus.LOSE):
                    break

                # Update state and action for the next step
                state = next_state
                action = next_action
                
                self.environment.render_q_values(self)

            # Append total reward to reward history
            self.reward_history.append(total_reward)

            # Check for convergence every few episodes
            if iteration % check_iterations_every == 0:
                all_wins, win_rate = self.environment.check_win_all(self)
                win_history.append((iteration, win_rate))
                if all_wins and stop_at_convergence:
                    logging.info("won from all starting cells, stop learning")
                    break

            # Update epsilon for exploration rate decay
            epsilon *= exploration_decay
            self.epsilon_values.append(epsilon)

        logging.info("iterations: {:d} | time spent: {}".format(iteration, datetime.now() - start_time))

        return self.reward_history, win_history, iteration, datetime.now() - start_time

    def choose_action(self, state, epsilon):
        # Choose action based on epsilon-greedy policy
        if np.random.random() < epsilon:
            return random.choice(self.environment.available_actions)
        else:
            return self.predict_value(state)

    def predict_value(self, state):
        # Predict action based on Q-values using epsilon-greedy policy
        q_values = self.q_values(state)
        actions = np.nonzero(q_values == np.max(q_values))[0]
        return random.choice(actions)
    
    def q_values(self, state):
        # Get Q-values for all actions for a given state
        if type(state) == np.ndarray:
            state = tuple(state.flatten())
        return np.array([self.Q.get((state, action), 0.0) for action in self.environment.available_actions])

    def get_epsilon_values(self):
        # Get epsilon values over training episodes
        return self.epsilon_values

    def get_reward_history(self):
        # Get reward history over training episodes
        return self.reward_history
