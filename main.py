# import csv
# import logging
# from enum import Enum, auto

# import matplotlib.pyplot as plt
# import numpy as np

# import models
# from environment.maze import AgentAction, MazeGame, RenderOption as Render


# logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
#                     datefmt="%Y-%m-%d %H:%M:%S",
#                     level=logging.INFO)  # Only show messages *equal to or above* this level


# modelToTrain = models.QTable

# def train_and_play_in_maze(maze, learning_rates, gamma=0.95, epsilon=0.1, n_episodes=2000):
#     game = MazeGame(maze)
    
#     if modelToTrain == models.QTable:
#         # Train using tabular Q-learning
#         game.render(Render.TRAINING_MODE)
#         model = models.QTable(game)
#         h, w, _, _  =  model.train(gamma=gamma, epsilon=epsilon, learning_rate=lr, n_eval_episodes=n_episodes)

#     '''
   
#     if modelToTrain == models.SARSATable:
#         # Train using tabular SARSA
#         game.render(Render.TRAINING_MODE)
#         model = models.SARSATable(game)
#         h, w, _, _ = model.train(gamma=0.95, epsilon=0.10, learning_rate=0.7, n_eval_episodes=200,
#                                  stop_at_convergence=True)
#        '''     

#     # Show the training results
#     try:
#         h  # check if h is defined
#         fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True) # create a figure with four subplots
#         fig.canvas.manager.set_window_title(model.name) 
#         ax1.plot(*zip(*w))
#         ax1.set_xlabel("episode")
#         ax1.set_ylabel("win rate")
#         ax2.plot(h)
#         ax2.set_xlabel("episode")
#         ax2.set_ylabel("cumulative reward")
        
#         plt.show()
#     except NameError:
#         print("h is not defined.")

#     game.render(Render.MOVEMENT) # show the moves
#     game.play(model) # play the game
#     plt.show()  # must be placed here else the image disappears immediately at the end of the program
    
#     return h, w


# # Original maze
# maze1 = np.array([
#     [0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 1, 0, 1, 0, 0],
#     [0, 0, 0, 1, 1, 0, 1, 0],
#     [0, 1, 0, 1, 0, 0, 0, 0],
#     [1, 0, 0, 1, 0, 1, 0, 0],
#     [0, 0, 0, 1, 0, 1, 1, 1],
#     [0, 1, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0]
# ])
# '''
# # Second maze
# maze2 = np.array([
#     [0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 1, 0, 1, 0],
#     [0, 0, 1, 0, 1, 0, 1, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0],
#     [0, 1, 1, 1, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0],
#     [0, 1, 1, 1, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0]
# ])

# # Third maze
# maze3 = np.array([
#     [0, 0, 0, 0, 0, 0, 1, 0],
#     [0, 1, 0, 1, 1, 0, 1, 0],
#     [0, 1, 0, 1, 0, 0, 1, 0],
#     [0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 0, 0, 1, 1, 0],
#     [0, 1, 0, 0, 0, 0, 0, 0]
# ])
# '''

# # Train and play in each maze and show the training results
# h1, w1 = train_and_play_in_maze(maze1)
# '''
# h2, w2 = train_and_play_in_maze(maze2)
# h3, w3 = train_and_play_in_maze(maze3)
# # Plot the training results of each maze
# fig, (ax1, ax2) = plt.subplots(2, 3, figsize=(15, 8), tight_layout=True)

# ax1[0].plot(*zip(*w1))
# ax1[0].set_title("Maze 1")
# ax1[0].set_xlabel("episode")
# ax1[0].set_ylabel("win rate")

# ax1[1].plot(*zip(*w2))
# ax1[1].set_title("Maze 2")
# ax1[1].set_xlabel("episode")
# ax1[1].set_ylabel("win rate")

# ax1[2].plot(*zip(*w3))
# ax1[2].set_title("Maze 3")
# ax1[2].set_xlabel("episode")
# ax1[2].set_ylabel("win rate")

# ax2[0].plot(h1)
# ax2[0].set_xlabel("episode")
# ax2[0].set_ylabel("cumulative reward")

# ax2[1].plot(h2)
# ax2[1].set_xlabel("episode")
# ax2[1].set_ylabel("cumulative reward")

# ax2[2].plot(h3)
# ax2[2].set_xlabel("episode")
# ax2[2].set_ylabel("cumulative reward")

# plt.show()
# '''


import numpy as np
import matplotlib.pyplot as plt
import logging
from models import QTable, SARSATable
from environment.maze import MazeGame, RenderOption as Render

logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

# modelToTrain = QTable
modeleToTrain = SARSATable

# def train_and_play_in_maze(maze, learning_rate = 0.9, gamma = 0.99, epsilon = [0.01, 0.1, 0.25, 0.5], n_episodes=100):
#     game = MazeGame(maze)
#     results = {}
#     for epsilon in epsilons:
#         results[epsilon] = {}
#         game.render(Render.TRAINING_MODE)
#         model = QTable(game)
#         _, win_history, _, _ = model.train(gamma=gamma, epsilon=epsilon, learning_rate= learning_rate, n_eval_episodes=n_episodes)
#         results[epsilon][(learning_rate, gamma, epsilon)] = win_history
#     return results

def train_and_play_in_maze(maze, learning_rate = [0.1, 0.5, 0.7, 0.9], gamma = 0.95, epsilon = 0.1, n_episodes=100):
    game = MazeGame(maze)
    results = {}
    for learning_rate in learning_rates:
        results[learning_rate] = {}
        game.render(Render.TRAINING_MODE)
        model = QTable(game)
        _, win_history, _, _ = model.train(gamma=gamma, epsilon=epsilon, learning_rate= learning_rate, n_eval_episodes=n_episodes)
        results[learning_rate][(learning_rate, gamma, epsilon)] = win_history
    return results

def plot_results(results, title):
    plt.figure(figsize=(12, 8))
    plt.title(title)
    for key, values in results.items():
        for params, scores in values.items():
            learning_rate, gamma, epsilon = params
            episodes = [x[0] for x in scores]
            average_scores = [x[1] for x in scores]
            plt.plot(episodes, average_scores, label=f'Learning Rate = {learning_rate}, Gamma = {gamma}, Epsilon = {epsilon}')
    
    plt.xlabel('Episodes')
    plt.ylabel('Average Score')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def plot_combined_results(results, title):
    plt.figure(figsize=(12, 8))
    plt.title(title)
    for params, scores in results.items():
        learning_rate, gamma, epsilon = params
        episodes = [x[0] for x in scores]
        average_scores = [x[1] for x in scores]
        label = f'Learning Rate = {learning_rate}, Gamma = {gamma}, Epsilon = {epsilon}'
        plt.plot(episodes, average_scores, label=label)
    
    plt.xlabel('Episodes')
    plt.ylabel('Average Score')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# Define your maze configuration
maze = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0]
])

# List of learning rates to test
learning_rates = [0.1, 0.5, 0.7, 0.9]

# List of gamma values to test
gammas = [0.5, 0.7, 0.9, 0.99]

# List of epsilon values to test
epsilons = [0.01, 0.1, 0.25, 0.5]

# Train and plot results for learning rates, gamma values, and epsilon values
results = train_and_play_in_maze(maze)

plot_results(results, title='Impact of Learning Rate, Gamma, and Epsilon on Performance using SARSA Learning')
