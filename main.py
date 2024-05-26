import numpy as np
import matplotlib.pyplot as plt
import logging
from models import QTable, SARSATable
from environment.maze import MazeGame, RenderOption as Render

logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def QLearning_in_maze(maze, learning_rate =0.9, gamma = 0.99, epsilon = 0.1, n_episodes=100):
    game = MazeGame(maze)
    game.render(Render.TRAINING_MODE)
    model = QTable(game)
    cumulative_reward_history, win_rate_history, _, _  =  model.train(gamma=gamma, epsilon=epsilon, learning_rate=learning_rate, n_eval_episodes=n_episodes)

    try:
        cumulative_reward_history  # check if h is defined
        fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True) # create a figure with four subplots
        fig.canvas.manager.set_window_title(model.name) 
        ax1.plot(*zip(*win_rate_history))
        ax1.set_xlabel("episode")
        ax1.set_ylabel("win rate")
        ax2.plot(cumulative_reward_history)
        ax2.set_xlabel("episode")
        ax2.set_ylabel("cumulative reward")
        
        plt.show()
    except NameError:
        print("h is not defined.")

    game.render(Render.MOVEMENT) # show the moves
    game.play(model) # play the game
    plt.show()  # must be placed here else the image disappears immediately at the end of the program
    
    return cumulative_reward_history, win_rate_history

def SARSA_in_maze(maze, learning_rate = 0.9, gamma = 0.99, epsilon = 0.1, n_episodes=100):
    game = MazeGame(maze)
    results = {}
    game.render(Render.TRAINING_MODE)
    model = SARSATable(game)
    cumulative_reward_history, win_rate_history, _, _ = model.train(gamma=gamma, epsilon=epsilon, learning_rate=learning_rate, n_eval_episodes=n_episodes)
    try:
        cumulative_reward_history  # check if h is defined
        fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True) # create a figure with four subplots
        fig.canvas.manager.set_window_title(model.name) 
        ax1.plot(*zip(*win_rate_history))
        ax1.set_xlabel("episode")
        ax1.set_ylabel("win rate")
        ax2.plot(cumulative_reward_history)
        ax2.set_xlabel("episode")
        ax2.set_ylabel("cumulative reward")
        
        plt.show()
    except NameError:
        print("h is not defined.")

    game.render(Render.MOVEMENT) # show the moves
    game.play(model) # play the game
    plt.show()  # must be placed here else the image disappears immediately at the end of the program
    
    return cumulative_reward_history, win_rate_history

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


# Second maze
maze2 = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

# Third maze
maze3 = np.array([
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0]
])

# List of learning rates to test
learning_rates = [0.1, 0.5, 0.7, 0.9]

# List of gamma values to test
gammas = [0.5, 0.7, 0.9, 0.99]

# List of epsilon values to test
epsilons = [0.01, 0.1, 0.25, 0.5]


# Train and plot results for Q-Learning
Qresults = QLearning_in_maze(maze2)
# plot_results(Qresults, title='Impact of Learning Rate, Gamma, and Epsilon on Performance using Q-Learning')

# Train and plot results for SARSA
SARSAresults = SARSA_in_maze(maze2)
