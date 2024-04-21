import logging
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np

import models
from environment.maze import MazeGame, RenderOption as Render


logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)  # Only show messages *equal to or above* this level


class Test(Enum):
    SHOW_MAZE_ONLY = auto()
    Q_LEARNING = auto()
    


test = Test.Q_LEARNING  # which test to run

# 0 = free, 1 = occupied
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

maze3 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 1, 0, 1, 0, 1, 0],
    [1, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 1, 0],
    [1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])



game = MazeGame(maze)

# only show the maze
if test == Test.SHOW_MAZE_ONLY:
    game.render(Render.MOVEMENT)
    game.reset()

# train using tabular Q-learning
if test == Test.Q_LEARNING:
    game.render(Render.TRAINING_MODE)
    model = models.QTable(game)
    h, w, _, _ = model.train(gamma=0.95, epsilon=0.10, learning_rate=0.7, n_eval_episodes=200,
                             stop_at_convergence=True)


# show the training results
try:
    h  # check if h is defined
    fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True) # create a figure with two subplots
    fig.canvas.manager.set_window_title(model.name) 
    ax1.plot(*zip(*w))
    ax1.set_xlabel("episode")
    ax1.set_ylabel("win rate")
    ax2.plot(h)
    ax2.set_xlabel("episode")
    ax2.set_ylabel("cumulative reward")
    plt.show()
except NameError:
    pass


game.render(Render.MOVES) # show the moves
game.play(model, start_cell=(4, 1)) # play the game starting at cell (4, 1)

plt.show()  # must be placed here else the image disappears immediately at the end of the program
