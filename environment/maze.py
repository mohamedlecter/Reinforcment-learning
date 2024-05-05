import logging
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, IntEnum

class MazeCell(IntEnum): # Enumerate the possible cell types
    EMPTY = 0 
    WALL = 1  
    AGENT = 2  
class AgentAction(IntEnum):  # Enumerate the possible actions
    MOVE_LEFT = 0 
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3
class RenderOption(Enum):  # Enumerate the possible rendering options
    NONE = 0 
    TRAINING_MODE = 1
    MOVEMENT = 2
    
class GameStatus(Enum):  # Enumerate the possible game statuses
    WIN = 0
    LOSE = 1
    PLAYING = 2
    
class MazeGame:
    available_actions = [AgentAction.MOVE_LEFT, AgentAction.MOVE_RIGHT, AgentAction.MOVE_UP, AgentAction.MOVE_DOWN]  # all possible actions
    exit_reward = 7.0  # exit reward
    move_penalty = -0.10  # penalty for each move
    visited_penalty = -0.30 
    wrong_move_penalty = -0.70  # penalty for trying to move into a wall or out of the maze
    
    def __init__(self, maze, starting_cell=(0, 0), exit_cell=None):
        self.maze = maze
        self.__min_reward_threshold = -0.5 * self.maze.size  # stop game if accumulated reward is below this threshold (-0.5 * the size of the maze), i.e. too much loss
        nrows, ncols = self.maze.shape
        self.cells = [(col, row) for col in range(ncols) for row in range(nrows)]
        self.empty_cells = [(col, row) for col in range(ncols) for row in range(nrows) if self.maze[row, col] == MazeCell.EMPTY]
        self.__exit_cell = (ncols - 1, nrows - 1) if exit_cell is None else exit_cell
        self.empty_cells.remove(self.__exit_cell)

        # Check for impossible maze layout
        if self.__exit_cell not in self.cells:
            raise Exception("Error: exit cell at {} is not inside maze".format(self.__exit_cell))
        if self.maze[self.__exit_cell[::-1]] == MazeCell.WALL:
            raise Exception("Error: exit cell at {} is not free".format(self.__exit_cell))

        # Variables for rendering using Matplotlib
        self.__render_option = RenderOption.NONE  # what to render
        self.__movement_ax = None  # axes for rendering the moves
        self.__action_ax = None  # axes for rendering the best action per cell

        self.reset(starting_cell)
        
    def reset(self, starting_cell=(0, 0)):
        """
        Reset the maze game to the initial state.
        """
        if starting_cell not in self.cells:
            raise Exception("Error: start cell at {} is not inside maze".format(starting_cell))
        if self.maze[starting_cell[::-1]] == MazeCell.WALL:
            raise Exception("Error: start cell at {} is not free".format(starting_cell))
        if starting_cell == self.__exit_cell:
            raise Exception("Error: start- and exit cell cannot be the same {}".format(starting_cell))

        self.__previous_cell = self.__current_cell = starting_cell
        self.__total_reward = 0.0  # accumulated reward
        self.__visited_cells = set()  # a set() only stores unique values

        if self.__render_option in (RenderOption.TRAINING_MODE, RenderOption.MOVEMENT):
            # render the maze
            nrows, ncols = self.maze.shape
            self.__movement_ax.clear()
            self.__movement_ax.set_xticks(np.arange(0.5, nrows, step=1))
            self.__movement_ax.set_xticklabels([])
            self.__movement_ax.set_yticks(np.arange(0.5, ncols, step=1))
            self.__movement_ax.set_yticklabels([])
            self.__movement_ax.grid(True)
            self.__movement_ax.plot(*self.__current_cell, "rs", markersize=30)  # start is a big red square
            self.__movement_ax.text(*self.__current_cell, "Start", ha="center", va="center", color="white")
            self.__movement_ax.plot(*self.__exit_cell, "gs", markersize=30)  # exit is a big green square
            self.__movement_ax.text(*self.__exit_cell, "Exit", ha="center", va="center", color="white")
            self.__movement_ax.imshow(self.maze, cmap="binary")
            self.__movement_ax.get_figure().canvas.draw()
            self.__movement_ax.get_figure().canvas.flush_events()

        return self.__observe()
    
    def __draw_movement(self):
        """
        Draw a line from the agent's previous cell to its current cell.
        """
        self.__movement_ax.plot(*zip(*[self.__previous_cell, self.__current_cell]), "bo-")  # previous cells are blue dots
        self.__movement_ax.plot(*self.__current_cell, "ro")  # current cell is a red dot
        self.__movement_ax.get_figure().canvas.draw()
        self.__movement_ax.get_figure().canvas.flush_events()
        
        
    def render(self, content=RenderOption.NONE):
        """
        Render the maze based on the specified content.
        """
        self.__render_option = content

        if self.__render_option == RenderOption.NONE:
            if self.__movement_ax:
                self.__movement_ax.get_figure().close()
                self.__movement_ax = None
            if self.__action_ax:
                self.__action_ax.get_figure().close()
                self.__action_ax = None
        if self.__render_option == RenderOption.TRAINING_MODE:
            if self.__action_ax is None:
                fig, self.__action_ax = plt.subplots(1, 1, tight_layout=True)
                self.__action_ax.set_axis_off()
                self.render_q_values(None)
        if self.__render_option in (RenderOption.MOVEMENT, RenderOption.TRAINING_MODE):
            if self.__movement_ax is None:
                fig, self.__movement_ax = plt.subplots(1, 1, tight_layout=True)

        plt.show(block=False)
        
        
    def step(self, action):
        """
        Take a step in the maze game based on the specified action.
        """
        reward = self.__execute_action(action)
        self.__total_reward += reward
        status = self.__get_game_status()
        state = self.__observe()
        logging.debug("action: {:10s} | reward: {: .2f} | status: {}".format(AgentAction(action).name, reward, status))
        return state, reward, status
    
    def __execute_action(self, action):
        """
        Execute the specified action in the maze game.
        """
        possible_actions = self.__possible_actions(self.__current_cell)

        if not possible_actions:
            reward = self.__min_reward_threshold - 1  # cannot move anywhere, force end of game
        elif action in possible_actions:
            col, row = self.__current_cell
            if action == AgentAction.MOVE_LEFT:
                col -= 1
            elif action == AgentAction.MOVE_UP:
                row -= 1
            if action == AgentAction.MOVE_RIGHT:
                col += 1
            elif action == AgentAction.MOVE_DOWN:
                row += 1

            self.__previous_cell = self.__current_cell
            self.__current_cell = (col, row)

            if self.__render_option != RenderOption.NONE:
                self.__draw_movement()

            if self.__current_cell == self.__exit_cell:
                reward = MazeGame.exit_reward  # maximum reward when reaching the exit cell
            elif self.__current_cell in self.__visited_cells:
                reward = MazeGame.visited_penalty  # penalty when returning to a cell which was visited earlier
            else:
                reward = MazeGame.move_penalty  # penalty for a move which did not result in finding the exit cell

            self.__visited_cells.add(self.__current_cell)
        else:
            reward = MazeGame.wrong_move_penalty  # penalty for trying to enter an occupied cell or move out of the maze

        return reward
    
    def __possible_actions(self, cell=None):
        """
        Determine the possible actions from the specified cell.
        """
        if cell is None:
            col, row = self.__current_cell
        else:
            col, row = cell

        possible_actions = MazeGame.available_actions.copy()  # initially allow all

        # now restrict the initial list by removing impossible actions
        nrows, ncols = self.maze.shape
        if row == 0 or (row > 0 and self.maze[row - 1, col] == MazeCell.WALL):
            possible_actions.remove(AgentAction.MOVE_UP)
        if row == nrows - 1 or (row < nrows - 1 and self.maze[row + 1, col] == MazeCell.WALL):
            possible_actions.remove(AgentAction.MOVE_DOWN)

        if col == 0 or (col > 0 and self.maze[row, col - 1] == MazeCell.WALL):
            possible_actions.remove(AgentAction.MOVE_LEFT)
        if col == ncols - 1 or (col < ncols - 1 and self.maze[row, col + 1] == MazeCell.WALL):
            possible_actions.remove(AgentAction.MOVE_RIGHT)

        return possible_actions
    
    def __get_game_status(self):
        """
        Get the current status of the game.
        """
        if self.__current_cell == self.__exit_cell:
            return GameStatus.WIN

        if self.__total_reward < self.__min_reward_threshold:  # force end of game after too much loss
            return GameStatus.LOSE

        return GameStatus.PLAYING
    
    def __observe(self):
        """
        Observe the current state of the maze game.
        """
        return np.array([[*self.__current_cell]])
    
    def play(self, model, starting_cell=(0, 0)):
        """
        Play the maze game with the specified model starting from the specified cell.
        """
        self.reset(starting_cell)

        state = self.__observe()

        while True:
            action = model.predict_value(state=state)
            state, reward, status = self.step(action)
            if status in (GameStatus.WIN, GameStatus.LOSE):
                return status
            
    def check_win_all(self, model):
        """
        Check if the model can win the maze game from all empty cells.
        """
        previous_render = self.__render_option
        self.__render_option = RenderOption.NONE  # avoid rendering anything during execution of the check games

        win_count = 0
        lose_count = 0

        for cell in self.empty_cells:
            if self.play(model, cell) == GameStatus.WIN:
                win_count += 1
            else:
                lose_count += 1

        self.__render_option = previous_render  # restore previous rendering setting

        logging.info("won: {} | lost: {} | win rate: {:.5f}".format(win_count, lose_count, win_count / (win_count + lose_count)))

        result = True if lose_count == 0 else False

        return result, win_count / (win_count + lose_count)
    
    def render_q_values(self, model):
        """
        Render the Q-values of the model in the maze game.
        """
        def clip(n):
            return max(min(1, n), 0)

        if self.__render_option == RenderOption.TRAINING_MODE:
            nrows, ncols = self.maze.shape

            self.__action_ax.clear()
            self.__action_ax.set_xticks(np.arange(0.5, nrows, step=1))
            self.__action_ax.set_xticklabels([])
            self.__action_ax.set_yticks(np.arange(0.5, ncols, step=1))
            self.__action_ax.set_yticklabels([])
            self.__action_ax.grid(True)
            self.__action_ax.plot(*self.__exit_cell, "gs", markersize=30) 
            self.__action_ax.text(*self.__exit_cell, "Exit", ha="center", va="center", color="white")

            for cell in self.empty_cells:
                q_values = model.q_values(cell) if model is not None else [0, 0, 0, 0]
                best_actions = np.nonzero(q_values == np.max(q_values))[0]

                for action in best_actions:
                    dx = 0
                    dy = 0
                    if action == AgentAction.MOVE_LEFT:
                        dx = -0.2
                    if action == AgentAction.MOVE_RIGHT:
                        dx = +0.2
                    if action == AgentAction.MOVE_UP:
                        dy = -0.2
                    if action == AgentAction.MOVE_DOWN:
                        dy = 0.2

                    # color (from red to green) represents the certainty of the preferred action(s)
                    max_value = 1
                    min_value = -1
                    color = clip((q_values[action] - min_value) / (max_value - min_value))  # normalize in [-1, 1]

                    self.__action_ax.arrow(*cell, dx, dy, color=(1 - color, color, 0), head_width=0.2, head_length=0.1)

            self.__action_ax.imshow(self.maze, cmap="binary")
            self.__action_ax.get_figure().canvas.draw()
