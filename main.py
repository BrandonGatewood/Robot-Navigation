# Brandon Gatewood
# CS 441
# Program 3: Robot Navigation
import random

# This is a simple program that implements an RL agent to clean a 10x10 grid. Robby the Robot use Q-learning to learn to
# correctly pick up cans and avoid walls in his grid world.

import numpy as np

# ROBOT class contains
class ROBOT:
    # Robots initially placed in a random grid square
    x = random.randrange(10)
    y = random.randrange(10)
    reward = 0

    # Robby the robot has 5 sensors: Current, North, South, East and West. At any time step, these each return the
    # "value" of the respective location, where the possible values are Empty, Can and Wall.
    def sensor_current(self, grid):
        return grid[self.x][self.y]

    def sensor_north(self, grid):
        return grid[self.x][self.y + 1]

    def sensor_south(self, grid):
        return grid[self.x][self.y - 1]

    def sensor_east(self, grid):
        return grid[self.x + 1][self.y]

    def sensor_west(self, grid):
        return grid[self.x - 1][self.y]

    # Robby the robot has 5 possible actions: Move-North, Move-South, Move-East, Move-West and Pick-Up-Can.
    # If Robby picks up a can, then the can is gone from the grid.
    def action_pick_up_can(self, grid):
        # Can is present in the grid
        if grid[self.x][self.y] == 1:
            # Remove can from grid
            grid[self.x][self.y] = 0

            return True
        return False

    def action_move_north(self, grid):

        return False

    def action_move_south(self, grid):

        return False
    def action_move_east(self, grid):

        return False
    def action_move_west(self, grid):

        return False

