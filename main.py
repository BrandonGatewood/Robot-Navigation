# Brandon Gatewood
# CS 441
# Program 3: Robot Navigation
import random

# This is a simple program that implements an RL agent to clean a 10x10 grid. Robby the Robot use Q-learning to learn to
# correctly pick up cans and avoid walls in his grid world. Walls will be represented as 99 in the grid world.

import numpy as np
import matplotlib.pyplot as plt


class ROBBY:
    def __init__(self, x=0, y=0, reward=0, cans_collected=0):
        self.x = x
        self.y = y
        self.reward = reward
        self.cans_collected = cans_collected

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
    # If Robby picks up a can, then the can is gone from the grid. If Robby crashes into the wall then Robby will stay
    # in its current square.
    def action_pick_up_can(self, grid):
        # Remove can if can is present
        if grid[self.x][self.y] == 1:
            grid[self.x][self.y] = 0
            return True
        return False

    def action_move_north(self, grid):
        # Move north if there is no wall
        if self.sensor_north(grid) != 99:
            self.y += 1
            return True
        return False

    def action_move_south(self, grid):
        # Move south if there is no wall
        if self.sensor_south(grid) != 99:
            self.y -= 1
            return True
        return False

    def action_move_east(self, grid):
        # Move east if there is no wall
        if self.sensor_east(grid) != 99:
            self.x += 1
            return True
        return False

    def action_move_west(self, grid):
        # Move west if there is no wall
        if self.sensor_west(grid) != 99:
            self.x -= 1
            return True
        return False

    # Robby chooses actions with epsilon-greedy action selection. This will result in robby exploiting the best action
    # to take or exploring random actions.
    @staticmethod
    def select_action(curr_state, Q_matrix, epsilon):
        # Exploit the best action to take
        if np.random.rand() > epsilon:
            possible_actions = []

            # Add all actions q values into list
            for i in range(5):
                actions = Q_matrix[curr_state][i]
                possible_actions.append(actions)

            # Select the max q value of all possible actions
            m = max(possible_actions)
            for i in range(5):
                if m == possible_actions[i]:
                    action = i

                    return action
        # Explore, choose an action uniformly at random
        else:
            action = random.randint(0, 4)

            return action

    # Robby will perform a selected action and receive reward for it. Robby will receive a reward of 10 for each can
    # that is picked up, a reward of -5 for crashing into walls and a reward of -1 if picking up a can on an empty
    # square.
    def perform_action(self, action, grid):
        # Perform pick up can action
        if action == 0:
            result = self.action_pick_up_can(grid)

            # Robby picked up a can
            if result:
                self.cans_collected += 1
                return 10

            # Robby tried picking up a can on an empty square
            return -1

        # Perform move north action
        elif action == 1:
            result = self.action_move_north(grid)

            # Robby moved north
            if result:
                return 0

            # Robby crashed into the wall
            return -5

        # Perform move south action
        elif action == 2:
            result = self.action_move_south(grid)

            # Robby moved south
            if result:
                return 0
            # Robby crashed into the wall
            return -5

        # Perform move east action
        elif action == 3:
            result = self.action_move_east(grid)

            # Robby moved east
            if result:
                return 0
            # Robby crashed into the wall
            return -5

        # Perform move west action
        elif action == 4:
            result = self.action_move_west(grid)

            # Robby moved west
            if result:
                return 0
            # Robby crashed into the wall
            return -5

    # Represent all of Robby's sensors as a tuple
    def tuple_sensors(self, grid):
        sensors = (self.sensor_current(grid), self.sensor_north(grid), self.sensor_south(grid), self.sensor_east(grid),
                   self.sensor_west(grid))

        return sensors

    # At each time step during an episode, Robby's current state is observed, Robby chooses an action using
    # epsilon-greedy action selection, Robby performs that action, Robby receives a reward, Robby's new state is
    # observed, and apply the update rule for Q-Learning.
    def episode(self, grid, Q_matrix, epsilon):
        steps = 200
        learning_rate = 0.2
        discount_factor = 0.9
        curr_step = 0

        while curr_step < steps:
            # Observe current state
            curr_state = self.tuple_sensors(grid)
            if curr_state not in Q_matrix:
                Q_matrix[curr_state] = np.zeros(5)

            # Select action then perform the selected action and save the given reward
            action = self.select_action(curr_state, Q_matrix, epsilon)
            reward = self.perform_action(action, grid)
            self.reward += reward

            # Observe new state
            new_state = self.tuple_sensors(grid)
            if new_state not in Q_matrix:
                Q_matrix[new_state] = np.zeros(5)

            # Update rule for Q-Learning
            Q_matrix[curr_state][action] = Q_matrix[curr_state][action] + learning_rate * (reward + discount_factor * max(Q_matrix[new_state]) - Q_matrix[curr_state][action])
            curr_step += 1

    # Generate a grid world
    @staticmethod
    def generate_grid():
        grid = np.random.choice([0, 1], size=(12, 12), p=[.5, .5])
        for i in range(12):
            for j in range(12):
                if j == 0 or j == 11 or i == 0 or i == 11:
                    grid[i][j] = 99
        return grid

    # Robby will learn over a series of N episodes, during each of which he will perform M actions.  The initial state
    # of the grid in each episode is a random placement of cans, where each grid square has a probability of 0.5 to
    # contain a can (and 0.5 not to contain a can). Robby is initially placed in a random grid square.
    def train_robby(self, Q_matrix):
        episodes = 5000
        current_episode = 0
        epsilon = 0.1
        rewards = []
        while current_episode < episodes:
            # At the end of each episode, generate a new distribution of cans and place Robby in a random grid
            # square to start the next episode
            grid = self.generate_grid()

            self.x = random.randint(1, 10)
            self.y = random.randint(1, 10)
            self.reward = 0
            self.cans_collected = 0

            # Run an episode
            self.episode(grid, Q_matrix, epsilon)

            # Save total reward for the episode
            rewards.append(self.reward)

            current_episode += 1

            # Progressively decrease epsilon every 50 epochs until it reaches 0
            if epsilon > 0:
                if current_episode % 50 == 0:
                    epsilon -= 0.01

                    if epsilon < 0:
                        epsilon = 0

        # Plot the total sum of rewards per episode (plotting a point every 100 episodes). This indicates the extent to
        # which Robby is learning to improve the cumulative reward.
        print("Train-Average: ")
        print(sum(rewards) / episodes)
        print("Train-Standard-Deviation: ")
        print(np.std(rewards))
        plt.title("Training Reward Plot")
        plt.xlabel("Number of Episodes")
        plt.ylabel("Sum of Rewards")
        plt.plot(rewards, marker='o', mfc='red', markevery=100)
        plt.show()

    # Test Robby is the same as training Robby, however epsilon will continue to stay at 0.1.
    def test_robby(self, Q_matrix):
        episodes = 5000
        current_episode = 0
        epsilon = 0.1
        rewards = []
        while current_episode < episodes:
            grid = self.generate_grid()

            self.x = random.randint(1, 10)
            self.y = random.randint(1, 10)
            self.reward = 0
            self.cans_collected = 0

            # Run an episode
            self.episode(grid, Q_matrix, epsilon)

            # Save total reward for the episode
            rewards.append(self.reward)

            current_episode += 1

        # Calculate the average over sum-of-rewards-per-episode, and the standard deviation
        print("Test-Average: ")
        print(sum(rewards) / episodes)
        print("Test-Standard-Deviation: ")
        print(np.std(rewards))


Q_matrix = {}
robby = ROBBY()

# Train Robby
robby.train_robby(Q_matrix)
# Test Robby with the trained Q_matrix
robby.test_robby(Q_matrix)