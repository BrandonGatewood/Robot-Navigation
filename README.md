# A Simple Robot-Navigation
This program uses Q-learning to learn to correctly pick up cans and avoid walls in his grid world.

## Robby the Robot
Robby lives in a 10 x 10 grid, surrounded by a wall. Some of the grid squares contain soda cans.

Robby has five “sensors”:  Current, North, South, East, and West. At any time step, these each return the “value” of the respective location, where the possible values are Empty, Can, and Wall.

Robby has five possible actions:  Move-North, Move-South, Move-East, Move-West, and Pick- Up-Can. If Robby picks up a can, the can is then gone from the grid. Robby receives a reward of 10 for each can he picks up; a “reward” of −5 if he crashes into a wall (after which he immediately bounces back to the square he was in); and a reward of −1 if he tries to pick up a can in an empty square.

### Program:
Program implements a Q-learning method for Robby, using a Q-matrix, in which the rows correspond to states and the columns correspond to actions. The Q-matrix is initialized to all zeros at the beginning of a run. 

During a run, Robby will learn over a series of N episodes, during each of which he will perform M actions. The initial state of the grid in each episode is a random placement of cans, where each grid square has a probability of 0.5 to contain a can (and 0.5 not to contain a can). Robby is initially placed in a random grid square. At each time step t during an episode, the code will do the following:
  • Observe Robby’s current state, <sub>st</sub>
  • Choose an action a<sub>t</sub>, using epsilon-greedy action selection
  • Perform the action
  • Receive reward r<sub>t</sub>(which is zero except in the cases specified above)
  •  Observe Robby’s new state s<sub>t+1</sub>
  •  Update Q(s<sub>t</sub>, a<sub>t</sub>) +  𝜂(r<sub>t</sub> + 𝛾max<sub>a'</sub>Q(s<sub>t + 1</sub>, a') - Q(s<sub>t</sub>, a<sub>t</sub>))

At the end of each episode, it generate a new distribution of cans and places Robby in a random grid square to start the next episode. The Q-matrix will continue updating over the N episodes, keeping track of the total reward gained per episode. 

The following parameter values used to do a run consisting of N episodes of M steps each
  N = 5,000 ; M = 200 ; 𝜂 =  0.2;  𝛾 =  0.9

For choosing actions with epsilon-greedy, epsilon is set to e=0.1 initially, and progressivley decreases every 50 epochs or until it reaches 0. After that, it stays at 0.

After training is completed, it will run N test episodes using the trained Q-matrix, but with e=0.1 for all N episodes.


