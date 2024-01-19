This GitHub contains the code for both DQN and NEAT models, implemented in the game of Tetris.

The models simulate all possible positions based on the current board, and a given tetromino (Tetris piece). For each position, a state vector is calculated based on the listed heuristics below:

## Features
1. Cleared Lines
2. Bumpiness (Sum of height difference between each column)
3. Holes (Space with block on top of it)
4. Sum of heights

## Reward
The rewards for the agents can be tweaked to the prefered system:
* soft drops = difference between initial height position to final placement height

### Reward system 1 (NES Tetris)

* 0 lines cleared = number of soft drops
* 1 lines cleared = 40 + number of soft drops
* 2 lines cleared = 100 + number of soft drops
* 3 lines cleared = 300 + number of soft drops
* 4 lines cleared = 1200 + number of soft drops
* temination = -25
  
(difference between initial height position to final placement height)



### Reward system 2
* $(cleared Lines^2) \cdot board Width + number Of Soft Drops$
* termination = -5

### Reward system 2
* $(cleared Lines^3) \cdot Board Width + number Of Soft Drops$
* termination = -5

## Prerequisites
* python 3.8
* pyTorch 2.0
* pygame
* python-neat
* matplotlib
* graphviz
* numpy
* tensorboard

To start the training, simply run neat_main and train_dqn
 
