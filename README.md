# Tetris-DQN-NEAT 

This repository contains the code for both Deep Q-Learning (DQN) and NeuroEvolution of Augmenting Topologies (NEAT) models, implemented in the game of Tetris.

## Overview

This codebase is part of the projects under the course **02461: Introduction to Intelligent Systems** at the Technical University of Denmark (DTU). The project demonstrates the application of reinforcement learning and evolutionary algorithms to optimize Tetris gameplay.

The models simulate all possible positions for the current game board and a given tetromino (Tetris piece). For each position, a state vector is computed based on the heuristics described below.

## Heuristics

The state vector for each possible position is determined based on these features:
1. Cleared Lines: The number of lines cleared by placing the tetromino.
2. Bumpiness: The sum of height differences between adjacent columns.
3. Holes: Empty spaces with blocks above them on the board.
4. Total Height: The sum of the column heights on the board.

## Reward Systems

The rewards for the agents can be adjusted to fit the task. Below are the implemented reward systems:

### Reward System 1 (NES Tetris-inspired)
Rewards:
* Soft Drops: Number of rows descended by the tetromino (initial height minus final height).
* Scoring:
  * 0 lines cleared = Number of soft drops.
  * 1 line cleared = 40 + Number of soft drops.
  * 2 lines cleared = 100 + Number of soft drops.
  * 3 lines cleared = 300 + Number of soft drops.
  * 4 lines cleared = 1200 + Number of soft drops.
  * Game termination = -25.

### Reward System 2
Reward formulation:
* \( (Cleared \; Lines)^2 \times Board \; Width + Number \; of \; Soft \; Drops \)
* Game termination = -5.

### Reward System 3
Reward formulation:
* \( (Cleared \; Lines)^3 \times Board \; Width + Number \; of \; Soft \; Drops \)
* Game termination = -5.

## Prerequisites
Before running the code, ensure that the following dependencies are installed:
- Python 3.8
- PyTorch 2.0
- Pygame
- Python-NEAT
- Matplotlib
- Graphviz
- NumPy
- TensorBoard

You can install the required libraries using `pip`:
```bash
pip install pygame neat-python matplotlib graphviz numpy torch tensorboard
```

## Usage

To start the training process for either model, execute the following scripts:

- For NEAT, run:
  ```bash
  python neat_main.py
  ```

- For DQN, run:
  ```bash
  python train_dqn.py
  ```

For further instructions and code explanations, refer to the corresponding files.

---

This project exemplifies the use of reinforcement learning (DQN) and neuro-evolution (NEAT) to tackle decision-making and optimization problems within the realm of intelligent systems.
