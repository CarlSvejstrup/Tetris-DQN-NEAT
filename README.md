# Tetris Autonomous Agents: DQN & NEAT

This repository contains the implementation of two distinct artificial intelligence approaches—**Deep Q-Networks (DQN)** and **NeuroEvolution of Augmenting Topologies (NEAT)**—designed to master the game of Tetris. 

Developed as part of the *Introduction to Intelligent Systems* (02461) course at the Technical University of Denmark (DTU), this project explores the comparative efficacy of reinforcement learning versus evolutionary strategies in dynamic environments.

## 🧠 Methodology

The agents do not rely on raw pixel input. Instead, the game state is abstracted into a lower-dimensional feature vector to accelerate convergence.

### State Space Representation
The agents evaluate the board based on four critical heuristics:
1.  **Lines Cleared**: The immediate reward signal for completing rows.
2.  **Bumpiness**: The sum of absolute height differences between adjacent columns (a measure of surface roughness).
3.  **Holes**: The count of empty cells buried under filled cells (indicating structural inefficiency).
4.  **Aggregate Height**: The sum of all column heights.

### Algorithms
* **Deep Q-Network (DQN)**: A reinforcement learning agent that approximates the Q-value function to predict the utility of specific actions (rotations/placements) given the current board state.
* **NEAT (NeuroEvolution of Augmenting Topologies)**: An evolutionary algorithm that starts with simple neural networks and progressively evolves both their weights and topologies (adding nodes/connections) to optimize fitness.

## ⚖️ Reward Shaping

Effective training relies on a robust reward function. This implementation supports multiple reward configurations to balance immediate objectives (clearing lines) with long-term survival (minimizing height and holes).

**Primary Configuration (NES-Style):**
* **Survival**: Points for soft drops (incentivizing faster play).
* **Line Clears**: Exponential scaling based on lines cleared at once (40, 100, 300, 1200 points).
* **Penalty**: -25 penalty for game termination.

## 🛠 Installation

Ensure you have Python 3.8+ installed.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/CarlSvejstrup/Tetris-DQN-NEAT.git](https://github.com/CarlSvejstrup/Tetris-DQN-NEAT.git)
    cd Tetris-DQN-NEAT
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Dependencies include: `torch`, `pygame`, `neat-python`, `numpy`, `matplotlib`, `graphviz`, and `tensorboard`.*

## 🚀 Usage

The project is structured with separate entry points for training the two different model architectures.

### Training the NEAT Agent
To initialize the population and begin the evolutionary process:
```bash
python neat_main.py
