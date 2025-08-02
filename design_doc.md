# Project File Overview

## Purpose of Each Code File

- `game.py`: Main experiment driver. Contains the TicTacToe environment, game logic, experiment loop, agent orchestration, evaluation, and plotting of results.
- `agents/tabular_agent.py`: Implements the Tabular SARSA agent for learning to play tic-tac-toe using a table-based approach.
- `agents/nn_agent.py`: Implements the Neural Network SARSA agent for learning to play tic-tac-toe using a neural network function approximator.
- `common.py`: Contains shared types and enums (such as `GameMode`) to avoid circular imports and provide common definitions.
- `plot_results.py`: (If used) Script for plotting experiment results from saved data or logs.
- `requirements.txt`: Lists all Python dependencies required to run the project.
- `README.md`: Project overview, setup instructions, and usage information.
- `design_doc.md`: This design document, describing the project structure, goals, and implementation details.

# Project Title: Tic-Tac-AI - Comparing SARSA Learning in Tabular vs Neural Network Agents

## Objective
Develop a Python project to compare the learning rate and performance of two tic-tac-toe playing AIs using the SARSA reinforcement learning algorithm:
1. A tabular SARSA agent (Q-values in a dictionary)
2. A neural network SARSA agent (Q-values approximated by a neural network using scikit-learn)

## Key Features and Changes
- Introduced a `GameMode` enum to clearly distinguish between training and evaluation phases. All agent `select_action` methods now take a `mode` argument to control exploration (epsilon-greedy for training, greedy for evaluation).
- Evaluation is always performed with greedy action selection (epsilon=0), while training uses epsilon-greedy (epsilon=0.1).
- After training, each agent is evaluated over 1000 games against a random opponent, and the final win rate is displayed.
- Baseline win rates are established for both random vs random and perfect (minimax) vs random play, and these are plotted for comparison.
- The code is modular, with clear separation between game logic, agents, training, and evaluation.
- Circular imports are avoided by placing shared enums and types in `common.py`.

## Project Requirements
- The AI agent always plays first as "x". The opponent ("o") always plays randomly.
- Implement the tic-tac-toe game logic, including win/draw detection and valid move generation.
- Baseline: Simulate 10,000 games between two random agents. Record the win rate for "x" to establish a baseline.
- Implement two SARSA agents:
    - Tabular agent: Q-values stored in a table (Python dict)
    - Neural network agent: Q-values approximated by a neural network (scikit-learn)
- Training:
    - Train each agent for a configurable number of games (default: 20,000)
    - After every 10 training games, evaluate the agent by playing 100 games against a random opponent and record the win rate
    - Store win rate metrics for plotting
- After training, plot win rates over time for both agents, including smoothed curves, and compare to both random and perfect play baselines
- After training, evaluate each agent over 1000 games and display the final win rate

## Implementation Notes
- Use Python 3.8+ and standard libraries where possible
- Use numpy for numerical operations and matplotlib for plotting
- Structure code into modules: game logic, agents, training loop, evaluation, and plotting
- Provide clear function/class docstrings and comments
- Save all results and plots to disk
- Use a `common.py` module for shared types (e.g., `GameMode`)

## Stretch Goals (optional)
- Allow for configurable board size or agent parameters
- Add CLI or simple UI for running experiments

## Deliverables
- Source code for all modules
- requirements.txt listing all dependencies
- Plots comparing win rates of both agents, including baselines
- README.md with instructions to run experiments and interpret results
