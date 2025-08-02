# --- Imports ---
import random
import numpy as np
import matplotlib.pyplot as plt
from agents.tabular_agent import TabularSARSAAgent
from agents.nn_agent import NNSARSAAgent
from common import GameMode
from functools import lru_cache

# --- Agent Classes ---
class RandomAgent:
    def select_action(self, state, available_actions, mode):
        return random.choice(available_actions)
    def update(self, *args, **kwargs):
        pass

@lru_cache(maxsize=100000)
def minimax(state, player):
    board = np.array(state).reshape(3, 3)
    for p in [1, -1]:
        if any(np.all(board[i, :] == p) for i in range(3)) or \
           any(np.all(board[:, j] == p) for j in range(3)) or \
           np.all(np.diag(board) == p) or \
           np.all(np.diag(np.fliplr(board)) == p):
            return p
    if not any(v == 0 for v in state):
        return 0  # Draw
    scores = []
    for i, v in enumerate(state):
        if v == 0:
            next_state = list(state)
            next_state[i] = player
            score = minimax(tuple(next_state), -player)
            scores.append(score)
    if player == 1:
        return max(scores)
    else:
        return min(scores)

class MinimaxAgent:
    def select_action(self, state, available_actions, mode):
        best_score = -float('inf')
        best_action = None
        for action in available_actions:
            next_state = list(state)
            next_state[action] = 1
            score = minimax(tuple(next_state), -1)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action
    def update(self, *args, **kwargs):
        pass

# --- Game Logic ---
class TicTacToe:
    def __init__(self):
        self.reset()
    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.get_state()
    def get_state(self) -> tuple:
        return tuple(self.board.flatten())
    def available_actions(self) -> list:
        return [i for i, v in enumerate(self.board.flatten()) if v == 0]
    def step(self, action: int) -> tuple:
        if self.done or self.board.flatten()[action] != 0:
            raise ValueError("Invalid move")
        row, col = divmod(action, 3)
        self.board[row, col] = self.current_player
        reward, self.done, self.winner = self._check_game_status()
        self.current_player *= -1
        return self.get_state(), reward, self.done, {}
    def _check_game_status(self):
        for player in [1, -1]:
            if any(np.all(self.board[i, :] == player) for i in range(3)) or \
               any(np.all(self.board[:, j] == player) for j in range(3)) or \
               np.all(np.diag(self.board) == player) or \
               np.all(np.diag(np.fliplr(self.board)) == player):
                return (1 if player == 1 else -1), True, player
        if not self.available_actions():
            return 0, True, 0
        return 0, False, None
    def render(self):
        symbols = {1: 'X', -1: 'O', 0: '.'}
        for row in self.board:
            print(' '.join(symbols[x] for x in row))
        print()

def play_game(env, agent_x, agent_o, mode=GameMode.TRAINING):
    state = env.reset()
    done = False
    action = agent_x.select_action(state, env.available_actions(), mode)
    while not done:
        next_state, reward, done, _ = env.step(action)
        if done:
            if mode == GameMode.TRAINING:
                agent_x.update(state, action, reward, next_state, 0)
            break
        next_action = agent_o.select_action(next_state, env.available_actions(), mode)
        if mode == GameMode.TRAINING:
            agent_x.update(state, action, reward, next_state, next_action)
        state, action = next_state, next_action
        agent_x, agent_o = agent_o, agent_x
    return env.winner

def evaluate_agent(agent, env, n_games=100):
    wins = 0
    for _ in range(n_games):
        winner = play_game(env, agent, RandomAgent(), mode=GameMode.EVALUATION)
        if winner == 1:
            wins += 1
    return wins / n_games

# --- Experiment/Runner ---
if __name__ == "__main__":
    # Baseline: random vs random
    print("Establishing baseline win rate for X (random vs random)...")
    env = TicTacToe()
    x_wins = 0
    o_wins = 0
    draws = 0
    for _ in range(10000):
        winner = play_game(env, RandomAgent(), RandomAgent(), mode=GameMode.EVALUATION)
        if winner == 1:
            x_wins += 1
        elif winner == -1:
            o_wins += 1
        else:
            draws += 1
    print(f"Random vs Random: X wins: {x_wins}, O wins: {o_wins}, Draws: {draws}")
    print(f"Baseline X win rate: {x_wins/10000:.3f}")

    # Baseline: perfect vs random
    print("\nCalculating max expected win rate for a perfect player (minimax vs random)...")
    perfect_x_wins = 0
    perfect_o_wins = 0
    perfect_draws = 0
    for _ in range(1000):
        winner = play_game(env, MinimaxAgent(), RandomAgent(), mode=GameMode.EVALUATION)
        if winner == 1:
            perfect_x_wins += 1
        elif winner == -1:
            perfect_o_wins += 1
        else:
            perfect_draws += 1
    perfect_x_win_rate = perfect_x_wins / 1000
    print(f"Perfect X vs Random O: X wins: {perfect_x_wins}, O wins: {perfect_o_wins}, Draws: {perfect_draws}")
    print(f"Perfect X win rate: {perfect_x_win_rate:.3f}")

    # Train both agents
    n_train_games = 50000
    eval_interval = 50
    eval_games = 200
    agents = {
        'tabular': TabularSARSAAgent(),
        'nn': NNSARSAAgent()
    }
    results = {k: [] for k in agents}
    print("\nTraining agents...")
    for name, agent in agents.items():
        print(f"Training {name} agent...")
        for i in range(1, n_train_games + 1):
            play_game(env, agent, RandomAgent(), mode=GameMode.TRAINING)
            if i % eval_interval == 0:
                win_rate = evaluate_agent(agent, env, n_games=eval_games)
                results[name].append(win_rate)
        print(f"{name} agent final win rate (moving average): {results[name][-1]:.3f}")
        final_eval = evaluate_agent(agent, env, n_games=1000)
        print(f"{name} agent final win rate (1000 games): {final_eval:.3f}")

    # Smoothing function
    def moving_average(data, window_size=50):
        if len(data) < window_size:
            return np.array(data)
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    # Plot results with smoothing and baselines
    plt.figure(figsize=(10,6))
    x_vals = np.arange(len(results['tabular'])) * eval_interval
    for agent in results:
        smoothed = moving_average(results[agent])
        x_smoothed = x_vals[:len(smoothed)]
        plt.plot(x_smoothed, smoothed, label=f"{agent} (smoothed)")
    plt.axhline(y=x_wins/10000, color='k', linestyle='--', label='Baseline (Random X win rate)')
    plt.axhline(y=perfect_x_win_rate, color='g', linestyle=':', label='Perfect X win rate')
    plt.xlabel('Training Games')
    plt.ylabel('Win Rate vs Random')
    plt.title('SARSA Tic-Tac-Toe: Tabular vs Neural Network Agent')
    plt.legend()
    plt.grid(True)
    plt.show()
