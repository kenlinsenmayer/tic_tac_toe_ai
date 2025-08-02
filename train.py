import numpy as np
import random
from agents.tabular_agent import TabularSARSAAgent
from agents.nn_agent import NNSARSAAgent
from game import TicTacToe

def play_game(env, agent_x, agent_o, train=False):
    state = env.reset()
    done = False
    action = agent_x.select_action(state, env.available_actions())
    while not done:
        next_state, reward, done, _ = env.step(action)
        if done:
            if train:
                agent_x.update(state, action, reward, next_state, 0)
            break
        next_action = agent_o.select_action(next_state, env.available_actions())
        if train:
            agent_x.update(state, action, reward, next_state, next_action)
        state, action = next_state, next_action
        agent_x, agent_o = agent_o, agent_x  # Switch players
    return env.winner

def evaluate_agent(agent, env, n_games=100):
    wins = 0
    for _ in range(n_games):
        winner = play_game(env, agent, RandomAgent(), train=False)
        if winner == 1:
            wins += 1
    return wins / n_games

class RandomAgent:
    def select_action(self, state, available_actions):
        return random.choice(available_actions)
    def update(self, *args, **kwargs):
        pass

def main():
    n_train_games = 10000
    eval_interval = 10
    eval_games = 100
    env = TicTacToe()
    agents = {
        'tabular': TabularSARSAAgent(),
        'nn': NNSARSAAgent()
    }
    results = {k: [] for k in agents}
    for name, agent in agents.items():
        for i in range(1, n_train_games + 1):
            play_game(env, agent, RandomAgent(), train=True)
            if i % eval_interval == 0:
                win_rate = evaluate_agent(agent, env, n_games=eval_games)
                results[name].append(win_rate)
    np.savez('results.npz', **results)

if __name__ == '__main__':
    main()
