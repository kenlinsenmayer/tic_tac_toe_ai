import numpy as np
import matplotlib.pyplot as plt

def plot_results(results_file='results.npz'):
    data = np.load(results_file)
    plt.figure(figsize=(10,6))
    for agent in data.files:
        plt.plot(np.arange(len(data[agent])) * 10, data[agent], label=agent)
    plt.xlabel('Training Games')
    plt.ylabel('Win Rate vs Random')
    plt.title('SARSA Tic-Tac-Toe: Tabular vs Neural Network Agent')
    plt.legend()
    plt.grid(True)
    plt.savefig('win_rates.png')
    plt.show()

if __name__ == '__main__':
    plot_results()
