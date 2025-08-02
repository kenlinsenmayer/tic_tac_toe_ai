import numpy as np
import random
from collections import defaultdict

from common import GameMode

class TabularSARSAAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = defaultdict(lambda: np.zeros(9))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state, available_actions, mode):
        eps = self.epsilon if mode == GameMode.TRAINING else 0.0
        if random.random() < eps:
            return random.choice(available_actions)
        q_values = self.Q[state].copy()
        q_values = [q if i in available_actions else -np.inf for i, q in enumerate(q_values)]
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, next_action):
        td_target = reward + self.gamma * self.Q[next_state][next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
