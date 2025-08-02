import numpy as np
import random
from sklearn.neural_network import MLPRegressor

from common import GameMode

class NNSARSAAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.model = MLPRegressor(hidden_layer_sizes=(32,), max_iter=1, warm_start=True)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.fitted = False

    def _state_to_input(self, state):
        return np.array(state).reshape(1, -1)

    def select_action(self, state, available_actions, mode):
        eps = self.epsilon if mode == GameMode.TRAINING else 0.0
        if not self.fitted or random.random() < eps:
            return random.choice(available_actions)
        q_values = self.model.predict(self._state_to_input(state))[0]
        q_values = [q if i in available_actions else -np.inf for i, q in enumerate(q_values)]
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, next_action):
        x = self._state_to_input(state)
        if not self.fitted:
            y = np.zeros((1, 9))
            self.model.partial_fit(x, y)
            self.fitted = True
        q_values = self.model.predict(x)[0]
        next_q = self.model.predict(self._state_to_input(next_state))[0][next_action]
        td_target = reward + self.gamma * next_q
        q_values[action] += self.alpha * (td_target - q_values[action])
        self.model.partial_fit(x, [q_values])
