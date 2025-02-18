import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class QLearningAgent:
    def __init__(self, num_actions):
        self.num_actions = 2  # Numero di azioni possibili
        self.alpha = 0.001  # Tasso di apprendimento
        self.gamma = 0.99  # Fattore di sconto
        self.epsilon = 1.0  # Fattore di esplorazione iniziale
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Adaptive reward scaling
        self.reward_norm_factor = None
        self.reward_norm_decay = 0.99

        # Inizializza la tabella Q a zero
        self.q_table = {}

    def get_q_values(self, state):
        """Restituisce i Q-value per lo stato (inizializza con zeri se non presente)."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        return self.q_table[state]

    def choose_action(self, state):
        """Politica epsilon-greedy: con probabilità epsilon esplora, altrimenti sfrutta."""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.num_actions)
        q_values = self.get_q_values(state)
        return np.argmax(q_values)

    def updateNoNorm(self, state, action, reward, next_state, done):
        """Aggiorna il Q-value per la coppia (stato, azione) usando la formula classica."""
        q_values = self.get_q_values(state)
        q_next = self.get_q_values(next_state)
        max_q_next = 0 if done else np.max(q_next)
        target = reward + self.gamma * max_q_next
        q_values[action] += self.alpha * (target - q_values[action])
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update(self, state, action, reward, next_state, done):
        """Aggiorna il Q-value per la coppia (stato, azione) usando la formula classica,
        con normalizzazione del reward."""

        # Update the reward normalization factor using the absolute value of the current reward
        current_reward_abs = abs(reward)
        if self.reward_norm_factor is None:
            self.reward_norm_factor = current_reward_abs
        else:
            self.reward_norm_factor = (
                    self.reward_norm_decay * self.reward_norm_factor +
                    (1 - self.reward_norm_decay) * current_reward_abs
            )

        # Normalize and clip the reward between -5 and 5
        normalized_reward = np.clip(reward / self.reward_norm_factor, -5, 5)

        # Get current Q-values and next Q-values
        q_values = self.get_q_values(state)
        q_next = self.get_q_values(next_state)

        # Compute target Q-value
        max_q_next = 0 if done else np.max(q_next)
        target = normalized_reward + self.gamma * max_q_next

        # Update the Q-value for the taken action using the learning rate (alpha)
        q_values[action] += self.alpha * (target - q_values[action])

        # Decay the exploration rate if the episode is done
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
