import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque
import random

class SARSAAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000000)  # Large memory for diverse experiences
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.learning_rate = 1e-2
        self.model = self.build_model()

        # For adaptive reward scaling: initialize with no value and a decay factor
        self.reward_norm_factor = None  # Adaptive scaling factor
        self.reward_norm_decay = 0.99  # Determines how fast the scaling factor adapts

    def build_model(self):
        """Builds the neural network for approximating Q-values."""
        return nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
        )

    def remember(self, state, action, reward, next_state, next_action, done):
        """Stores the experience tuple in memory."""
        self.memory.append((state, action, reward, next_state, next_action, done))

    def act(self, state):
        """Selects an action based on an epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state).detach().numpy()
        return np.argmax(q_values[0])

    def replay(self, batch_size, optimizer, scheduler, loss_fn):
        """Updates the Q-function using the SARSA update rule with reward normalization."""
        if len(self.memory) < batch_size:
            return

        # Compute max_reward from recent experiences
        recent_rewards = [abs(memory[2]) for memory in list(self.memory)[-1000:]]
        current_max = max(recent_rewards) if recent_rewards else 1e-6

        if self.reward_norm_factor is None:
            self.reward_norm_factor = current_max
        else:
            # Exponential moving average update for the normalization factor
            self.reward_norm_factor = (
                    self.reward_norm_decay * self.reward_norm_factor +
                    (1 - self.reward_norm_decay) * current_max)

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, next_action, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # Normalize reward with updated max_reward
            normalized_reward = max(-5, min(5, reward / self.reward_norm_factor))

            target = normalized_reward
            if not done:
                next_q_value = self.model(next_state)[0][next_action].item()
                target += self.gamma * next_q_value

            target_f = self.model(state).detach().clone()
            target_f[0][action] = target

            optimizer.zero_grad()
            output = self.model(state)
            loss = loss_fn(output, target_f)
            loss.backward()
            optimizer.step()

        # Adjust epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)