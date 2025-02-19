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
        self.memory = deque(maxlen=1000000)  # Large memory for diverse experiences
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.learning_rate = 1e-2
        self.model = self.build_model()
        self.target_model = self.build_model()

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

    def update_target_network(self):
        """Updates the target network by copying weights from the current model."""
        self.target_model.load_state_dict(self.model.state_dict())

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
        """Updates the Q-function using the agent update rule with reward normalization."""
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

        # Sample a minibatch of transitions (including next_action for deep SARSA)
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, next_actions, dones = zip(*minibatch)

        # Convert lists to tensors
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        next_actions_tensor = torch.tensor(next_actions, dtype=torch.int64).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Normalize and clip rewards between -5 and 5
        normalized_rewards = torch.clamp(rewards_tensor / self.reward_norm_factor, -5, 5)

        # Compute Q-values for the current states and select the ones for the taken actions
        current_q_values = self.model(states_tensor)
        predicted_q = current_q_values.gather(1, actions_tensor)

        # In deep SARSA, use the next action from memory (on-policy) for the update:
        with torch.no_grad():
            next_q_values = self.target_model(next_states_tensor).detach().gather(1, next_actions_tensor)

        # Compute target Q-values: if done, no next Q-value is added
        target_q = normalized_rewards + (1 - dones_tensor) * self.gamma * next_q_values

        # Compute loss between the predicted Q-value for the taken action and the target
        loss = loss_fn(predicted_q, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Decay the exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)