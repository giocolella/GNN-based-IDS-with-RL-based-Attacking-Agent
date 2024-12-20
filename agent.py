import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000) #Increasing Memory Size Allows the agent to retain more diverse experiences (less overfitting) for training but Can increase computational cost and memory usage
        self.gamma = 0.95 #Determines how much the agent values future rewards compared to immediate rewards. Values closer to 1.0 prioritize long-term rewards, while values closer to 0.0 prioritize immediate rewards
        self.epsilon = 0.999 #previously 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995 #previously 0.995
        self.learning_rate = 1e-2
        self.model = self.build_model()

    def build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
        )

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state).detach().numpy()
        return np.argmax(q_values[0])

    def replay(self, batch_size, optimizer, loss_fn, max_reward=1.0):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            max_reward = max(max_reward, abs(reward))

            # Normalize reward
            normalized_reward = reward / max_reward

            # Calculate target
            target = normalized_reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()

            # Prepare target values for the specific action
            target_f = self.model(state).detach().clone()
            target_f[0][action] = target

            # Debug target computation
            #print("Target computation debug:")
            #print("Reward:", reward)
            #print("Normalized Reward:", normalized_reward)
            #print("Next Q-values:", self.model(next_state).detach().numpy())
            #print("Max next Q-value:", torch.max(self.model(next_state)).item())
            #print("Target:", target)

            # Debug Q-values before update
            #print("Q-values before update:", self.model(state).detach().numpy())

            # Perform optimization step
            optimizer.zero_grad()
            output = self.model(state)  # Raw outputs (logits)
            loss = loss_fn(output, target_f)
            loss.backward()

            # Debug gradients


            optimizer.step()

            # Debug Q-values after update
            #print("Q-values after update:", self.model(state).detach().numpy())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
