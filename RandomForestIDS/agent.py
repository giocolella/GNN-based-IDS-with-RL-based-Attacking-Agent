import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        # Changed from 2 to 4 actions:
        # 0 = benign; 1 = low-intensity malicious; 2 = medium-intensity malicious; 3 = high-intensity malicious
        self.action_size = action_size
        self.memory = deque(maxlen=5000000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-2
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_network()

    def build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
        )

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, dynamic=True):
        if dynamic and np.random.rand() > self.epsilon:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state_tensor).detach().numpy()[0]
            return np.argmax(q_values)
        return random.randrange(self.action_size)

    def replay(self, batch_size, optimizer, scheduler, loss_fn):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(next_state_tensor)).item()

            target_f = self.model(state_tensor).detach().clone()
            target_f[0][action] = target

            optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = loss_fn(output, target_f)
            loss.backward()
            optimizer.step()

        scheduler.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
