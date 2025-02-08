import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # Replay buffer grande ma non eccessivo: 1M (puoi alzare o abbassare)
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.99   # fattore di sconto
        self.epsilon = 1.0  # esplorazione iniziale
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # Abbassiamo ulteriormente il lr dell'agente
        self.learning_rate = 1e-4

        # Costruzione modelli
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_network()

        # Per il reward scaling adattivo
        self.reward_norm_factor = None
        self.reward_norm_decay = 0.99

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

    def clear_memory(self):
        self.memory.clear()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, dynamic=True):
        if dynamic and np.random.rand() > self.epsilon:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor).numpy()
            return np.argmax(q_values[0])
        else:
            return random.choice(range(self.action_size))

    def replay(self, batch_size, optimizer, scheduler, loss_fn):
        if len(self.memory) < batch_size:
            return

        # Calcolo del reward massimo negli ultimi 1000 campioni
        recent_rewards = [abs(item[2]) for item in list(self.memory)[-1000:]]
        current_max = max(recent_rewards) if recent_rewards else 1e-6

        if self.reward_norm_factor is None:
            self.reward_norm_factor = current_max
        else:
            self.reward_norm_factor = (
                self.reward_norm_decay * self.reward_norm_factor
                + (1 - self.reward_norm_decay) * current_max
            )

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # Normalizza e clampa il reward
            normalized_reward = np.clip(
                reward / (self.reward_norm_factor + 1e-8),
                -5,
                5
            )
            target = normalized_reward
            if not done:
                with torch.no_grad():
                    target += self.gamma * torch.max(self.target_model(next_state_tensor)).item()

            # Aggiornamento
            target_f = self.model(state_tensor).detach().clone()
            target_f[0][action] = target

            optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = loss_fn(output, target_f)
            loss.backward()
            optimizer.step()

        # Scheduler step (riduce lr se gamma < 1)
        scheduler.step()
        # Decay dell'epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)