import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000000) #Increasing Memory Size Allows the agent to retain more diverse experiences (less overfitting) for training but Can increase computational cost and memory usage
        self.gamma = 0.99 #Determines how much the agent values future rewards compared to immediate rewards. Values closer to 1.0 prioritize long-term rewards, while values closer to 0.0 prioritize immediate rewards
        self.epsilon = 1.0 #previously 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995 #previously 0.995
        self.learning_rate = 1e-2
        self.model = self.build_model()
        self.target_model = self.build_model()  # Target network
        self.update_target_network()  # Initialize target network

        # For adaptive reward scaling: initialize with no value and a decay factor
        self.reward_norm_factor = None  # Adaptive scaling factor
        self.reward_norm_decay = 0.99  # Determines how fast the scaling factor adapts

    def build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
        )

    def update_target_network(self):
        # Copy weights from the main model to the target model
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, dynamic=True):
        if dynamic and np.random.rand() > self.epsilon:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state).detach().numpy()
            return np.argmax(q_values[0])
        return random.choice([0, 1])  # Random benign/malicious action

    def replayoldMaxReward(self, batch_size, optimizer, loss_fn, max_reward=1.0):
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

    def replayNoScheduler(self, batch_size, optimizer, loss_fn):
        if len(self.memory) < batch_size:
            return

        # Compute max_reward from recent experiences
        recent_rewards = [abs(memory[2]) for memory in list(self.memory)[-1000:]]
        max_reward = max(max(recent_rewards) if recent_rewards else 0, 1e-6)

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # Normalize reward with updated max_reward
            normalized_reward = max(-5, min(5, reward / max_reward))
            # print(f"Normalized Reward: {normalized_reward}")

            # Calculate target using the target network
            target = normalized_reward
            if not done:
                target += self.gamma * torch.max(self.target_model(next_state)).item()

            # Prepare target values for the specific action
            target_f = self.model(state).detach().clone()
            target_f[0][action] = target

            # Perform optimization step
            optimizer.zero_grad()
            output = self.model(state)
            loss = loss_fn(output, target_f)
            loss.backward()
            optimizer.step()
            # print(f"Q-values for state: {self.model(state).detach().numpy()}")

        # Adjust epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def replaySkewed(self, batch_size, optimizer, scheduler, loss_fn):
        if len(self.memory) < batch_size:
            return

        # Compute max_reward from recent experiences
        recent_rewards = [abs(memory[2]) for memory in list(self.memory)[-1000:]]
        max_reward = max(max(recent_rewards) if recent_rewards else 0, 1e-6)

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # Normalize reward with updated max_reward
            normalized_reward = max(-5, min(5, reward / max_reward))
            # print(f"Normalized Reward: {normalized_reward}")

            # Calculate target using the target network
            target = normalized_reward
            if not done:
                target += self.gamma * torch.max(self.target_model(next_state)).item()

            # Prepare target values for the specific action
            target_f = self.model(state).detach().clone()
            target_f[0][action] = target

            # Perform optimization step
            optimizer.zero_grad()
            output = self.model(state)
            loss = loss_fn(output, target_f)
            loss.backward()
            optimizer.step()
            # print(f"Q-values for state: {self.model(state).detach().numpy()}")

        # Decay learning rate
        scheduler.step()

        # Adjust epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def replay(self, batch_size, optimizer, scheduler, loss_fn):
        if len(self.memory) < batch_size:
            return

        #Adaptive Reward Scaling using a sliding window
        # Use only the last 1000 experiences for computing the scaling factor.
        recent_rewards = [abs(memory[2]) for memory in list(self.memory)[-1000:]]
        current_max = max(recent_rewards) if recent_rewards else 1e-6

        if self.reward_norm_factor is None:
            self.reward_norm_factor = current_max
        else:
            # Exponential moving average update for the normalization factor
            self.reward_norm_factor = (
                    self.reward_norm_decay * self.reward_norm_factor +
                    (1 - self.reward_norm_decay) * current_max
            )

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # Normalize reward using the adaptive scaling factor and clip it between -5 and 5
            normalized_reward = max(-5, min(5, reward / self.reward_norm_factor))
            #print(f"Debug: Reward={reward}, Normalized Reward={normalized_reward:.4f}, Norm Factor={self.reward_norm_factor:.4f}")

            # Calculate target using the target network
            target = normalized_reward
            if not done:
                target += self.gamma * torch.max(self.target_model(next_state_tensor)).item()

            # Prepare target Q-values for the specific action
            target_f = self.model(state_tensor).detach().clone()
            target_f[0][action] = target

            optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = loss_fn(output, target_f)
            loss.backward()
            optimizer.step()

        scheduler.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)