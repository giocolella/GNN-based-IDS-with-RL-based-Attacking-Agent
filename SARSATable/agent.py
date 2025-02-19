import numpy as np


def discretize_state(state, bins=10):
    """
    Converts a vector of 13 features into a discrete state represented as a tuple.
    It assumes the features are normalized to [0, 1]. For each feature, multiply by 'bins' and round.
    """
    discrete = tuple(int(min(bins - 1, max(0, round(feature * bins)))) for feature in state)
    return discrete

class TableSARSAAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = 0.01  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.bins = 13  # Number of bins per feature for discretization
        self.Q = {}  # Q-table: keys are discrete states, values are numpy arrays of Q-values

        # Reward normalization attributes
        self.reward_norm_decay = 0.99
        self.reward_norm_factor = None  # Updated using exponential moving average

    def discretize(self, state):
        """
        First normalizes the raw state using predetermined min/max values,
        then discretizes it using the provided discretize_state function.
        """
        return discretize_state(state, bins=self.bins)

    def get_Q(self, state):
        """
        Returns the Q-values for a state, initializing if necessary.
        The agent handles discretization.
        """
        discrete_state = self.discretize(state)
        if discrete_state not in self.Q:
            self.Q[discrete_state] = np.zeros(self.action_size)
        return self.Q[discrete_state]

    def act(self, state):
        """Selects an action using an epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        Qs = self.get_Q(state)
        return np.argmax(Qs)

    def normalize_reward(self, reward):
        """
        Normalizes and clips the reward based on an exponential moving average of absolute rewards.
        """
        current_abs = max(abs(reward), 1e-6)  # Avoid division by zero
        if self.reward_norm_factor is None:
            self.reward_norm_factor = current_abs
        else:
            self.reward_norm_factor = (self.reward_norm_decay * self.reward_norm_factor +
                                       (1 - self.reward_norm_decay) * current_abs)
        norm_reward = np.clip(reward / self.reward_norm_factor, -5, 5)
        return norm_reward

    def update(self, state, action, reward, next_state, next_action):
        """
        Performs the SARSA update with reward normalization:
            Q(s, a) = Q(s, a) + alpha * (normalized_reward + gamma * Q(s', a') - Q(s, a))
        """
        norm_reward = self.normalize_reward(reward)
        s_key = self.discretize(state)
        ns_key = self.discretize(next_state)

        # Ensure Q-values are initialized
        self.get_Q(state)
        self.get_Q(next_state)

        current = self.Q[s_key][action]
        next_val = self.Q[ns_key][next_action]
        td_target = norm_reward + self.gamma * next_val
        td_error = td_target - current
        self.Q[s_key][action] += self.alpha * td_error

        # Decay epsilon after each update
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
