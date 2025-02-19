import numpy as np


def discretize_state(state, bins=10):
    """
    Converts a vector of 13 features into a discrete state represented as a tuple.
    It assumes the features are normalized to [0, 1]. For each feature, multiply by 'bins' and round.
    """
    discrete = tuple(int(min(bins - 1, max(0, round(feature * bins)))) for feature in state)
    return discrete


def normalize_state(state, feature_mins, feature_maxes):
    """
    Normalizes the raw state using per-feature min and max values.
    The result is clipped to [0, 1] for each feature.
    """
    state = np.array(state)
    normalized = (state - feature_mins) / (feature_maxes - feature_mins)
    normalized = np.clip(normalized, 0, 1)
    return normalized


class TableSARSAAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.bins = 10  # Number of bins per feature for discretization
        self.Q = {}  # Q-table: keys are discrete states, values are numpy arrays of Q-values

        # Reward normalization attributes
        self.reward_norm_decay = 0.99
        self.reward_norm_factor = None  # Updated using exponential moving average

        # Define per-feature min and max values (order must match that of the state vector)
        # These values are chosen based on your environment's generate_traffic() method.
        # Adjust these as needed.
        self.feature_mins = np.array([
            1,  # Dst Port (min: 1)
            1,  # Flow Duration (min: could be as low as 1 for some malicious traffic)
            1,  # Tot Fwd Pkts (min: 1)
            0,  # TotLen Fwd Pkts (min: 0)
            0,  # Flow Byts/s (min: 0)
            0,  # Fwd Pkt Len Max (min: 0)
            0,  # Protocol (min: 0, since malicious may set it to 0)
            0,  # Flow Pkts/s (min: 0)
            0,  # ACK Flag Cnt (min: 0)
            0,  # SYN Flag Cnt (min: 0)
            0,  # Fwd IAT Mean (min: 0)
            0,  # Init Fwd Win Byts (min: 0)
            0  # Active Mean (min: 0)
        ], dtype=np.float32)

        self.feature_maxes = np.array([
            65535,  # Dst Port (max: 65535)
            5000,  # Flow Duration (max: 5000, though some malicious traffic may be lower)
            1000,  # Tot Fwd Pkts (max: 1000)
            1e6,  # TotLen Fwd Pkts (max: 1e6)
            1e6,  # Flow Byts/s (max: 1e6, to cover flood attacks)
            1500,  # Fwd Pkt Len Max (max: 1500)
            17,  # Protocol (max: 17, covers typical protocols)
            1e4,  # Flow Pkts/s (max: 1e4, to cover high-rate traffic)
            50,  # ACK Flag Cnt (max: 50)
            50,  # SYN Flag Cnt (max: 50)
            1e4,  # Fwd IAT Mean (max: 1e4)
            1e5,  # Init Fwd Win Byts (max: 1e5)
            1e3  # Active Mean (max: 1e3)
        ], dtype=np.float32)

    def discretize(self, state):
        """
        First normalizes the raw state using predetermined min/max values,
        then discretizes it using the provided discretize_state function.
        """
        norm_state = normalize_state(state, self.feature_mins, self.feature_maxes)
        return discretize_state(norm_state, bins=self.bins)

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
