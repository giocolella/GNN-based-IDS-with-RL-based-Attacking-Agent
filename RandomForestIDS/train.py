import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score, matthews_corrcoef
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from environment import *
from agent import *
from sklearn.utils import shuffle
# Import the Random Forest IDS
from gnn_ids import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def load_csv_data(file_path):
    data = pd.read_csv(file_path)
    features = data.iloc[:, :-1].values  # All columns except the last one
    labels = data['Label'].apply(lambda x: 0 if x == "Benign" else 1).values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return train_test_split(features, labels, test_size=0.2, random_state=42)

# Pre-train the RL agent
def pretrain_agentMSE(agent, features, labels, epochs=1, batch_size=32):
    optimizer = optim.Adam(agent.model.parameters(), lr=agent.learning_rate)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(epochs):
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            optimizer.zero_grad()
            inputs = torch.tensor(batch_features, dtype=torch.float32)
            targets = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(1)
            outputs = agent.model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

set_seed(16)

# Initialize the environment (the environment still expects a .forward(x, edge_index) method)
env = NetworkEnvironment(gnn_model=None)
state_size = env.state_size
action_size = env.action_size

# Initialize the IDS as a Random Forest classifier
gnn_model = RFIDS()

# (No optimizer, scheduler, or loss_fn are needed for the IDS now.)
env.gnn_model = gnn_model

# Initialize the RL agent (unchanged)
agent = DDQNAgent(state_size=state_size, action_size=4)
optimizerAgent = optim.Adam(agent.model.parameters(), lr=0.14)
schedulerAgent = torch.optim.lr_scheduler.StepLR(optimizerAgent, step_size=10, gamma=1)

# Training hyperparameters
num_episodes = 160
batch_size = 32
retrain_interval = 20

# Metrics storage
episodic_rewards = []
epsilon_values = []
traffic_data = []
labels = []
recorded_episodes = []
ids_metrics = {
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'Balanced Accuracy': [],
    'Mcc': []
}

window_size = 10000

# Load and preprocess the dataset for pretraining the RL agent
X_train, X_test, y_train, y_test = load_csv_data("C:/Users/colg/Desktop/mergedfiltered.csv")

pretrain_agentMSE(agent, X_train, y_train)

# Pretrain the IDS classifier (Random Forest) using a CSV dataset
gnn_model.pretrain("C:/Users/colg/Desktop/cleaned_ids2018_sampledfiltered.csv")
print("Pre-training Completed")

# Main training loop
for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0
    actions_taken = []

    for step in range(50):
        action = agent.act(state)
        actions_taken.append(action)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
        if done:
            break

    episodic_rewards.append(total_reward)
    epsilon_values.append(agent.epsilon)
    print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward:.1f}")

    benign_count = actions_taken.count(0)
    malicious_count = actions_taken.count(1)
    #print(f"Actions Taken - Benign: {benign_count}, Malicious: {malicious_count}")

    # Train the RL agent
    if len(agent.memory) > batch_size:
        agent.replay(batch_size, optimizerAgent, schedulerAgent, torch.nn.MSELoss())

    # Collect traffic data for retraining
    traffic_data.extend(env.traffic_data)
    labels.extend(env.labels)
    if len(traffic_data) > window_size:
        traffic_data = traffic_data[-window_size:]
        labels = labels[-window_size:]

    # Retrain the IDS every retrain_interval episodes
    if episode % retrain_interval == 0:
        agent.update_target_network()
        print("Retraining IDS (Random Forest)...")
        # Retrain the RFIDS on the collected traffic data
        logits_before = gnn_model.forward(torch.tensor(traffic_data[:5], dtype=torch.float32))
        #print("IDS Output Before Training:", torch.sigmoid(logits_before).detach().numpy())
        unique, counts = np.unique(labels, return_counts=True)
        #print("Label distribution before retraining IDS:", dict(zip(unique, counts)))

        # Shuffle the data to prevent overfitting
        traffic_data, labels = shuffle(traffic_data, labels)

        # Limit training samples to prevent excessive memorization
        sample_size = min(len(traffic_data), 10000)  # Limit to 5000 samples
        gnn_model.retrain(traffic_data[:sample_size], labels[:sample_size])

        logits_after = gnn_model.forward(torch.tensor(traffic_data[:5], dtype=torch.float32))
        #print("IDS Output After Training:", torch.sigmoid(logits_after).detach().numpy())

        # Evaluate IDS performance
        X_all = np.array(traffic_data, dtype=np.float32)
        X_tensor = torch.tensor(X_all)
        logits = gnn_model.forward(X_tensor)
        # Using sigmoid to convert logits to probabilities, then round to 0/1
        predictions = torch.sigmoid(logits).detach().round().squeeze().numpy()

        accuracy = accuracy_score(labels, predictions) * 100
        precision = precision_score(labels, predictions, average="binary", zero_division=1) * 100
        recall = recall_score(labels, predictions, average="binary", zero_division=1) * 100
        f1 = f1_score(labels, predictions, average="binary",  zero_division=1) * 100
        balanced_accuracy = balanced_accuracy_score(labels, predictions) * 100
        mcc = matthews_corrcoef(labels, predictions)

        ids_metrics['Accuracy'].append(accuracy)
        ids_metrics['Precision'].append(precision)
        ids_metrics['Recall'].append(recall)
        ids_metrics['F1 Score'].append(f1)
        ids_metrics['Balanced Accuracy'].append(balanced_accuracy)
        ids_metrics['Mcc'].append(mcc)

        print(f"IDS Performance - Accuracy: {accuracy:.4f}%, Precision: {precision:.4f}%, Recall: {recall:.4f}%, F1-Score: {f1:.4f}%")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}%, MCC: {mcc:.4f}")
        cm = confusion_matrix(labels, predictions)
        print("Confusion Matrix:")
        print(cm)

        env.traffic_data = []
        env.labels = []

print("Training complete.")
