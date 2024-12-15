import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from environment import NetworkEnvironment
from gnn_ids import GCNIDS, preprocess_data, retrain
from agent import DQNAgent

# Initialize the environment
env = NetworkEnvironment(gnn_model=None)  # GNN model will be attached later
state_size = env.state_size
action_size = env.action_size

# Initialize the GNN-based IDS
gnn_model = GCNIDS(input_dim=state_size, hidden_dim=32, output_dim=1)
optimizer = optim.Adam(gnn_model.parameters(), lr=0.001)
loss_fn = torch.nn.BCELoss()

# Attach GNN model to the environment
env.gnn_model = gnn_model

# Initialize the RL agent
agent = DQNAgent(state_size=state_size, action_size=action_size)

# Training hyperparameters
num_episodes = 100
batch_size = 32
retrain_interval = 5

# Store metrics
ids_metrics = []
traffic_data = []
labels = []

for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0
    actions_taken = []

    for step in range(50):  # Simulate up to 50 steps per episode
        action = agent.act(state)
        actions_taken.append(action)
        next_state, reward, done, _ = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state

        if done:
            break

    print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward:.1f}")

    # Train the agent
    if len(agent.memory) > batch_size:
        agent.replay(batch_size, optimizer, loss_fn)

    # Collect traffic data for retraining
    traffic_data.extend(env.traffic_data)
    labels.extend(env.labels)

    # Retrain the IDS every retrain_interval episodes
    if episode % retrain_interval == 0:
        print("Retraining IDS...")
        retrain(gnn_model, traffic_data, labels, epochs=5, batch_size=batch_size)

        # Evaluate IDS performance
        graph_data = preprocess_data(traffic_data, labels)
        #predictions = gnn_model(graph_data.x, graph_data.edge_index).round().squeeze().numpy()
        predictions = gnn_model(graph_data.x, graph_data.edge_index).detach().round().squeeze().numpy() #Added detach() to calculate FP and FN

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions) * 100
        precision = precision_score(labels, predictions, zero_division=1) * 100
        recall = recall_score(labels, predictions, zero_division=1) * 100
        f1 = f1_score(labels, predictions, zero_division=1) * 100

        # Calculate FP and FN
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()  # Extract confusion matrix values

        print(f"Retrained IDS - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        print(f"False Positives: {fp}, False Negatives: {fn}")