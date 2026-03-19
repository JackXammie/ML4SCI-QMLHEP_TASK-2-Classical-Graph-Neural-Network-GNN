import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# 1. Load dataset
data = np.load("data/QG_jets.npz")
X = data["X"]
y = data["y"]

# Reduce dataset for faster training
X = X[:2000]
y = y[:2000]

# 2. Graph construction (k-NN)
def build_graph(particles, k=5):
    coords = particles[:, 1:3]  # rapidity, phi
    num_nodes = coords.shape[0]

    edge_index = []

    for i in range(num_nodes):
        distances = np.linalg.norm(coords - coords[i], axis=1)
        nearest = np.argsort(distances)[1:k+1]

        for j in nearest:
            edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(particles, dtype=torch.float)

    return Data(x=x, edge_index=edge_index)

# 3. Convert dataset to graphs
graphs = []
for i in range(len(X)):
    graph = build_graph(X[i])
    graph.y = torch.tensor([y[i]], dtype=torch.long)
    graphs.append(graph)

# 4. Train-test split
train_graphs, test_graphs = train_test_split(graphs, test_size=0.2)

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32)

# 5. Define GCN model
class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(4, 32)
        self.conv2 = GCNConv(32, 64)
        self.fc = nn.Linear(64, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.fc(x)

# 6. Define GAT model
class GAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(4, 32, heads=2)
        self.conv2 = GATConv(64, 64)
        self.fc = nn.Linear(64, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.fc(x)

# 7. Training function
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

# 8. Evaluation
def evaluate(model, loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            out = model(data)
            probs = torch.softmax(out, dim=1)[:, 1]
            preds = torch.argmax(out, dim=1)

            y_true.extend(data.y.numpy())
            y_pred.extend(probs.numpy())

    acc = accuracy_score(y_true, np.round(y_pred))
    auc = roc_auc_score(y_true, y_pred)

    return acc, auc

# 9. Train both models
def run_model(model_class, name):
    model = model_class()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        loss = train(model, train_loader, optimizer, criterion)
        print(f"{name} Epoch {epoch+1}, Loss: {loss:.4f}")

    acc, auc = evaluate(model, test_loader)
    print(f"{name} Accuracy: {acc:.4f}, AUC: {auc:.4f}\n")

# 10. Run experiments
run_model(GCN, "GCN")
run_model(GAT, "GAT")
