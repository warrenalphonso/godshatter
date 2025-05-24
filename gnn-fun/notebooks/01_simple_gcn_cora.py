#%% [markdown]
# # Simple GCN on Cora Dataset
#
# This notebook implements a basic Graph Convolutional Network (GCN)
# using PyTorch Geometric on the Cora dataset.

#%%
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

torch.manual_seed(12345) # For reproducibility

print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available(): # MPS for Apple Silicon
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

#%% [markdown]
# ## 1. Load Data
#
# We'll use the Cora dataset, a standard citation network benchmark.
# It consists of scientific publications connected by citations.
# The task is to classify each publication into one of several categories.

#%%
dataset = Planetoid(root='../data/Cora', name='Cora') # Save dataset locally in ../data/
data = dataset[0].to(device)

print("\nDataset Details:")
print(f"  Graphs: {len(dataset)}, Features: {dataset.num_features}, Classes: {dataset.num_classes}")
print(f"Graph object: {data}")
print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}")
print(f"  Isolated nodes: {data.has_isolated_nodes()}, Self-loops: {data.has_self_loops()}, Undirected: {data.is_undirected()}")
print(f"  Node attributes: {list(data.to_dict().keys())}")
print(f"  Training nodes: {data.train_mask.sum().item()}, Val nodes: {data.val_mask.sum().item()}, Test nodes: {data.test_mask.sum().item()}")

#%% [markdown]
# ## 2. Define the GCN Model
#
# We'll define a simple GCN model with two `GCNConv` layers.

#%%
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=16):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1) # Use log_softmax for NLLLoss

model = GCN(dataset.num_features, dataset.num_classes).to(device)
print("\nModel Architecture:")
print(model)

#%% [markdown]
# ## 3. Training the Model
#
# We'll implement a standard training loop.

#%%
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()

def train():
      model.train()
      optimizer.zero_grad()
      out = model(data.x, data.edge_index)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()
      return loss.item()

print("\nStarting training...")
for epoch in range(1, 201):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

#%% [markdown]
# ## 4. Evaluating the Model
#
# After training, we'll evaluate the model on the test set.

#%%
@torch.no_grad() # Decorator to disable gradient calculations during inference
def test():
      model.eval() # Set the model to evaluation mode
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)

      # Calculate accuracy for train, validation, and test sets
      train_correct = pred[data.train_mask] == data.y[data.train_mask]
      train_acc = int(train_correct.sum()) / int(data.train_mask.sum())

      val_correct = pred[data.val_mask] == data.y[data.val_mask]
      val_acc = int(val_correct.sum()) / int(data.val_mask.sum())

      test_correct = pred[data.test_mask] == data.y[data.test_mask]
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())

      return train_acc, val_acc, test_acc

train_acc, val_acc, test_acc = test()
print(f'\nFinished training and evaluation.')
print(f'Train Accuracy: {train_acc:.4f}')
print(f'Validation Accuracy: {val_acc:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')

# %%
