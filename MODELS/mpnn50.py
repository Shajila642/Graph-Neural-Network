import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load the preprocessed data
processed_data = pd.read_excel("H:/MODEL/Datasettprocessed.xlsx")

# Drop rows with missing target values (yield)
processed_data = processed_data.dropna(subset=['yield'])

# Split features and target variable
X = processed_data.drop(columns=['yield'])  # Features
y = processed_data['yield']  # Target variable

# Handle missing values and encoding non-numeric features
non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = imputer.fit_transform(X)
X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)
X_encoded = pd.get_dummies(X_imputed_df, columns=non_numeric_columns, drop_first=True)

# Handle infinities and scale features
X_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
mean_imputer = SimpleImputer(strategy='mean')
X_encoded_imputed = mean_imputer.fit_transform(X_encoded)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded_imputed)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# RandomForest for feature extraction
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
X_train_rf_features = rf_regressor.predict(X_train).reshape(-1, 1)
X_test_rf_features = rf_regressor.predict(X_test).reshape(-1, 1)

# Create graph data with k-nearest neighbors
def prepare_graph_data_kneighbors(X_features, y_target, k=5):
    edge_index = kneighbors_graph(X_features, k, mode='connectivity', include_self=False).tocoo()
    edge_indices = torch.tensor(np.vstack((edge_index.row, edge_index.col)), dtype=torch.long)
    x = torch.tensor(X_features, dtype=torch.float)
    y = torch.tensor(y_target.values, dtype=torch.float).view(-1, 1)
    return Data(x=x, edge_index=edge_indices, y=y)

graph_data_train = prepare_graph_data_kneighbors(X_train_rf_features, y_train)
graph_data_test = prepare_graph_data_kneighbors(X_test_rf_features, y_test)

# Define the MPNN Layer
class MPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNNLayer, self).__init__(aggr='mean')  # Use mean aggregation
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Start message passing
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # Message computation: linear transformation of neighbor features
        return self.linear(x_j)

# Define the MPNN Model
class MPNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MPNNModel, self).__init__()
        self.mpnn1 = MPNNLayer(input_dim, hidden_dim)
        self.mpnn2 = MPNNLayer(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.mpnn1(x, edge_index))
        x = F.relu(self.mpnn2(x, edge_index))
        x = self.fc_out(x)
        return x

# Initialize model, loss function, and optimizer
model = MPNNModel(input_dim=1, hidden_dim=64, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 50
train_losses = []  # To store the training losses

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data_train)
    loss = criterion(out, graph_data_train.y)
    loss.backward()
    optimizer.step()

    # Track loss
    train_losses.append(loss.item())

    # Print progress
    if epoch % 10 == 0:
        train_r2 = r2_score(y_train, out.detach().numpy().flatten())
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Train R^2: {train_r2:.4f}')

# Evaluate the model on test data
model.eval()
with torch.no_grad():
    train_pred = model(graph_data_train)
    test_pred = model(graph_data_test)
    train_loss = criterion(train_pred, graph_data_train.y)
    test_loss = criterion(test_pred, graph_data_test.y)

    # Calculate R^2 scores
    r2_train = r2_score(y_train, train_pred.numpy().flatten())
    r2_test = r2_score(y_test, test_pred.numpy().flatten())

print(f'Training Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
print(f'R^2 Score (Train): {r2_train:.4f}, R^2 Score (Test): {r2_test:.4f}')

# Visualization functions
def plot_experimental_vs_predicted(y_true, y_pred, image_path=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel('Experimental Yields')
    plt.ylabel('Predicted Yields')
    plt.title('Experimental vs. Predicted Yields')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2)  # y=x line
    plt.grid(True)
    
    # Calculate and display metrics
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    plt.text(min(y_true), max(y_pred), f'R^2 = {r2:.2f}', fontsize=12, color='black')
    plt.text(min(y_true), max(y_pred) - 5, f'RMSE = {rmse:.2f}', fontsize=12, color='black')
    plt.text(min(y_true), max(y_pred) - 10, f'MAE = {mae:.2f}', fontsize=12, color='black')
    
    if image_path:
        plt.savefig(image_path)
    plt.show()

# Plot Experimental vs. Predicted Yields
plot_experimental_vs_predicted(y_test, test_pred.cpu().numpy().flatten(), 'experimental_vs_predicted.png')

# Plot Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')
plt.show()

# Plot R-squared Scores
plt.figure(figsize=(10, 6))
bars = plt.bar(['Training', 'Test'], [r2_train, r2_test], color=['blue', 'green'])
plt.xlabel('Dataset')
plt.ylabel('R-squared Score')
plt.title('R-squared Scores for Training and Test Sets')

# Add text annotations
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), va='bottom')  # va: vertical alignment

plt.grid(True)
plt.savefig('r2_scores.png')
plt.show()
