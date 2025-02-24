import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.neighbors import kneighbors_graph

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

# Identify non-numeric columns
non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns

# Impute missing values with most frequent strategy for non-numeric columns
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = imputer.fit_transform(X)

# Convert X_imputed back to a DataFrame
X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

# Encode non-numeric data (using dummy encoding as an example)
X_encoded = pd.get_dummies(X_imputed_df, columns=non_numeric_columns, drop_first=True)

# Check for infinite values and replace them with NaN
X_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)

# Impute remaining NaN values using mean for numeric features
mean_imputer = SimpleImputer(strategy='mean')
X_encoded_imputed = mean_imputer.fit_transform(X_encoded)

# Check for NaN or Infinity in the encoded DataFrame
if np.any(np.isnan(X_encoded_imputed)) or np.any(np.isinf(X_encoded_imputed)):
    raise ValueError("X_encoded_imputed contains NaN or infinity values.")

# Convert to DataFrame and ensure all are numeric
X_encoded_imputed = pd.DataFrame(X_encoded_imputed, columns=X_encoded.columns)
X_encoded_imputed = X_encoded_imputed.apply(pd.to_numeric, errors='coerce')

# Handle large values
X_encoded_imputed = np.clip(X_encoded_imputed, -1e10, 1e10)  # Adjust as necessary

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_regressor.fit(X_train, y_train)

# Extract features using the trained RF model
X_train_rf_features = rf_regressor.predict(X_train).reshape(-1, 1)
X_test_rf_features = rf_regressor.predict(X_test).reshape(-1, 1)

# Prepare graph data with k-nearest neighbors for edge creation
def prepare_graph_data_kneighbors(X_features, y_target, k=5):
    num_nodes = X_features.shape[0]
    # Use k-nearest neighbors to create edges
    edge_index = kneighbors_graph(X_features, k, mode='connectivity', include_self=False).tocoo()
    edge_indices = torch.tensor(np.vstack((edge_index.row, edge_index.col)), dtype=torch.long)
    x = torch.tensor(X_features, dtype=torch.float)
    y = torch.tensor(y_target.values, dtype=torch.float).view(-1, 1)
    data = Data(x=x, edge_index=edge_indices, y=y)
    return data

# Prepare graph data using k-nearest neighbors
graph_data_train = prepare_graph_data_kneighbors(X_train_rf_features, y_train)
graph_data_test = prepare_graph_data_kneighbors(X_test_rf_features, y_test)

# Define the GCN model for regression
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, output_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # First GCN layer + ReLU
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        
        # Second GCN layer + ReLU
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        # Output layer (no activation for regression)
        x = self.conv3(x, edge_index)
        return x

# Initialize the GCN model
input_dim = 1  # Adjust based on the number of features (here, RandomForest features)
hidden_dim1 = 64
hidden_dim2 = 128
output_dim = 1  # For regression, e.g., predicting yield

model = GCNModel(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=output_dim)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop for GCN
def train_gcn(model, data, optimizer, criterion, epochs=50):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()  # Zero gradients
        out = model(data)  # Forward pass
        loss = criterion(out, data.y)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss: {loss.item():.4f}')

# Train the GCN model
train_gcn(model, graph_data_train, optimizer, criterion, epochs=50)

model.eval()
with torch.no_grad():
    train_pred = model(graph_data_train)
    test_pred = model(graph_data_test)
    train_loss = criterion(train_pred, graph_data_train.y)
    test_loss = criterion(test_pred, graph_data_test.y)
    print('Training Loss: {:.4f}, Test Loss: {:.4f}'.format(train_loss.item(), test_loss.item()))

# Calculate R^2 scores with GCN
r2_train_gnn = r2_score(y_train, train_pred.cpu().numpy().flatten())
r2_test_gnn = r2_score(y_test, test_pred.cpu().numpy().flatten())
print("R^2 Score (Training, with GCN):", r2_train_gnn)
print("R^2 Score (Test, with GCN):", r2_test_gnn)

# Visualization

# Scatter Plot of Experimental vs. Predicted Yields
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
        plt.savefig("H:/MODEL/gcn50.png")
    plt.show()

# Plot Experimental vs. Predicted Yields
plot_experimental_vs_predicted(y_test, test_pred.cpu().numpy().flatten(), 'experimental_vs_predicted.png')

