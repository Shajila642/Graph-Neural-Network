
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import ChebConv
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from torch_geometric.nn import GCNConv

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

# Check shapes before train-test split
print("Shape of X_scaled:", X_scaled.shape)
print("Shape of y:", y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_regressor.fit(X_train, y_train)

# Extract features using trained RF model
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
# Add these imports at the beginning of your code if not already present


class ResGCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ResGCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # First convolution + activation
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = self.dropout(x1)
        
        # Second convolution
        x2 = self.conv2(x1, edge_index)
        
        # Residual connection
        x2 += x  # Adding the input to the output (residual)
        return x2

# Initialize the ResGCN model
model = ResGCNModel(input_dim=1, hidden_dim=64, output_dim=1)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop for ResGCN
def train_resgcn(model, data, optimizer, criterion, epochs=50):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()  # Zero gradients
        out = model(data)  # Forward pass
        loss = criterion(out, data.y)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss: {loss.item():.4f}')

# Train the model
train_resgcn(model, graph_data_train, optimizer, criterion, epochs=50)

# Evaluation on the test set
def evaluate_resgcn(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        return out

# Evaluate the model on test data
y_pred = evaluate_resgcn(model, graph_data_test)

# Compute R^2 score or any other metric as needed
from sklearn.metrics import r2_score

y_test_numpy = graph_data_test.y.numpy()  # Convert to NumPy array for sklearn
r2 = r2_score(y_test_numpy, y_pred.numpy())
print(f'R^2 Score: {r2:.4f}')
