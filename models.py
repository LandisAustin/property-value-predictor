import pandas as pd
import numpy as np #Importing numpy earlier so I can log scale the output
# ===== Load Cleaned Dataset ===== #
# Read in the dataset after preprocessing/cleaning
df = pd.read_csv("zillow-data/properties_2016_cleaned.csv", low_memory=False)

print(df.columns)


# ===== Separate Features (X) and Target (y) ===== #
# X = all input variables (features) used to make predictions
# y = output variable (target) we are trying to predict
# IMPORTANT: The target must NOT be included in X, or the model will "cheat"
X = df.drop(columns=['taxvaluedollarcnt'])  # remove target column from features
y = np.log1p(df['taxvaluedollarcnt']) # isolate target column
#Change to log values

# ===== Train/Test Split ===== #
from sklearn.model_selection import train_test_split

# Split dataset into training and testing sets:
# - Training set: used to train the model (learn patterns)
# - Testing set: used to evaluate model performance on unseen data
#
# test_size=0.2 - 20% of data is reserved for testing, 80% for training
# random_state=42 - ensures the split is reproducible (same rows every run)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ===== Inspect Split Sizes ===== #
# Print shapes to verify split worked as expected
# Format: (number of rows, number of features)
print("Training set shape (X_train):", X_train.shape)
print("Testing set shape  (X_test):", X_test.shape)

# These should roughly follow an 80/20 split of the original dataset


# ===== Feature Scaling ===== #
# Linear regression performs better when features are on similar scales
# (e.g., square feet vs. number of rooms are very different magnitudes)
# StandardScaler transforms data to have mean=0 and standard deviation=1

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit scaler ONLY on training data (IMPORTANT: prevents data leakage)
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same transformation to test data to maintain learned weights
X_test_scaled = scaler.transform(X_test)


# ===== Train Linear Regression Model ===== #
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor


# ----- Helper function to evaluate a model ----- #
def evaluate_model(y_test_log, y_pred_log, model_name="Model"):
    """
    Evaluates a regression model in both log-space and original dollar-space.
    
    Parameters:
    - y_test_log : np.array or pd.Series
        The log-transformed true target values (np.log1p)
    - y_pred_log : np.array or pd.Series
        The model predictions in log-space
    - model_name : str
        Name of the model for labeling outputs
    """
    # ----- Log Error Metrics -----
    log_error = y_pred_log - y_test_log
    print(f"\n{model_name} Log Error Metrics:")
    print(f"Mean Log Error: {np.mean(log_error):.6f}")
    print(f"Mean Abs Log Error: {np.mean(np.abs(log_error)):.6f}")
    print(f"Median Log Error: {np.median(log_error):.6f}")
    
    
    plt.figure(figsize=(6,4))
    plt.hist(log_error, bins=100, alpha=0.7)
    plt.title(f"{model_name} Log Error Distribution")
    plt.xlabel("Log Error")
    plt.ylabel("Frequency")
    plt.show()
    
    
    # ----- Convert back to original dollar-space -----
    y_pred_real = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test_log)
    
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae = mean_absolute_error(y_test_real, y_pred_real)
    r2 = r2_score(y_test_real, y_pred_real)
    
    print(f"\n{model_name} Dollar-Space Metrics:")
    print(f"RMSE (dollars): {rmse:,.2f}")
    print(f"MAE  (dollars): {mae:,.2f}")
    print(f"R²: {r2:.4f}")
    
    return {
        "log_error_mean": np.mean(log_error),
        "log_error_mean_abs": np.mean(np.abs(log_error)),
        "log_error_median": np.median(log_error),
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


# ---- Linear Regression ----
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred_lr_log = model.predict(X_test_scaled)
evaluate_model(y_test, y_pred_lr_log, "Linear Regression")

# ---- Ridge Regression ----
alphas = [0.1] # 1.0, 10.0, 100.0, 1000.0 seem to always be worse
for a in alphas:
    model = Ridge(alpha=a)
    model.fit(X_train_scaled, y_train)
    y_pred_ridge_log = model.predict(X_test_scaled)
    evaluate_model(y_test, y_pred_ridge_log, f"Ridge alpha={a}")
	
	
#X_train = X_train.sample(n=750_000, random_state=42) #Bump up to 750,000 samples to see if it yields better accuracy ~ G
#y_train = y_train.loc[X_train.index]
	

# ---- Decision Tree ----
tree = DecisionTreeRegressor(max_depth=20, min_samples_leaf=5, random_state=42,
    max_features="sqrt"
)
tree.fit(X_train, y_train)
y_pred_tree_log = tree.predict(X_test)
evaluate_model(y_test, y_pred_tree_log, "Decision Tree")

# ---- Random Forest ----
rf_model = RandomForestRegressor(
    n_estimators=500, max_depth=30, min_samples_split=10, random_state=42, n_jobs=-1,
    min_samples_leaf=5, max_features="sqrt"
)
rf_model.fit(X_train, y_train)
y_pred_rf_log = rf_model.predict(X_test)
evaluate_model(y_test, y_pred_rf_log, "Random Forest")

"""
# ---- Gradient Boosting ----
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
gbr.fit(X_train, y_train)
y_pred_gbr_log = gbr.predict(X_test)
evaluate_model(y_test, y_pred_gbr_log, "Gradient Boosting")

# ---- XGBoost ----
xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, subsample=0.8,
                   colsample_bytree=0.8, random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
y_pred_xgb_log = xgb.predict(X_test)
evaluate_model(y_test, y_pred_xgb_log, "XGBoost")
"""

"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Convert data to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)

# Dataset and DataLoader
train_ds = TensorDataset(X_train_tensor, y_train_tensor)
train_dl = DataLoader(train_ds, batch_size=1024, shuffle=True)

# Define a simple feedforward NN (MLP)
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.layers(x)

model = MLP(X_train_scaled.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # increase epochs as needed
    for xb, yb in train_dl:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Predict
with torch.no_grad():
    y_pred_log = model(X_test_tensor).numpy().flatten()
evaluate_model(y_test, y_pred_log, "PyTorch MLP")
"""