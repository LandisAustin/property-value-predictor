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


# Initialize model
model = LinearRegression()

# Train the model using training data
# The model learns a relationship: features (X) -> target (y)
model.fit(X_train_scaled, y_train)


# ===== Make Predictions ===== #
# Use the trained model to predict house values for unseen test data
y_pred = model.predict(X_test_scaled)
y_pred = np.expm1(y_pred)

#Convert y-test back into dollar, and use it moving forward ~ G
y_test_real = np.expm1(y_test)

# ===== Evaluate Precision ===== #
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))
mae = mean_absolute_error(y_test_real, y_pred)
r2 = r2_score(y_test_real, y_pred) 

print("\nLinear Regression:")
print("RMSE (dollars):", rmse)
print("MAE  (dollars):", mae)
print("R²:", r2)

print("\nRidge (Linear) Regression:")
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]  # lower = closer to LinearRegression, higher = more regularization

for a in alphas:
    model = Ridge(alpha=a)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))
    mae = mean_absolute_error(y_test_real, y_pred)
    print(f"alpha={a} -> RMSE={rmse:.2f} | MAE={mae:.2f}")


from sklearn.tree import DecisionTreeRegressor
# Initialize the model
tree = DecisionTreeRegressor(max_depth=10, random_state=42)  # limit depth to prevent overfitting

# Fit model
tree.fit(X_train, y_train)

# Predict
y_pred_tree = tree.predict(X_test)
y_pred_tree = np.expm1(y_pred_tree)

# Evaluate
rmse_tree = np.sqrt(mean_squared_error(y_test_real, y_pred_tree))
mae_tree = mean_absolute_error(y_test_real, y_pred_tree)
r2_tree = r2_score(y_test_real, y_pred_tree)

print("\nDecision Tree Regression:")
print("RMSE (dollars):", rmse_tree)
print("MAE  (dollars):", mae_tree)
print("R²:", r2_tree)


from sklearn.ensemble import RandomForestRegressor
X_train = X_train.sample(n=750_000, random_state=42) #Bump up to 750,000 samples to see if it yields better accuracy ~ G
y_train = y_train.loc[X_train.index]

# ===== Initialize Random Forest ===== #
rf_model = RandomForestRegressor(
    n_estimators=200,    # number of trees
    max_depth=25,        # max depth of each tree (tune for performance vs overfitting)
    min_samples_split = 7, #Requires at least 7 samples for it to split into its own node ~ G
    random_state=42,
    n_jobs=-1            # use all CPU cores
)

# ===== Train Model ===== #
rf_model.fit(X_train, y_train)
print("Model Fit")

# ===== Make Predictions ===== #
y_pred_rf = rf_model.predict(X_test)
y_pred_rf = np.expm1(y_pred_rf)
print("Model Predicted")

# ===== Evaluate Performance ===== #
rmse_rf = np.sqrt(mean_squared_error(y_test_real, y_pred_rf))
mae_rf = mean_absolute_error(y_test_real, y_pred_rf)
r2_rf = r2_score(y_test_real, y_pred_rf)

print("\nRandom Forest Regression:")
print(f"RMSE (dollars): {rmse_rf}")
print(f"MAE  (dollars): {mae_rf}")
print(f"R²: {r2_rf}")


# ***** ALL NEW STUFF ***** ~ G #

# ===== Developing the Deep Learning Models ===== #
import torch
import torch.nn as nn #Importing neural network component of pytorch

#First define the FFNN
class FeedforwardNeuralNetwork(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(FeedforwardNeuralNetwork, self).__init__() #Allows inheritance of PyTorch functionality
        self.fc1 = nn.Linear(inputDim, hiddenDim) #The first linear layer, feeds to hidden layer
        self.bn1 = nn.BatchNorm1d(hiddenDim)
        self.relu = nn.ReLU() #Second layed, non-linear, ReLu, transforms to non-linear representations
        self.dropout = nn.Dropout(0.2) #Dropout using 20% probabilty to reduce overfitting, randomly drops neurons during training
        self.fc2 = nn.Linear(hiddenDim, outputDim) #Output layer, linear, hidden
    
    #Method to forward the classifications
    def forward(self, input):
        out = self.fc1(input) #Feed the input in to get an initial output
        out = self.bn1(out)
        #Continue feeding the output through
        out = self.relu(out) #applies the non-linear aspect
        out = self.dropout(out) #Apply dropout
        out = self.fc2(out) #output layer
        return out 

#Train the FFNN
dimIn = 69 #Number of input dimensions
dimHidden = 2 #Number of hidden dimensions, modified to alter results
dimOut = 1 #Number of output dimensions, just 'taxvaluedollarcount'

model = FeedforwardNeuralNetwork(dimIn, dimHidden, dimOut) #Define the model using specific parameters
criterion = nn.MSELoss() #MSE Loss for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #Adam optimizer because its the fastest

# ---Create tensors for the numpy arrays---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---Sample 750,000 rows from the training set---
X_train_sampled = X_train.sample(n=750_000, random_state=42)
y_train_sampled = y_train.loc[X_train_sampled.index]

# ---Scale features---
X_train_scaled = scaler.fit_transform(X_train_sampled)  # Only scale sampled rows
X_test_scaled = scaler.transform(X_test)  # test set stays full

# --- Convert to tensors ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_sampled.values, dtype=torch.float32).unsqueeze(1).to(device)

print("\nFeedforward Neural Network Regression: ")

print("\n---Epoch Losses---")
# ---Loop through and train---
num_epochs = 40
for epoch in range(num_epochs):
    inputs = X_train_tensor
    targets = y_train_tensor

    #Pass forward
    outputs = model(inputs)

    #Compute loss
    loss = criterion(outputs, targets)

    #Back propagation
    optimizer.zero_grad() #Clear previous gradients
    loss.backward() #New graidents 
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #Gradient clipping to remove outliers
    optimizer.step() #Does the actual update of weights

    #Mainly for progress tracking ever 5 passes
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# ---Run on the test data---
#Create tensors for test data
x_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1).to(device)
y_pred_dl = model(x_test_tensor)
y_pred_real_dl = np.expm1(y_pred_dl.detach().cpu().numpy()) #Convert back from log scale
y_pred_real_dl = np.clip(y_pred_real_dl, 0, 1e8) #Clip extreme values, mostly for larger values
y_test_real_dl = np.expm1(y_test_tensor.cpu().numpy())
y_test_real_dl = np.clip(y_test_real_dl, 0, 1e8)

# ===== Evaluate Performance ===== #
rmse_dl = np.sqrt(mean_squared_error(y_test_real_dl, y_pred_real_dl))
mae_dl = mean_absolute_error(y_test_real_dl, y_pred_real_dl)
r2_dl = r2_score(y_test_real_dl, y_pred_real_dl)

print("\nFeedforward Neural Network Regression Metrics:")
print(f"RMSE (dollars): {rmse_dl}")
print(f"MAE  (dollars): {mae_dl}")
print(f"R²: {r2_dl}")



