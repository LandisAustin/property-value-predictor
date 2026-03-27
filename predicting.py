import pandas as pd

# ===== Load Cleaned Dataset ===== #
# Read in the dataset after preprocessing/cleaning
df = pd.read_csv("zillow-data/properties_2016_cleaned.csv", low_memory=False)

print(df.columns)


# ===== Separate Features (X) and Target (y) ===== #
# X = all input variables (features) used to make predictions
# y = output variable (target) we are trying to predict
# IMPORTANT: The target must NOT be included in X, or the model will "cheat"
X = df.drop(columns=['taxvaluedollarcnt'])  # remove target column from features
y = df['taxvaluedollarcnt']                 # isolate target column


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
import numpy as np

# Initialize model
model = LinearRegression()

# Train the model using training data
# The model learns a relationship: features (X) -> target (y)
model.fit(X_train_scaled, y_train)


# ===== Make Predictions ===== #
# Use the trained model to predict house values for unseen test data
y_pred = model.predict(X_test_scaled)

# ===== Evaluate Precision ===== #

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

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
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"alpha={a} -> RMSE={rmse:.2f} | MAE={mae:.2f}")


from sklearn.tree import DecisionTreeRegressor
# Initialize the model
tree = DecisionTreeRegressor(max_depth=10, random_state=42)  # limit depth to prevent overfitting

# Fit model
tree.fit(X_train, y_train)

# Predict
y_pred_tree = tree.predict(X_test)

# Evaluate
rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
mae_tree = mean_absolute_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print("\nDecision Tree Regression:")
print("RMSE (dollars):", rmse_tree)
print("MAE  (dollars):", mae_tree)
print("R²:", r2_tree)


from sklearn.ensemble import RandomForestRegressor
X_train = X_train.sample(n=500_000, random_state=42)
y_train = y_train.loc[X_train.index]

# ===== Initialize Random Forest ===== #
rf_model = RandomForestRegressor(
    n_estimators=50,    # number of trees
    max_depth=15,        # max depth of each tree (tune for performance vs overfitting)
    random_state=42,
    n_jobs=-1            # use all CPU cores
)

# ===== Train Model ===== #
rf_model.fit(X_train, y_train)
print("Model Fit")

# ===== Make Predictions ===== #
y_pred_rf = rf_model.predict(X_test)
print("Model Predicted")

# ===== Evaluate Performance ===== #
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Regression:")
print(f"RMSE (dollars): {rmse_rf}")
print(f"MAE  (dollars): {mae_rf}")
print(f"R²: {r2_rf}")