from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("zillow-data/properties_2016_cleaned.csv", low_memory=False)



# Evaluation helper function

def evaluate_model(y_pred_log, y_test_log, model_name = "Model"):
    y_pred_real = np.expm1(y_pred_log)  # inverse of log1p
    y_test_log = y_test
    y_test_real = np.expm1(y_test)

    # Log errors
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

    # Dollar errors
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae = mean_absolute_error(y_test_real, y_pred_real)
    r2 = r2_score(y_test_real, y_pred_real)

    print(f"\n{model_name} Dollar Metrics:")
    print(f"RMSE (dollars): {rmse:,.2f}")
    print(f"MAE  (dollars): {mae:,.2f}")
    print(f"R²: {r2:.4f}")

    # Compare ranges
    print("Predicted range:", y_pred_real.min(), "-", y_pred_real.max())
    print("Actual range   :", y_test_real.min(), "-", y_test_real.max())

    # Plot predicted vs actual
    plt.figure(figsize=(6,6))
    plt.scatter(y_test_real, y_pred_real, alpha=0.3, s=10)
    max_val = max(y_test_real.max(), y_pred_real.max())
    plt.plot([0, max_val], [0, max_val], color='red', linewidth=2)
    plt.xlabel("Actual Home Value")
    plt.ylabel("Predicted Home Value")
    plt.title(f"{model_name} - Predicted vs Actual")
    plt.show()



# 1. Trim outliers from the training set
# Some houses has value of 1, some with 219million. So there's definitely errors in some values
df = df[(df['taxvaluedollarcnt'] > 50_000) & (df['taxvaluedollarcnt'] < 10_000_000)]

p = 0.07
lower = df['taxvaluedollarcnt'].quantile(p)
upper = df['taxvaluedollarcnt'].quantile(1 - p)
df = df[(df['taxvaluedollarcnt'] >= lower) & (df['taxvaluedollarcnt'] <= upper)]

print(df['taxvaluedollarcnt'].describe())


# 2. Features / target
X = df.drop(columns=['taxvaluedollarcnt'])
y = np.log1p(df['taxvaluedollarcnt'])  # log1p avoids log(0)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Models & Predict & Evaluate

# Linear Regressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred_log = model.predict(X_test_scaled)
evaluate_model(y_pred_log, y_test, "Linear Regression")

X_train = X_train.sample(n=100_000, random_state=42)
y_train = y_train.loc[X_train.index]

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
# Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=30,
    min_samples_split=10,
    min_samples_leaf=7,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_log = rf_model.predict(X_test)
evaluate_model(y_pred_log, y_test, "Random Forest")

# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    min_samples_leaf=7,
    random_state=42
)
gb_model.fit(X_train, y_train)
y_pred_log = gb_model.predict(X_test)
evaluate_model(y_pred_log, y_test, "GBoosted")


# ---- Ensemble Random Forest ----
from sklearn.model_selection import cross_val_score
# Model
rf_model = RandomForestRegressor(
    n_estimators=200,          # lower for speed during CV
    max_depth=30,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

# Custom RMSE scorer (dollar space)
def rmse_dollars(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse_dollars, greater_is_better=False)

# Cross-validation
scores = cross_val_score(
    rf_model,
    X_small,
    y_small,
    cv=5,
    scoring=rmse_scorer
)

rmse_scores = -scores

print("RMSE for each fold:", rmse_scores)
print("Average RMSE:", rmse_scores.mean())