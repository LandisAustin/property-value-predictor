from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

ErrorGraph = False
ValueCompGraph = False

LinearRegressor = True
RandomForest = True
ExtraTrees = True
BaggingRegressor = True
VotingRegressor = True
StackRegressor = False
GradientBoostingRegressor = True
XGBRegressor = True

RangeLimited = False
DroppedQuantiles = True
QuantilePercent = 3

print("Reading Data...")
df = pd.read_csv("zillow-data/properties_2016_cleaned.csv", low_memory=False)
print("Data Read Complete")


# Evaluation helper function
results_df = None

def evaluate_model(y_pred_log, y_test_log, model_name="Model"):
    global results_df

    # Convert back to dollars
    y_pred_real = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test_log)

    # -------------------------
    # Store results in table
    # -------------------------
    if results_df is None:
        results_df = pd.DataFrame({
            "Actual": y_test_real
        })

    results_df[model_name] = y_pred_real
    results_df[f"{model_name} Error"] = y_pred_real - y_test_real
    results_df[f"{model_name} % Error"] = (
        (y_pred_real - y_test_real) / y_test_real
    )

    # -------------------------
    # Log errors
    # -------------------------
    log_error = y_pred_log - y_test_log
    print(f"\n{model_name} Log Error Metrics:")
    print(f"Mean Log Error: {np.mean(log_error):.6f}")
    print(f"Mean Abs Log Error: {np.mean(np.abs(log_error)):.6f}")
    print(f"Median Log Error: {np.median(log_error):.6f}")

    if ErrorGraph:
        plt.figure(figsize=(6,4))
        plt.hist(log_error, bins=100, alpha=0.7)
        plt.title(f"{model_name} Log Error Distribution")
        plt.xlabel("Log Error")
        plt.ylabel("Frequency")
        plt.show()

    # -------------------------
    # Dollar errors
    # -------------------------
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mae = mean_absolute_error(y_test_real, y_pred_real)
    r2 = r2_score(y_test_real, y_pred_real)

    print(f"\n{model_name} Dollar Metrics:")
    print(f"RMSE (dollars): {rmse:,.2f}")
    print(f"MAE  (dollars): {mae:,.2f}")
    print(f"R²: {r2:.4f}")

    print("Predicted range:", y_pred_real.min(), "-", y_pred_real.max())
    print("Actual range   :", y_test_real.min(), "-", y_test_real.max())

    # -------------------------
    # Scatter plot
    # -------------------------
    if ValueCompGraph:
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
print("Clipping Data...")
if RangeLimited:
    df = df[(df['taxvaluedollarcnt'] > 50_000) & (df['taxvaluedollarcnt'] < 10_000_000)]
if DroppedQuantiles:
    p = QuantilePercent/100
    lower = df['taxvaluedollarcnt'].quantile(p)
    upper = df['taxvaluedollarcnt'].quantile(1 - p)
    df = df[(df['taxvaluedollarcnt'] >= lower) & (df['taxvaluedollarcnt'] <= upper)]



# 2. Features / target
X = df.drop(columns=['taxvaluedollarcnt'])
y = np.log1p(df['taxvaluedollarcnt'])  # log1p avoids log(0)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Models & Predict & Evaluate

X_train = X_train.sample(n=100_000, random_state=42)
y_train = y_train.loc[X_train.index]

# Linear Regressor
if LinearRegressor:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    print("\nStarting Linear Regressor...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_log = lr_model.predict(X_test_scaled)
    evaluate_model(y_pred_log, y_test, "Linear Regression")

# Random Forest Regressor
if RandomForest:
    from sklearn.ensemble import RandomForestRegressor

    print("\nStarting Random Forest...")
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

# Extra Trees (Randomized RF)
if ExtraTrees:
    from sklearn.ensemble import ExtraTreesRegressor

    print("\nStarting Extra Trees...")
    et_model = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=30,
        min_samples_leaf=7,
        n_jobs=-1,
        random_state=42
    )
    et_model.fit(X_train, y_train)
    y_pred_log = et_model.predict(X_test)
    evaluate_model(y_pred_log, y_test, "Extra Trees")

# Bagging Regressor
if BaggingRegressor:
    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor

    print("\nStarting Bagging Regressor...")
    bag_model = BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=20),
        n_estimators=100,
        n_jobs=-1,
        random_state=42
    )
    bag_model.fit(X_train, y_train)
    y_pred_log = bag_model.predict(X_test)
    evaluate_model(y_pred_log, y_test, "Bagging Regressor")

# Gradient Boosting Regressor
if GradientBoostingRegressor:
    from sklearn.ensemble import GradientBoostingRegressor

    print("\nStarting Gradient Boosted Regressor...")
    gb_model = GradientBoostingRegressor(
        n_estimators=800,        # more trees
        learning_rate=0.03,      # slower learning = better generalization
        max_depth=5,             # slightly shallower trees
        min_samples_leaf=10,
        subsample=0.8,           # stochastic boosting (VERY important)
        max_features=0.8,        # adds randomness like RF
        random_state=42
    )

    gb_model.fit(X_train, y_train)
    y_pred_log = gb_model.predict(X_test)
    evaluate_model(y_pred_log, y_test, "Gradient Boosted Regressor")

if VotingRegressor:
    from sklearn.ensemble import VotingRegressor

    print("\nStarting Voting Regressor...")
    voting_model = VotingRegressor([
        ("rf", rf_model),
        ("gb", gb_model),
        ("lr", lr_model)
    ])
    voting_model.fit(X_train, y_train)
    y_pred_log = voting_model.predict(X_test)
    evaluate_model(y_pred_log, y_test, "Voting Ensemble")

if StackRegressor:
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge

    print("\nStarting Stacking Regressor...")
    stack_model = StackingRegressor(
        estimators=[
            ("rf", rf_model),
            ("gb", gb_model)
        ],
        final_estimator=Ridge()
    )

    stack_model.fit(X_train, y_train)
    y_pred_log = stack_model.predict(X_test)
    evaluate_model(y_pred_log, y_test, "Stacking Ensemble")

if XGBRegressor:
    from xgboost import XGBRegressor

    print("\nStarting XGB Regressor...")
    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,      # L1 regularization
        reg_lambda=1.0,     # L2 regularization
        n_jobs=-1,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)
    y_pred_log = xgb_model.predict(X_test)
    evaluate_model(y_pred_log, y_test, "XGB Regressor")

results_df.to_csv("predictions_table_dropped_quantiles.csv", index=False)
