from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

target = 'taxvaluedollarcnt'

# ===== Read Data ===== #
print("Reading 2016 data...")
df = pd.read_csv("zillow-data/properties_2016.csv", low_memory=False)
print("Data read complete")
shape = df.shape
print("Starting Shape:", shape)

def refine_columns(numeric_cols, object_cols, category_cols, remove_cols):
    numeric_cols = [col for col in numeric_cols if col not in remove_cols]
    object_cols = [col for col in object_cols if col not in remove_cols]
    category_cols = [col for col in category_cols if col not in remove_cols]
    return numeric_cols, object_cols, category_cols


df.drop_duplicates(subset=['parcelid'])
print("Dedupe Shape:", df.shape)

print("\nColumn Datatypes:\n", df[df.columns].dtypes)

# either has deck or doesn't (only 1 type of deck, so just use true/false)
df["has_deck"] = df["decktypeid"].notnull().astype(bool)
df = df.drop(columns=["decktypeid"])
# either has pool or doesn't (none have more than 1 pool, so just use true/false)
df['poolcnt'] = df['poolcnt'].fillna(0)

#df['fireplacecnt'] = df['fireplacecnt'].fillna(0)
#df['fireplaceflag'] = df['fireplaceflag'].fillna(0)
df['fireplaceflag'] = df['fireplaceflag'].astype(bool)

df['hashottuborspa'] = df['hashottuborspa'].fillna(0)
df['hashottuborspa'] = df['hashottuborspa'].astype(bool)


# indentify feature datatypes
numeric_cols = []
object_cols = []

from pandas.api.types import is_numeric_dtype, is_string_dtype

for col in df.columns:
    if is_string_dtype(df[col]):
        object_cols.append(col)
    elif is_numeric_dtype(df[col]):
        numeric_cols.append(col)

# identify categorical features
category_cols = [
    'airconditioningtypeid',
    'architecturalstyletypeid',
    'buildingclasstypeid',
    'fips',
    'heatingorsystemtypeid',
    'propertycountylandusecode',
    'propertylandusetypeid',
    'propertyzoningdesc',
    'rawcensustractandblock',
    'censustractandblock',
    'regionidcounty',
    'regionidcity',
    'regionidzip',
    'regionidneighborhood',
    'storytypeid',
    'typeconstructiontypeid',
]
#print(df[category_cols].dtypes)

# ====== Drop rows with missing target ======
df = df.dropna(subset=[target])

# ====== Drop high-null columns =====
threshold = 0.8
threshold_real = threshold * df.shape[0]

null_counts = df.isnull().sum()
drop_cols_series = null_counts[null_counts > threshold_real]
drop_cols = [col for col in drop_cols_series.index]
df = df.drop(columns=drop_cols)
print (f"1.) Drop high-null columns ({threshold * 100}%): \n {drop_cols}")
numeric_cols, object_cols, category_cols = refine_columns(
    numeric_cols, object_cols, category_cols, drop_cols
)

# ====== Drop trivial & constant features ======
trivial_cols = [
    'structuretaxvaluedollarcnt',
    'landtaxvaluedollarcnt',
    'taxamount',
    'assessmentyear'
]
df = df.drop(columns=trivial_cols)
print(f"2.) Drop trivial columns: \n{trivial_cols}")
numeric_cols, object_cols, category_cols = refine_columns(
    numeric_cols, object_cols, category_cols, trivial_cols
)

# ====== Impute nulls ======

print(f"\nBefore Imputation: \n", df.isnull().sum())

for col in df.columns:
    if col in numeric_cols and col not in category_cols:
        # Non categorical, numeric columns
        df[col] = df[col].fillna(df[col].median()) # Impute with median
    elif col in category_cols:
        # Categorical, numeric columns
        df[col] = df[col].fillna("missing") # Impute with mode
    elif col in object_cols:
        # Object columns
        df[col] = df[col].fillna(df[col].mode()[0]) # impute with mode

print(f"After Imputation: \n", df.isnull().sum())

# ====== Clip outliers using quantiles (WINSORIZATION) ======

# View data distribution
"""
plt.figure(figsize=(8,4))
plt.boxplot(df[target].sample(n=100_000, random_state=42), vert=False)
plt.title("Value Distribution")
"""

# count number of "outliers"
Q1 = df[target].quantile(0.25)
Q3 = df[target].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df[target] < lower) | (df[target] > upper)]
print(f"Number of Outliers: ",len(outliers))
#plt.show() # show graph after printing outlier count so they can be viewed at the same time

# Count number of properties in bottom and top 1 quantiles
lower = df[target].quantile(0.01)
upper = df[target].quantile(0.99)
outliers = df[(df[target] < lower) | (df[target] > upper)]
print("Dropped by quantile clip:",len(outliers))
df[target] = df[target].clip(lower, upper) # drop outliers to account for probable data mistakes


df["bath_bed_ratio"] = df["bathroomcnt"] / (df["bedroomcnt"] + 1)
df["lot_to_house_ratio"] = df["lotsizesquarefeet"] / (df["calculatedfinishedsquarefeet"] + 1)
df["total_rooms"] = df["bathroomcnt"] + df["bedroomcnt"]
df["sqft_per_room"] = df["calculatedfinishedsquarefeet"] / (df["total_rooms"] + 1)


# ====== Prep for encoding  & understand cardinality ====== #

# view cardinality of each category
print(f"\nCardinality of category columns: \n", df[category_cols].nunique())

# drop extremely high cardinality
extreme_cardinality = []
for col in category_cols:
    if df[col].nunique() > 10000:
        df = df.drop(columns=[col])
        extreme_cardinality.append(col)

print(f"Drop extreme cardinality:\n", extreme_cardinality)
numeric_cols, object_cols, category_cols = refine_columns(
    numeric_cols, object_cols, category_cols, extreme_cardinality
)

# ====== Encode low-cardinality categorical columns ======

# these columns use encoding strategies that don't risk data leakage
low_cardinality_threshold = 10
one_hot_columns = []

for col in category_cols:
    if col in df.columns and col not in object_cols:
        if df[col].nunique() <= low_cardinality_threshold:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
            one_hot_columns.append(col)

print(f"One-hot encoded: \n", one_hot_columns)
numeric_cols, object_cols, category_cols = refine_columns(
    numeric_cols, object_cols, category_cols, one_hot_columns
)
print(f"\n Current Columns: \n", df.columns)


# ====== Apply m-estimate encoding to remaining categories ====== #

from sklearn.model_selection import train_test_split

# Split dataset into features (X) and target (y)
# X contains all input variables, y is the value we want to predict
X = df.drop(columns=[target])
y = df[target]

# Split into training and testing sets
# Train set is used to learn encoding + model
# Test set is used only for final evaluation (prevents data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# M-estimate encoding (TRAIN FIT)
# -------------------------------
# Computes a smoothed target mean for each category
# Combines:
#   - category-specific mean
#   - global dataset mean (for smoothing rare categories)
def m_estimate_fit(X, y, col, m=50):
    global_mean = y.mean()  
    # overall average target value (baseline for smoothing)

    stats = X.groupby(col)[col].count()  
    # number of occurrences per category (frequency)

    means = X.join(y).groupby(col)[y.name].mean()  
    # average target value per category

    # m-estimate formula:
    # weighted balance between category mean and global mean
    enc = (means * stats + global_mean * m) / (stats + m)

    return enc, global_mean  
    # returns mapping (category → encoded value) + global reference mean


# -------------------------------
# TRANSFORM FUNCTION
# -------------------------------
# Applies learned encoding to dataset
# unseen categories → replaced with global mean
def m_estimate_transform(X, col, mapping, global_mean):
    return X[col].map(mapping).fillna(global_mean)


# -------------------------------
# SELECT HIGH-CARDINALITY COLUMNS
# -------------------------------

# recompute from training only
high_card_cols = [
    col for col in X_train.columns
    if col in category_cols and X_train[col].nunique() > low_cardinality_threshold
]
print("High Cardinality Columns:\n", high_card_cols)

for col in high_card_cols:

    # fit on TRAIN ONLY
    mapping, global_mean = m_estimate_fit(X_train, y_train, col, m=10)

    # transform train
    X_train[col] = m_estimate_transform(X_train, col, mapping, global_mean)

    # transform test
    X_test[col] = m_estimate_transform(X_test, col, mapping, global_mean)

# All columns should be numeric/bool now
print(f"\nData Types: \n",X_train[X_train.columns].dtypes)

null_counts = df.isnull().sum()
print(f"\nColumns with nulls:\n", null_counts[null_counts > 0])

# ====== Run tree models on already split data ====== #


def evaluate_model (y_test, y_pred, model = "Model"):
    print(f"--= {model} Eval. =--")
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("MAE: ", mae)
    print("RMSE: ", rmse)
    print("R2: ", r2)

    """
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.3)

    # perfect prediction line
    plt.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--')

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model} - Actual vs Predicted")
    plt.show()
    plt.close()

    mask = (y_test > 0) & (y_pred > 0)
    plt.figure(figsize=(6,6))
    plt.scatter(y_test[mask], y_pred[mask], alpha=0.3)

    plt.plot(
        [y_test[mask].min(), y_test[mask].max()],
        [y_test[mask].min(), y_test[mask].max()],
        'r--'
    )

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model} - Actual vs Predicted (Log Scale)")
    plt.show()
    plt.close()
    """


#X_train = X_train.sample(n=100_000, random_state=42)
#_train = y_train.loc[X_train.index]


# ====== LINEAR REGRESSOR ====== #
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
    
print("\nStart Linear Regressor")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
evaluate_model(y_test,y_pred,"LR")


# ====== RF REGRESSOR ====== #
from sklearn.ensemble import RandomForestRegressor
print("\nStart Random Forest")
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
"""
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=30,
    min_samples_split=10,
    min_samples_leaf=7,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)
"""
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
evaluate_model(y_test,y_pred,"RF")

# ====== ET REGRESSOR ====== #
from sklearn.ensemble import ExtraTreesRegressor
print("\nStart Extra Trees")
et = ExtraTreesRegressor(random_state=42, n_jobs=-1)
"""
et = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=30,
    min_samples_leaf=7,
    n_jobs=-1,
    random_state=42
)
"""
et.fit(X_train, y_train)
y_pred = et.predict(X_test)
evaluate_model(y_test,y_pred,"ET")

# ====== GBR REGRESSOR ====== #
from sklearn.ensemble import GradientBoostingRegressor

print("\nStart Gradient Boosted Regressor")
gb = GradientBoostingRegressor(random_state=42)
"""
gb = GradientBoostingRegressor(
    n_estimators=800,        # more trees
    learning_rate=0.03,      # slower learning = better generalization
    max_depth=5,             # slightly shallower trees
    min_samples_leaf=10,
    subsample=0.8,           # stochastic boosting (VERY important)
    max_features=0.8,        # adds randomness like RF
    random_state=42
)
"""
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
evaluate_model(y_test,y_pred,"GBR")



# ====== XGB REGRESSOR ====== #
print("\nStart XGB")
xgb = XGBRegressor(random_state=42, n_jobs=-1)
"""
xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=13,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,      # L1 regularization
    reg_lambda=1.0,     # L2 regularization
    n_jobs=-1,
    random_state=42
)
"""
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
evaluate_model(y_test,y_pred,"XGB")


# ====== CATBOOST REGRESSOR ====== #
from catboost import CatBoostRegressor
print("\nStart Catboost")
cat = CatBoostRegressor(
    random_seed=42,
    verbose=0
)
"""
cat = CatBoostRegressor(
    iterations=800,
    learning_rate=0.04,
    depth=13,
    loss_function="RMSE",
    random_seed=42,
    verbose=100
)
"""
cat.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    use_best_model=True
)

y_pred = cat.predict(X_test)
evaluate_model(y_test,y_pred,"CAT")