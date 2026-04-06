import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np

# ===== Read Data ===== #
print("Reading 2016 data...")
df1 = pd.read_csv("zillow-data/properties_2016.csv", low_memory=False)
print("Reading 2017 data...")
df2 = pd.read_csv("zillow-data/properties_2016.csv", low_memory=False)
df = pd.concat([df1, df2], ignore_index=True)
print("Data read complete")
shape = df.shape
print("Starting Shape:", shape)

# ===== High NULL Features ===== #
important_high_null = [
    'garagecarcnt', 'garagetotalsqft',
    'poolcnt', 'poolsizesum',
    'fireplacecnt', 'basementsqft',
    'hashottuborspa', 'threequarterbathnbr'
]

threshold = 0.8
num_rows = df.shape[0]
null_counts = df.isnull().sum()
drop_cols_series = null_counts[null_counts > num_rows * threshold]
drop_cols = [col for col in drop_cols_series.index if col not in important_high_null]
df_clean = df.drop(columns=drop_cols)
print(f"\nDropped features w/ NULL > {threshold*100}% (except important): {drop_cols}")
print("Shape:", shape, "->", df_clean.shape)
shape = df_clean.shape

# ===== Impute High-NULL Important Features ===== #
for col in important_high_null:
    df_clean[f'{col}_missing'] = df_clean[col].isnull().astype(int)
    # Convert to numeric and fill NaN with 0
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

# ===== Parcelid Handling ===== #
parcel_ids = df_clean['parcelid']  # Keep for tracking predictions
df_clean = df_clean.drop(columns=['parcelid'])
print("Dropped parcelid")
print("Shape:", shape, "->", df_clean.shape)
shape = df_clean.shape

# ===== Trivial & Useless Features ===== #
df_clean = df_clean.drop(columns=[
    "structuretaxvaluedollarcnt",
    "landtaxvaluedollarcnt",
    "taxamount",
    "propertyzoningdesc"
])
print("Dropped trivial features")
print("Shape:", shape, "->", df_clean.shape)
shape = df_clean.shape

# ===== High-Cardinality Categorical Features ===== #
df_clean = df_clean.drop(columns=[
    "censustractandblock",
    "rawcensustractandblock",
    "regionidneighborhood",
    "regionidzip",
    "regionidcity"
])
print("Dropped high-cardinality categorical features")
print("Shape:", shape, "->", df_clean.shape)
shape = df_clean.shape

# ===== Scale Longitude & Latitude ===== #
coords = df_clean[['latitude', 'longitude']]
scaler = MinMaxScaler(feature_range=(-1, 1))
coords_scaled = scaler.fit_transform(coords)
df_clean['latitude'] = coords_scaled[:, 0]
df_clean['longitude'] = coords_scaled[:, 1]
print("Scaled latitude & longitude")

# ===== Drop Rows with Missing Target ===== #
df_clean.dropna(subset=['taxvaluedollarcnt'], inplace=True)
print("Dropped rows w/o target")
print("Shape:", shape, "->", df_clean.shape)
shape = df_clean.shape

# ===== Impute Other Numeric Features with Median ===== #
numeric_median_cols = [
    'bathroomcnt', 'bedroomcnt', 'calculatedbathnbr',
    'calculatedfinishedsquarefeet', 'finishedsquarefeet12',
    'fullbathcnt', 'roomcnt', 'lotsizesquarefeet'
]
for col in numeric_median_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# ===== Drop Non-Numeric / Redundant Features ===== #
df_clean = df_clean.drop(columns=['propertycountylandusecode'])

# ===== Unitcnt Handling ===== #
df_clean['unitcnt'] = pd.to_numeric(df_clean['unitcnt'], errors='coerce').fillna(1)
df_clean['unitcnt'] = df_clean['unitcnt'].clip(upper=10)

# ===== One-Hot Encode Categorical Features ===== #
categorical_cols = [
    'airconditioningtypeid', 'heatingorsystemtypeid',
    'regionidcounty', 'propertylandusetypeid', 'fips'
]
for col in categorical_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    df_clean = pd.get_dummies(df_clean, columns=[col], prefix=col, drop_first=True, dtype=int)
print("One-Hot Encode")
print("Shape:", shape, "->", df_clean.shape)
shape = df_clean.shape

# ===== Flip buildingquality ===== #
df_clean['buildingquality_missing'] = df_clean['buildingqualitytypeid'].isnull().astype(int)
df_clean['buildingqualitytypeid'] = pd.to_numeric(df_clean['buildingqualitytypeid'], errors='coerce')
df_clean['buildingqualitytypeid'] = df_clean['buildingqualitytypeid'].fillna(df_clean['buildingqualitytypeid'].median())
max_quality = df_clean['buildingqualitytypeid'].max()
df_clean['buildingqualitytypeid'] = max_quality - df_clean['buildingqualitytypeid'] + 1

# ===== Property Age ===== #
df_clean.dropna(subset=['yearbuilt'], inplace=True)
df_clean['assessmentyear'] = pd.to_numeric(df_clean['assessmentyear'], errors='coerce').fillna(2016)
df_clean['property_age'] = df_clean['assessmentyear'] - df_clean['yearbuilt']
df_clean['property_age'] = df_clean['property_age'].clip(lower=0)
df_clean = df_clean.drop(columns=['assessmentyear', 'yearbuilt'])

# ===== Remove Outliers ===== # 
p = 0.07
lower = df_clean['taxvaluedollarcnt'].quantile(p)       # 8th percentile
upper = df_clean['taxvaluedollarcnt'].quantile(1 - p)   # 92nd percentile

df_trimmed = df_clean[(df_clean['taxvaluedollarcnt'] >= lower) &
                      (df_clean['taxvaluedollarcnt'] <= upper)]

df_clean = df_clean[
    (df_clean['taxvaluedollarcnt'] >= lower) &
    (df_clean['taxvaluedollarcnt'] <= upper)
]
print("Remove Outliers")
print("Shape:", shape, "->", df_clean.shape)
shape = df_clean.shape

# ===== Force All Columns Numeric & Fill Any Remaining NaN ===== #
for col in df_clean.columns:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean = df_clean.fillna(0)


df_clean['sqft_per_room'] = df_clean['calculatedfinishedsquarefeet'] / df_clean['roomcnt']
df_clean['lot_to_living_ratio'] = df_clean['lotsizesquarefeet'] / df_clean['calculatedfinishedsquarefeet']
df_clean['bath_bed_ratio'] = df_clean['bathroomcnt'] / df_clean['bedroomcnt']
df_clean['age_per_quality'] = df_clean['property_age'] / df_clean['buildingqualitytypeid']


for col in ['sqft_per_room', 'lot_to_living_ratio', 'bath_bed_ratio', 'age_per_quality']:
    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    df_clean[col] = df_clean[col].fillna(0)

# ----- Scale ----- #
scaler = StandardScaler()
numeric_cols = [
    'basementsqft', 'bathroomcnt', 'bedroomcnt', 'buildingqualitytypeid',
    'calculatedbathnbr', 'calculatedfinishedsquarefeet',
    'finishedsquarefeet12', 'fireplacecnt', 'fullbathcnt',
    'garagecarcnt', 'garagetotalsqft',
    'lotsizesquarefeet', 'poolsizesum', 'roomcnt',
    'threequarterbathnbr', 'unitcnt', 'numberofstories',
    'property_age',
    'sqft_per_room', 'lot_to_living_ratio', 'bath_bed_ratio'
]

# ===== Final Checks ===== #
print("\n--=== Cleaning Complete ===--")
print(f"Shape: {df_clean.shape}")
print("Total NaNs:", df_clean.isnull().sum().sum())

# ===== Save Cleaned Data ===== #
df_clean.to_csv("zillow-data/properties_2016_cleaned.csv", index=False)
print("Cleaned dataset saved!")