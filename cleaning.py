import pandas as pd

# ===== Read Data ===== #
print("Reading data...")
df = pd.read_csv("zillow-data/properties_2016.csv", low_memory=False)
print("Data read complete")
shape = df.shape
# ===================== #


# ===== High NULL features ===== #
threshold = 0.7
num_rows = df.shape[0] # Get number rows 
null_counts = df.isnull().sum() # Series of 'Column: # NULLS'
drop_cols_series = null_counts[df.isnull().sum() > num_rows * threshold] # Series of 'Column: # NULLs' where # NULLs > 70% of total rows
#print("Drop Features: ", drop_cols_series) # Display selected columns

drop_cols = drop_cols_series.index  # Get index of the series to obtain just column names
df_clean = df.drop(columns=drop_cols)  # Drop columns with >70% nulls
print("Drop features w/ >70% NULL")
print(shape, " -> ", df_clean.shape)
shape = df_clean.shape
# =============================== #


# ===== Parcelid Handling ===== #
parcel_ids = df_clean['parcelid']  # Keep for tracking predictions
df_clean = df_clean.drop(columns=["parcelid"])  # Remove from features
print("Drop parcelids")
print(shape, " -> ", df_clean.shape)
shape = df_clean.shape
# ============================= #


# ===== Trivial & Useless Features ===== #
df_clean = df_clean.drop(columns=[
    "structuretaxvaluedollarcnt",   # Value of the structure
    "landtaxvaluedollarcnt",        # Value of the land
    "taxamount",                    # Property tax amount
    "propertyzoningdesc",           # Correlates to properzoningid
])
print("Drop useless & trivial features")
print(shape, " -> ", df_clean.shape)
shape = df_clean.shape
# ======================================= #

# ===== High-cardinality Handling ===== #
df_clean = df_clean.drop(columns=[
    "censustractandblock",
    "rawcensustractandblock",
    "regionidneighborhood",
    "regionidzip",
    "regionidcity"
])
print("Drop high-cardinality features")
print(shape, " -> ", df_clean.shape)
shape = df_clean.shape
# ===================================== #


# ===== Scale Longitude & Latitude ===== #
# Scale/normalize long&lat to maintain potential correlation to target but reduce high-cardinality
from sklearn.preprocessing import MinMaxScaler
# Select latitude and longitude columns
coords = df_clean[['latitude', 'longitude']]
# Initialize scaler with range [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
# Fit scaler and transform coordinates
coords_scaled = scaler.fit_transform(coords)
# Replace original columns with scaled values
df_clean['latitude'] = coords_scaled[:, 0]
df_clean['longitude'] = coords_scaled[:, 1]
print("Scale longitude & latitude")
# ====================================== #


# ===== Imputations & Other Misc Drops ===== #
df_clean = df_clean.copy() # ensure independent DataFrame so edits persist properly
# Drop rows with missing taxvaluedollarcnt; Missing target value is useless, obviously
df_clean.dropna(subset = ['taxvaluedollarcnt'],inplace = True)
print("Drop rows w/o target")
print(shape, " -> ", df_clean.shape)
shape = df_clean.shape

# bedroomcnt, bathroomcnt, calculatedbathnbr, etc. numeric columns can be set as median to avoid biasing mean and still reflect typical property
df_clean['bathroomcnt'] = df_clean['bathroomcnt'].fillna(df_clean['bathroomcnt'].median())
df_clean['bedroomcnt'] = df_clean['bedroomcnt'].fillna(df_clean['bedroomcnt'].median())
df_clean['calculatedbathnbr'] = df_clean['calculatedbathnbr'].fillna(df_clean['calculatedbathnbr'].median())
df_clean['calculatedfinishedsquarefeet'] = df_clean['calculatedfinishedsquarefeet'].fillna(df_clean['calculatedfinishedsquarefeet'].median())
df_clean['finishedsquarefeet12'] = df_clean['finishedsquarefeet12'].fillna(df_clean['finishedsquarefeet12'].median())
df_clean['fullbathcnt'] = df_clean['fullbathcnt'].fillna(df_clean['fullbathcnt'].median())
df_clean['roomcnt'] = df_clean['roomcnt'].fillna(df_clean['roomcnt'].median())
df_clean['lotsizesquarefeet'] = df_clean['lotsizesquarefeet'].fillna(df_clean['lotsizesquarefeet'].median())
df_clean['buildingqualitytypeid'] = df_clean['buildingqualitytypeid'].fillna(df_clean['buildingqualitytypeid'].median())
print("Impute rows with NULL")

# Non-numeric & 200+ unique values; correlates to propertylandusetypeid
df_clean = df_clean.drop(columns=['propertycountylandusecode'])
print("Drop 'propertycountylandusecode'")
print(shape, " -> ", df_clean.shape)
shape = df_clean.shape
# ========================================= #


# ===== Unitcnt Handling ===== #
# Assume structures consist of 1 unit if NULL
df_clean['unitcnt'] = df_clean['unitcnt'].fillna(1)
#outliers = df_clean[df_clean['unitcnt'] > 10]
#print(outliers['unitcnt'].head(20))
# Some values in unitcnt are unreasonably high (100+; likely error?) so clipped to max of 10
df_clean['unitcnt'] = df_clean['unitcnt'].clip(upper=10)
# NOTE: Clip could be adjusted later, 10 is arbitrary
print("Impute NULL unitcnt with 1, clip max to 10")
# ============================= #


# ===== One-hot encode heatingsystem ===== #
# Fill heatingsystem NULLs with (new) category 0
df_clean['heatingorsystemtypeid'] = df_clean['heatingorsystemtypeid'].fillna(0)
print("Impute NULL 'heatingorsystemtypeid'")

df_clean = pd.get_dummies(
    df_clean,
    columns=['heatingorsystemtypeid'],
    prefix='heating',
    drop_first=True, # Could change to false for any model that isn't linear regression, but will still work with any other model
    dtype=int
)
print("One-hot encode 'heatingorsystemtypeid'")
print(shape, " -> ", df_clean.shape)
shape = df_clean.shape
# ========================================= #


# ===== One-hot encode regionidcounty ===== #
# Fill heatingsystem NULLs with (new) category 0
df_clean['regionidcounty'] = df_clean['regionidcounty'].fillna(0)
print("Impute NULL 'regionidcounty'")

df_clean = pd.get_dummies(
    df_clean,
    columns=['regionidcounty'],
    prefix='regionid',
    drop_first=True, # Could change to false for any model that isn't linear regression, but will still work with any other model
    dtype=int
)
print("One-hot encode 'regionidcounty'")
print(shape, " -> ", df_clean.shape)
shape = df_clean.shape
# ========================================= #


# ===== One-hot encode propertylanduse ===== #
df_clean = pd.get_dummies(
    df_clean, 
    columns=['propertylandusetypeid'], 
    prefix='plti',
    drop_first=True,
    dtype=int
)
print("One-hot encode 'propertylandusetypeid'")
print(shape, " -> ", df_clean.shape)
shape = df_clean.shape
# ========================================= #


# ===== One-hot encode fips ===== #
df_clean = pd.get_dummies(
    df_clean, 
    columns=['fips'], 
    prefix='fips',
    drop_first=True,
    dtype=int
)
print("One-hot encode 'fips'")
print(shape, " -> ", df_clean.shape)
shape = df_clean.shape
# ================================= #


# ===== Flip buildingquality ===== #
# FLip so that lower = worse and higher = better
max_quality = df_clean['buildingqualitytypeid'].max()
df_clean['buildingqualitytypeid'] = max_quality - df_clean['buildingqualitytypeid'] + 1
print("Flip 'buildingqualitytypeid'")
# ================================== #


# ===== Property Age ===== #
df_clean = df_clean.copy()
df_clean.dropna(subset = ['yearbuilt'],inplace = True) # drop rows with NULL yearbuilt
#df_clean['yearbuilt'] = df_clean['yearbuilt'].fillna(df_clean['yearbuilt'].median()) # Set yearbuilt to median if NULL
df_clean['assessmentyear'] = df_clean['assessmentyear'].fillna(2016) # None with NULL, but just in case set year to year of dataset
df_clean['property_age'] = df_clean['assessmentyear'] - df_clean['yearbuilt']
df_clean['property_age'] = df_clean['property_age'].clip(lower=0) # None go negative, but just in case set lower bound to 0
df_clean = df_clean.drop(columns=[
    "assessmentyear",
    "yearbuilt"
]) # Drop because not needed anymore
print("Add property_age, drop yearbuilt & assessmentyear")
print(shape, " -> ", df_clean.shape)
shape = df_clean.shape
# ============================= #


print("\n--=== Cleaning Complete ===--")
print(f"Shape: {df_clean.shape}")
print(f"Columns: {df_clean.columns}")

df_clean.to_csv("zillow-data/properties_2016_cleaned.csv", index=False)
print("Cleaned dataset saved to zillow-data/properties_2016_cleaned.csv")
