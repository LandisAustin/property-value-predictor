import pandas as pd

df = pd.read_csv("zillow-data/properties_2016.csv", low_memory=False)

# 1.)
#print(df.shape) # (rows, columns)
original_shape = df.shape


# 2.)
"""
print("Columns:\n")
print(df.columns) # All features/columns
print("Types:\n")
print(df.dtypes) # Data types of each column
"""

# 3.)
#print(df.isnull().sum()) # Get number NULLs per row

threshold = 0.7
num_rows = df.shape[0] # Get number rows 
null_counts = df.isnull().sum() # Series of 'Column: # NULLS'
drop_cols_series = null_counts[df.isnull().sum() > num_rows * threshold] # Series of 'Column: # NULLs' where # NULLs > 70% of total rows
#print("Drop Features: ", drop_cols_series) # Display selected columns

# 4.) Drop irrelevant or problematic columns
drop_cols = drop_cols_series.index  # Get index of the series to obtain just column names
df_clean = df.drop(columns=drop_cols)  # Drop columns with >70% nulls

# Drop columns that make the target trivial or are useless for prediction
df_clean = df_clean.drop(columns=[
    "structuretaxvaluedollarcnt",  # Value of the structure
    "landtaxvaluedollarcnt",        # Value of the land
    "taxamount",                    # Property tax amount
    "assessmentyear"                # Not predictive
])

# Save parcelids for later evaluation
parcel_ids = df_clean['parcelid']  # Keep for tracking predictions
df_clean = df_clean.drop(columns=["parcelid"])  # Remove from features

# Select potential categorical columns
categorical_cols = ['regionidzip', 'regionidcity', 'propertylandusetypeid', 'regionidneighborhood', 'censustractandblock']
# Check # of unique values to see if one-hot encoding is feasible
#for col in categorical_cols:
#    print(f"{col}: {df_clean[col].nunique()} unique values")

# NOTE: Reference when considering one-hot encoding
"""
regionidzip: 405 unique values           <---- Unfeasible; Too large, maybe other encoding later
regionidcity: 186 unique values          <---- Maybe Feasible; maybe group most commons and lump uncommon as 'Other'
propertylandusetypeid: 15 unique values  <---- Most feasible
regionidneighborhood: 528 unique values  <---- Unfeasible; Too large, maybe other encoding later
censustractandblock: 96771 unique values <---- Unfeasible; WAY too large, drop or other encoding later
"""

# Specifically for linear regression:
# Drop high-cardinality identifiers to avoid spurious numeric effects
df_clean = df_clean.drop(columns=[
    "censustractandblock",
    "rawcensustractandblock",
    "regionidneighborhood",
    "regionidzip",
    "regionidcity"
])


# Very sparse (963k) so just drop because hard to create a default
# NOTE: Maybe set a new 'missing' flag/feature for this column?
df_clean = df_clean.drop(columns=['propertyzoningdesc'])

print(original_shape, " -> ", df_clean.shape)  # Show original shape vs new shape
clean_shape = df_clean.shape

null_counts = df_clean.isnull().sum()
#print("Null Counts: \n",null_counts[null_counts > 0]) # View remaining null counts for each column
# Some features like bathroomcnt and bedroomcnt have nulls. Find default value for these (0?)
# Still some large null counts for features like buildingqualitytypeid and regionidneighborhood

#print ("General Description: \n", df_clean.describe()) # Show count, mean, std, min, max of columns

# Histograms -->
import matplotlib.pyplot as plt

numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns # Select columns with numeric data types

df_clean[numeric_cols].hist(figsize=(15,10), bins=50) # Draw histogram for these numeric columns 
#plt.show()
# <-- End of Histograms


numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns

# Check for low variance columns
"""
numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
cv = df_clean[numeric_cols].std() / df_clean[numeric_cols].mean()
low_variance_features = cv[cv < 0.05]  # features that vary less than 5% of their mean
print("Low-variance features (relative to mean):\n", low_variance_features)
"""

# Check for low percent of unique values
"""
num_rows = df_clean.shape[0]
low_unique_ratio_features = []
for col in numeric_cols:
    unique_ratio = df_clean[col].nunique() / num_rows
    if unique_ratio < 0.001:  # less than 0.1% unique
        low_unique_ratio_features.append(col)
print("Features with very few unique values:\n", low_unique_ratio_features)
"""
# The features that do fit into these filters turn out to be important so it doesn't really help drop any more columns

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
# check the min/max and a few sample values
#print(df_clean[['latitude', 'longitude']].describe())
#print(df_clean[['latitude', 'longitude']].head())

# Go back to NULL list to potentially do some more cleaning/defaulting
#null_counts = df_clean.isnull().sum()
#print("\nNull Counts: \n",null_counts[null_counts > 0])

df_clean = df_clean.copy() # ensure independent DataFrame so edits persist properly
# Drop rows with missing taxvaluedollarcnt; Missing target value is useless, obviously
df_clean.dropna(subset = ['taxvaluedollarcnt'],inplace = True)
# bedroomcnt, bathroomcnt, calculatedbathnbr, etc. numeric columns can be set as median to avoid biasing mean and still reflect typical property
df_clean['bathroomcnt'] = df_clean['bathroomcnt'].fillna(df_clean['bathroomcnt'].median())
df_clean['bedroomcnt'] = df_clean['bedroomcnt'].fillna(df_clean['bedroomcnt'].median())
df_clean['calculatedbathnbr'] = df_clean['calculatedbathnbr'].fillna(df_clean['calculatedbathnbr'].median())
df_clean['calculatedfinishedsquarefeet'] = df_clean['calculatedfinishedsquarefeet'].fillna(df_clean['calculatedfinishedsquarefeet'].median())
df_clean['finishedsquarefeet12'] = df_clean['finishedsquarefeet12'].fillna(df_clean['finishedsquarefeet12'].median())
df_clean['fullbathcnt'] = df_clean['fullbathcnt'].fillna(df_clean['fullbathcnt'].median())
df_clean['roomcnt'] = df_clean['roomcnt'].fillna(df_clean['roomcnt'].median())
df_clean['yearbuilt'] = df_clean['yearbuilt'].fillna(df_clean['yearbuilt'].median())
df_clean['lotsizesquarefeet'] = df_clean['lotsizesquarefeet'].fillna(df_clean['lotsizesquarefeet'].median())
df_clean['buildingqualitytypeid'] = df_clean['buildingqualitytypeid'].fillna(df_clean['buildingqualitytypeid'].median())
# Only 783 rows with NULLs in this column at this point, so safe to just drop
df_clean.dropna(subset = ['propertycountylandusecode'],inplace = True)

# Assume structures consist of 1 unit if NULL
df_clean['unitcnt'] = df_clean['unitcnt'].fillna(1)
#outliers = df_clean[df_clean['unitcnt'] > 10]
#print(outliers['unitcnt'].head(20))
# Some values in unitcnt are unreasonably high (100+; likely error?) so clipped to max of 10
df_clean['unitcnt'] = df_clean['unitcnt'].clip(upper=10)
# NOTE: Clip could be adjusted later, 10 is arbitrary

# HEATING SYSTEM STUFF
#print(df_clean['heatingorsystemtypeid'].describe())
# How many unique values? One-hot feasible? YES (14 unique values)
#print(df_clean['heatingorsystemtypeid'].nunique())
#print(df_clean['heatingorsystemtypeid'].value_counts())

# Fill heatingsystem NULLs with (new) category 0
# This is better than imputing with potentially incorrect information
df_clean['heatingorsystemtypeid'] = df_clean['heatingorsystemtypeid'].fillna(0)

# One-hot encode new category columns (ex: heating_2)
# drop_firt drops one of these columns to act as a 'baseline'
# This is only important for linear regression because otherwise there is multicollinearity where each column is dependent on the others
# This freaks linear regression out due to its linear algebra and solving matrix inverses
df_clean = pd.get_dummies(
    df_clean,
    columns=['heatingorsystemtypeid'],
    prefix='heating',
    drop_first=True, # Could change to false for any model that isn't linear regression, but will still work with any other model
    dtype=int
)
# NOTE: Maybe come back to this, has > 1.1mil NULLs that are being set

print(clean_shape, " -> ", df_clean.shape)
clean_shape = df_clean.shape

null_counts = df_clean.isnull().sum()
print("\nNull Counts: \n",null_counts[null_counts > 0])

print(df_clean.columns)

# FINAL NOTES TL;DR
# NOTE: ALL NULLS SHOULD BE ACCOUNTED FOR WITH CURRENT CLEANING
# NOTE: Consider creating new 'missing_<feature>' flag/column for potentially useful but high NULLed features (ie. propertyzoningdesc)
    # The absence of a feature could be informative itself

# NOTE: Consider one-hot encoding for a couple features that are currently dropped:
#   regionidzip: 405 unique values           <---- Unfeasible; Too large, maybe other encoding later
#   regionidcity: 186 unique values          <---- Maybe Feasible; maybe group most commons and lump uncommon as 'Other'
#   propertylandusetypeid: 15 unique values  <---- Most feasible
#   regionidneighborhood: 528 unique values  <---- Unfeasible; Too large, maybe other encoding later
#   censustractandblock: 96771 unique values <---- Unfeasible; WAY too large, drop or other encoding later
#
#   Also consider missing flag for NULLs like previously mentioned
#   NOTE: IF ENCODED & NOT DROPPED... remember to take care of NULLs

# NOTE: Check descriptions of non-dropped features to ensure outliers and potential errors are managed
# NOTE: Unitcnt is clipped to 10, but that is arbitrary and could be adjusted later

# This feature is reversed in the dictionary: lower = better, higher = worse
# Flip these values since it's directly opposite of how linear regression treats numeric values
# The model could learn this negative slope on its own, or we could help it out and flip them
max_quality = df_clean['buildingqualitytypeid'].max()
df_clean['buildingqualitytypeid'] = max_quality - df_clean['buildingqualitytypeid'] + 1

#df_clean.to_csv("zillow-data/properties_2016_cleaned.csv", index=False)
#print("Cleaned dataset saved to zillow-data/properties_2016_cleaned.csv")
