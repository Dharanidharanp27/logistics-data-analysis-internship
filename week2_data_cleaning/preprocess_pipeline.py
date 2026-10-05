"""Week 2 - Data collection, cleaning and preprocessing pipeline for logistics data.

Run:  python preprocess_pipeline.py
Output: logistics_clean.csv
"""

# ---- Step 1: Load and inspect ----
import pandas as pd
import numpy as np

# Use the real DataCo file if it is placed next to this script, else a simulated one
import os
if os.path.exists("DataCoSupplyChainDataset.csv"):
    df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding="latin-1")
else:
    from simulate_raw_data import make_data
    df = make_data()
    print("DataCo file not found - using simulated data")

print(df.shape)
df.info()

quality = pd.DataFrame({
    "missing": df.isna().sum(),
    "missing_%": df.isna().mean().mul(100).round(1),
    "unique": df.nunique()})
print(quality.sort_values("missing_%", ascending=False).head(10))

print(df.describe())                      # ranges reveal impossible values
print("exact duplicates:", df.duplicated().sum())


# ---- Step 2: Standardise names, text, duplicates ----
# 1) Consistent column names
df.columns = (df.columns.str.strip().str.lower()
                .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))
df = df.rename(columns={
    "order_date_dateorders": "order_date",
    "shipping_date_dateorders": "shipping_date",
    "days_for_shipping_real": "days_real",
    "days_for_shipment_scheduled": "days_scheduled",
    "order_item_quantity": "quantity",
    "order_profit_per_order": "profit",
    "order_region": "region",
    "customer_zipcode": "customer_zip"})

# 2) Clean text and fix inconsistent categories
for col in df.select_dtypes("object").columns:
    df[col] = df[col].str.strip()
df["shipping_mode"] = df["shipping_mode"].str.title()

# 3) Remove exact duplicate rows
df = df.drop_duplicates().reset_index(drop=True)


# ---- Step 3: Fix data types and logical errors ----
# Convert text to real dates; unparseable values become NaT
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df["shipping_date"] = pd.to_datetime(df["shipping_date"], errors="coerce")
df = df.dropna(subset=["order_date", "shipping_date"])

# Logical check: an order cannot ship before it was placed
bad_dates = df["shipping_date"] < df["order_date"]
print("shipping before order:", bad_dates.sum())
df = df[~bad_dates].copy()

# Impossible values -> NaN, so they are imputed rather than silently kept
for col in ["quantity", "sales"]:
    df.loc[df[col] <= 0, col] = np.nan
df.loc[df["days_real"] < 0, "days_real"] = np.nan


# ---- Step 4: Handle missing values ----
# Drop columns that are almost entirely empty (no information to recover)
missing_pct = df.isna().mean()
drop_cols = missing_pct[missing_pct > 0.60].index.tolist()
df = df.drop(columns=drop_cols)

# Keep a flag where "missing" itself may carry information
df["sales_was_missing"] = df["sales"].isna().astype(int)

# Numeric: median of the same shipping mode, then overall median as fallback
num_cols = ["quantity", "sales", "profit", "days_real", "days_scheduled"]
for col in num_cols:
    df[col] = df[col].fillna(df.groupby("shipping_mode")[col].transform("median"))
    df[col] = df[col].fillna(df[col].median())

# Categorical: most frequent value (mode)
for col in ["region", "delivery_status"]:
    df[col] = df[col].fillna(df[col].mode()[0])

# Identifier column: never impute a zip code, label it instead
df["customer_zip"] = df["customer_zip"].astype("Int64").astype("string").fillna("Unknown")


# ---- Step 5: Detect and treat outliers ----
def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

check_cols = ["sales", "profit", "days_real"]
outliers = {}
for col in check_cols:
    lo, hi = iqr_bounds(df[col])
    outliers[col] = int(((df[col] < lo) | (df[col] > hi)).sum())
print("IQR outliers:", outliers)

# Cross-check with z-score
z = (df["sales"] - df["sales"].mean()) / df["sales"].std()
print("z-score > 3:", int((z.abs() > 3).sum()))

# Treatment: winsorize (cap) at the 1st and 99th percentile
for col in check_cols:
    lo, hi = df[col].quantile([0.01, 0.99])
    df[col] = df[col].clip(lo, hi)


# ---- Step 6: Feature engineering, encoding, scaling ----
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Derived features
df["delay_days"] = df["days_real"] - df["days_scheduled"]
df["is_late"] = (df["delay_days"] > 0).astype(int)      # target for later modelling
df["order_month"] = df["order_date"].dt.month
df["sales_log"] = np.log1p(df["sales"])                  # reduces right skew

# Encode categories
df = pd.get_dummies(df, columns=["shipping_mode"], prefix="mode", dtype=int)
mode_cols = [c for c in df.columns if c.startswith("mode_")]

scale_cols = ["quantity", "sales_log", "days_scheduled"]
X = df[scale_cols + mode_cols]
y = df["is_late"]

# Split FIRST, then fit scalers on training data only (avoids data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler().fit(X_train[scale_cols])       # z-score: mean 0, std 1
X_train_s, X_test_s = X_train.copy(), X_test.copy()
X_train_s[scale_cols] = scaler.transform(X_train[scale_cols])
X_test_s[scale_cols] = scaler.transform(X_test[scale_cols])

# Alternative: Min-Max scaling to the 0-1 range
mm = MinMaxScaler().fit(X_train[scale_cols])
X_train_mm = mm.transform(X_train[scale_cols])


# ---- Step 7: Validate and save ----
# Automated sanity checks - the pipeline fails loudly if data is still dirty
assert df.isna().sum().sum() == 0, "missing values remain"
assert (df["quantity"] > 0).all() and (df["sales"] > 0).all()
assert (df["shipping_date"] >= df["order_date"]).all()
assert not df.duplicated().any()

df.to_csv("logistics_clean.csv", index=False)
print("Clean dataset:", df.shape)
