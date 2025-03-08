import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
from google.cloud import bigquery
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor, DMatrix, cv
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import catboost



# 📌 Set Up BigQuery Connection
PROJECT_ID = "travel-insider-452211"
DATASET_NAME = "travel_insider_dataset"
TABLE_NAME = "filtered_flights"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/sebastian/code/JPYY-96/travel_insider/raw_data/travel-insider-452211-181bd2eba48e.json"

# 📌 Initialize BigQuery Client
client = bigquery.Client()
query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_NAME}.{TABLE_NAME}` LIMIT 1000000"

# 📌 Load Data in Chunks with Timer
chunk_size = 1000000
data_chunks = []
start_time = time.time()

print(f"⏳ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading data from BigQuery in chunks...")
query_job = client.query(query)
result_iter = query_job.result(page_size=chunk_size)

# 📌 Process Each Chunk into a DataFrame
for page in result_iter.pages:
    chunk_data = [dict(row) for row in page]
    df_chunk = pd.DataFrame(chunk_data)
    data_chunks.append(df_chunk)
    print(f"✅ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded {len(df_chunk)} rows (total: {sum(len(c) for c in data_chunks)})")

# 📌 Combine all chunks into a single DataFrame
data_query = pd.concat(data_chunks, ignore_index=True)
print(f"✅ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Successfully loaded {len(data_query)} rows from BigQuery!")

# 📌 Convert Date Columns
data_query["searchDate"] = pd.to_datetime(data_query["searchDate"])
data_query["flightDate"] = pd.to_datetime(data_query["flightDate"])

# 📌 Feature Engineering
data_query["days_to_flight"] = (data_query["flightDate"] - data_query["searchDate"]).dt.days
data_query = data_query[data_query["days_to_flight"] > 0]

# 📌 One-Hot Encode Categorical Features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(data_query[['startingAirport', 'destinationAirport', 'segmentsAirlineName']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

# 📌 Merge Encoded Data
data_query = data_query.reset_index(drop=True)
data_query = pd.concat([data_query, encoded_df], axis=1)

# 📌 Select Features
X = data_query[['days_to_flight', 'seatsRemaining', 'isRefundable'] + list(encoded_df.columns)]
y = data_query['totalFare']

# 📌 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Define XGBoost Parameters (Tuned)
params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

# 📌 Perform 5-Fold Cross-Validation
print(f"⏳ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running 5-Fold Cross-Validation on XGBoost...")
dtrain = DMatrix(X_train, label=y_train)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cv(
    params,
    dtrain,
    num_boost_round=500,  # ✅ Controls trees instead of n_estimators
    folds=kf,
    metrics="mae",
    early_stopping_rounds=10
)
# 📌 Best MAE from CV
best_mae = cv_results['test-mae-mean'].min()
print(f"✅ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Best Cross-Validation MAE: ${best_mae:.2f}")

# 📌 Train XGBoost Model
print(f"⏳ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training XGBoost Model...")
model = XGBRegressor(**params)
model.fit(X_train, y_train)
print(f"✅ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] XGBoost Model Trained!")

# 📌 Make Predictions
y_pred = model.predict(X_test)

# 📌 Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mape = (mae / y_test.mean()) * 100  # Mean Absolute Percentage Error
print(f"\n📊 Mean Absolute Error (MAE): ${mae:.2f}")
print(f"📊 Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# 📌 Scatter Plot: Actual vs. Predicted Prices
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Actual vs. Predicted Flight Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2)
plt.show()

# 📌 Find Best Booking Date for Any Flight
flight_date = datetime(2025, 5, 1)
search_dates = [flight_date - timedelta(days=i) for i in range(1, 61)]

# Create DataFrame for Predictions
search_df = pd.DataFrame({
    'days_to_flight': [(flight_date - d).days for d in search_dates],
    'seatsRemaining': np.median(data_query['seatsRemaining']),
    'isRefundable': False
})

# Add Encoded Features
for col in encoded_df.columns:
    search_df[col] = 0

# Predict Prices for Different Booking Dates
predicted_fares = model.predict(search_df)

# Find Best Date to Book
best_search_date = search_dates[np.argmin(predicted_fares)]
print(f"\n📅 Best date to book for any flight (May 1st flight): {best_search_date.strftime('%Y-%m-%d')}")

# 📌 Print Total Execution Time
print(f"\n⏳ Total script execution time: {time.time() - start_time:.2f} seconds")
