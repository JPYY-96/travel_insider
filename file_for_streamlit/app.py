import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# === Load model + encoder ===
model = load_model("file_for_streamlit/LSTMv2.h5")
encoder = joblib.load("file_for_streamlit/encoder.joblib")

st.title("✈️ Best Flight Booking Date Advisor")
st.markdown("Select your route and flight date to find the best day to book.")

# === User Inputs ===
startingAirport = st.selectbox("From", ["ATL", "DFW", "DEN", "ORD", "LAX", "CLT", "MIA", "JFK", "EWR", "SFO", "DTW", "BOS", "PHL", "LGA", "IAD", "OAK"]
)
destinationAirport = st.selectbox("To", ["ATL", "DFW", "DEN", "ORD", "LAX", "CLT", "MIA", "JFK", "EWR", "SFO", "DTW", "BOS", "PHL", "LGA", "IAD", "OAK"]
)
airline = st.selectbox("Airline", ["American Airlines", "Delta", "United"])

min_flight_date = datetime.today().date() + timedelta(days=1)
max_flight_date = min_flight_date + timedelta(days=3650)

flightDate = st.date_input(
    "Flight Date",
    value=min_flight_date,
    min_value=min_flight_date,
    max_value=max_flight_date
)

isRefundable = st.selectbox("Refundable Ticket?", ["Yes", "No"])

# === Backend Logic ===
today = datetime.today().date()
search_dates = [
    today + timedelta(days=i)
    for i in range((flightDate - today).days)
    if (flightDate - (today + timedelta(days=i))).days > 0
]

rows = []
for searchDate in search_dates:
    days_to_flight = (flightDate - searchDate).days
    row = {
        "days_to_flight": days_to_flight,
        "days_to_flight_squared": days_to_flight ** 2,
        "day_of_week": flightDate.weekday(),
        "is_weekend": int(flightDate.weekday() >= 5),
        "is_holiday_season": int(flightDate.month in [6, 7, 12]),
        "flight_month": flightDate.month,
        "flight_year": flightDate.year,
        "search_month": searchDate.month,
        "search_day": searchDate.day,
        "days_to_flight_log": np.log1p(days_to_flight),
        "seatsRemaining": 3,
        "isRefundable": 1 if isRefundable == "Yes" else 0,
        "startingAirport": startingAirport,
        "destinationAirport": destinationAirport,
        "segmentsAirlineName": airline,
        "searchDate": searchDate.toordinal()  # convert to numeric
    }
    rows.append(row)

df = pd.DataFrame(rows)

# === One-Hot Encode ===
encoded = encoder.transform(df[["startingAirport", "destinationAirport", "segmentsAirlineName"]])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(), index=df.index)

df = pd.concat([
    df.drop(columns=["startingAirport", "destinationAirport", "segmentsAirlineName"]),
    encoded_df
], axis=1)

# === Feature Alignment ===
expected_cols = [
    'days_to_flight', 'days_to_flight_squared', 'day_of_week', 'is_weekend',
    'is_holiday_season', 'flight_month', 'flight_year', 'search_month', 'search_day',
    'days_to_flight_log', 'seatsRemaining', 'isRefundable',
    'startingAirport_ATL', 'startingAirport_BOS', 'startingAirport_CLT', 'startingAirport_DEN',
    'startingAirport_DFW', 'startingAirport_DTW', 'startingAirport_EWR', 'startingAirport_IAD',
    'startingAirport_JFK', 'startingAirport_LAX', 'startingAirport_LGA', 'startingAirport_MIA',
    'startingAirport_OAK', 'startingAirport_ORD', 'startingAirport_PHL', 'startingAirport_SFO',
    'destinationAirport_ATL', 'destinationAirport_BOS', 'destinationAirport_CLT', 'destinationAirport_DEN',
    'destinationAirport_DFW', 'destinationAirport_DTW', 'destinationAirport_EWR', 'destinationAirport_IAD',
    'destinationAirport_JFK', 'destinationAirport_LAX', 'destinationAirport_LGA', 'destinationAirport_MIA',
    'destinationAirport_OAK', 'destinationAirport_ORD', 'destinationAirport_PHL', 'destinationAirport_SFO',
    'segmentsAirlineName_American Airlines', 'segmentsAirlineName_Delta', 'segmentsAirlineName_United'
]

# Fill missing cols
missing_cols = [col for col in expected_cols if col not in df.columns]
for col in missing_cols:
    df[col] = 0

X = df[expected_cols].copy()

# === Predict ===
predictions = model.predict(X)

# === Add prediction + display info back to original df
df["predicted_fare"] = predictions
df["searchDate_display"] = df["searchDate"].apply(lambda x: datetime.fromordinal(int(x)))
# === Result ===
best_row = df.loc[df["predicted_fare"].idxmin()]
best_date = best_row["searchDate_display"]
best_fare = best_row["predicted_fare"]

st.success(f"📅 Best day to book: **{best_date.strftime('%Y-%m-%d')}**")
st.metric(label="Estimated Fare", value=f"${best_fare:.2f}")

# === Optional: Show Full Prediction Table ===
if st.checkbox("Show full fare prediction table"):
    st.dataframe(df[["searchDate_display", "predicted_fare"]].rename(columns={"searchDate_display": "searchDate"}).sort_values("searchDate"))
