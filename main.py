import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from utils import init_csv, append_row
import os
import random

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Fraud Risk Control System", layout="wide")

MODEL_PATH = "fraud_detection_model.pkl"
DATA_DIR = "data"
FRAUD_LOG = f"{DATA_DIR}/flagged_transactions.csv"
RETRAIN_DATA = f"{DATA_DIR}/retraining_dataset.csv"

BASE_COLUMNS = [
    'Transaction Hour (Step)',
    'Transaction Amount (₦)',
    'Origin Balance Before (₦)',
    'Origin Balance After (₦)',
    'Destination Balance Before (₦)',
    'Destination Balance After (₦)',
    'Transaction Type',
    'Sender Account ID',
    'Receiver Account ID',
    'Model Prediction (Fraud/Not Fraud)',
    'Investigation Status',
    'Timestamp'
]

# -------------------------------
# INIT STORAGE
# -------------------------------
os.makedirs(DATA_DIR, exist_ok=True)
init_csv(FRAUD_LOG, BASE_COLUMNS)
init_csv(RETRAIN_DATA, BASE_COLUMNS)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -------------------------------
# DUMMY ACCOUNTS
# -------------------------------
accounts_list = [str(random.randint(1000000000, 9999999999)) for _ in range(20)]

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("About Project")
st.sidebar.info(
    """
    Fraud Risk Control System using CatBoost model.
    The model predicts if a transaction is fraudulent.
    """
)

menu = st.sidebar.radio("Navigation", ["Fraud Prediction", "Fraud Cases", "Retraining Data"])

# -------------------------------
# HELPER FUNCTION
# -------------------------------
def clean_record(record):
    return {
        'Transaction Hour (Step)': record['transaction_hour'],
        'Transaction Amount (₦)': record['transaction_amount'],
        'Origin Balance Before (₦)': record['origin_balance_before'],
        'Origin Balance After (₦)': record['origin_balance_after'],
        'Destination Balance Before (₦)': record['destination_balance_before'],
        'Destination Balance After (₦)': record['destination_balance_after'],
        'Transaction Type': record['transaction_type'],
        'Sender Account ID': record['sender_account'],
        'Receiver Account ID': record['receiver_account'],
        'Model Prediction (Fraud/Not Fraud)': record['model_prediction'],
        'Investigation Status': record['investigation_status'],
        'Timestamp': record['timestamp']
    }

# -------------------------------
# FRAUD PREDICTION PAGE
# -------------------------------
if menu == "Fraud Prediction":
    st.title("Transaction Fraud Assessment")

    st.subheader("Account Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        sender_account = st.selectbox("Initiating Account ID", accounts_list)
    with col2:
        receiver_account = st.selectbox("Receiving Account ID", accounts_list)
    with col3:
        transaction_hour = st.number_input("Transaction Step", min_value=0, max_value=700, step=1, value=0)

    st.subheader("Transaction Details")
    col4, col5, col6 = st.columns(3)
    with col4:
        transaction_amount = st.number_input("Transaction Amount (₦)", min_value=0.0)
        origin_balance_before = st.number_input("Origin Balance Before (₦)", min_value=0.0)
    with col5:
        destination_balance_before = st.number_input("Destination Balance Before (₦)", min_value=0.0)
    with col6:
        origin_balance_after = origin_balance_before - transaction_amount
        if origin_balance_after < 0:
            st.error("❌ Origin balance insufficient for transaction")
            st.stop()
        destination_balance_after = destination_balance_before + transaction_amount
        st.markdown(f"**Origin Balance After (₦):** {origin_balance_after:.2f}")
        st.markdown(f"**Destination Balance After (₦):** {destination_balance_after:.2f}")

    st.subheader("Transaction Type")
    transaction_type = st.radio("Select Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])

    if st.button("Run Fraud Check"):
        # Prepare input for CatBoost
        tx_types = {f"transaction_type_{t}": 0 for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]}
        tx_types[f"transaction_type_{transaction_type}"] = 1
        X = pd.DataFrame([[transaction_hour,
                           transaction_amount,
                           origin_balance_before,
                           origin_balance_after,
                           destination_balance_before,
                           destination_balance_after,
                           *tx_types.values()]], columns=[
                               'transaction_hour','transaction_amount','origin_balance_before','origin_balance_after',
                               'destination_balance_before','destination_balance_after',
                               'transaction_type_CASH_IN','transaction_type_CASH_OUT','transaction_type_DEBIT',
                               'transaction_type_PAYMENT','transaction_type_TRANSFER'
                           ])
        # CatBoost prediction
        raw_pred = int(model.predict(X)[0])
        prediction_label = "Fraud" if raw_pred == 1 else "Not Fraud"

        # Save record
        record = {
            "transaction_hour": transaction_hour,
            "transaction_amount": transaction_amount,
            "origin_balance_before": origin_balance_before,
            "origin_balance_after": origin_balance_after,
            "destination_balance_before": destination_balance_before,
            "destination_balance_after": destination_balance_after,
            "transaction_type": transaction_type,
            "sender_account": sender_account,
            "receiver_account": receiver_account,
            "model_prediction": prediction_label,
            "investigation_status": "Pending" if prediction_label == "Fraud" else "Not Required",
            "timestamp": datetime.now()
        }

        append_row(RETRAIN_DATA, clean_record(record))
        if prediction_label == "Fraud":
            append_row(FRAUD_LOG, clean_record(record))
            st.error("🚨 Transaction flagged as Fraud!")
        else:
            st.success("✅ Transaction cleared.")

# -------------------------------
# FRAUD CASES PAGE
# -------------------------------
elif menu == "Fraud Cases":
    st.title("Fraud Investigation Queue")
    if os.path.exists(FRAUD_LOG) and os.path.getsize(FRAUD_LOG) > 0:
        df = pd.read_csv(FRAUD_LOG)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No fraud cases have been logged yet.")

# -------------------------------
# RETRAINING DATA PAGE
# -------------------------------
elif menu == "Retraining Data":
    st.title("Retraining Dataset")
    if os.path.exists(RETRAIN_DATA) and os.path.getsize(RETRAIN_DATA) > 0:
        df = pd.read_csv(RETRAIN_DATA)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No retraining data available yet.")
