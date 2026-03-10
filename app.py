from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from datetime import datetime
import os
import joblib

app = Flask(__name__)
app.secret_key = "supersecretkey123"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

FRAUD_LOG = os.path.join(DATA_DIR, "flagged_transactions.csv")
NORMAL_LOG = os.path.join(DATA_DIR, "normal_transactions.csv")
RETRAIN_DATA = os.path.join(DATA_DIR, "retraining_dataset.csv")

MODEL_PATH = "fraud_detection_model.pkl"

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
    'Latitude',
    'Longitude',
    'Model Prediction',
    'Investigation Status',
    'Timestamp'
]

# Initialize CSV files
for file in [FRAUD_LOG, NORMAL_LOG, RETRAIN_DATA]:
    if not os.path.exists(file):
        pd.DataFrame(columns=BASE_COLUMNS).to_csv(file, index=False)

model = joblib.load(MODEL_PATH)


# -----------------------------
# Helper Functions
# -----------------------------

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
        'Latitude': record['latitude'],
        'Longitude': record['longitude'],
        'Model Prediction': record['model_prediction'],
        'Investigation Status': record['investigation_status'],
        'Timestamp': record['timestamp']
    }


def append_row(file_path, record):
    df = pd.DataFrame([record])
    df.to_csv(file_path, mode="a", header=not os.path.exists(file_path), index=False)


def load_records(file_path):

    if not os.path.exists(file_path):
        return []

    df = pd.read_csv(file_path)

    df = df.rename(columns={
        "Sender Account ID": "sender_account",
        "Receiver Account ID": "receiver_account",
        "Transaction Type": "transaction_type",
        "Transaction Amount (₦)": "transaction_amount",
        "Origin Balance Before (₦)": "origin_balance_before",
        "Destination Balance Before (₦)": "destination_balance_before",
        "Transaction Hour (Step)": "transaction_hour",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Timestamp": "timestamp"
    })

    return df.to_dict(orient="records")


# -----------------------------
# Routes
# -----------------------------

@app.route("/")
@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/data-entry", methods=["GET", "POST"])
def data_entry():

    if request.method == "POST":

        data = {
            "transaction_hour": int(request.form["transaction_hour"]),
            "transaction_amount": float(request.form["transaction_amount"]),
            "origin_balance_before": float(request.form["origin_balance_before"]),
            "destination_balance_before": float(request.form["destination_balance_before"]),
            "transaction_type": request.form["transaction_type"],
            "sender_account": request.form["sender_account"],
            "receiver_account": request.form["receiver_account"],
            "latitude": float(request.form.get("latitude", 0)),
            "longitude": float(request.form.get("longitude", 0))
        }

        session["transaction_data"] = data
        return redirect(url_for("predict"))

    return render_template("data_entry.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    data = session.get("transaction_data")
    result = None
    error_message = None

    if request.method == "POST":
        data = {
            "transaction_hour": int(request.form["transaction_hour"]),
            "transaction_amount": float(request.form["transaction_amount"]),
            "origin_balance_before": float(request.form["origin_balance_before"]),
            "destination_balance_before": float(request.form["destination_balance_before"]),
            "transaction_type": request.form["transaction_type"],
            "sender_account": request.form["sender_account"],
            "receiver_account": request.form["receiver_account"],
            "latitude": float(request.form.get("latitude", 0)),
            "longitude": float(request.form.get("longitude", 0))
        }

    if data:

        # -----------------------------
        # VALIDATION RULE
        # -----------------------------
        if data["transaction_amount"] > data["origin_balance_before"]:

            result = "Invalid"
            error_message = "Invalid transaction: Sender balance is lower than the transaction amount."

            return render_template(
                "predict.html",
                data=data,
                result=result,
                error_message=error_message
            )

        # -----------------------------
        # Continue if valid
        # -----------------------------
        origin_after = data["origin_balance_before"] - data["transaction_amount"]
        dest_after = data["destination_balance_before"] + data["transaction_amount"]

        tx_types = {
            "transaction_type_CASH_IN": 0,
            "transaction_type_CASH_OUT": 0,
            "transaction_type_DEBIT": 0,
            "transaction_type_PAYMENT": 0,
            "transaction_type_TRANSFER": 0
        }

        tx_types[f"transaction_type_{data['transaction_type']}"] = 1

        X = pd.DataFrame([[ 
            data["transaction_hour"],
            data["transaction_amount"],
            data["origin_balance_before"],
            origin_after,
            data["destination_balance_before"],
            dest_after,
            tx_types["transaction_type_CASH_IN"],
            tx_types["transaction_type_CASH_OUT"],
            tx_types["transaction_type_DEBIT"],
            tx_types["transaction_type_PAYMENT"],
            tx_types["transaction_type_TRANSFER"]
        ]],
        columns=[
            'transaction_hour',
            'transaction_amount',
            'origin_balance_before',
            'origin_balance_after',
            'destination_balance_before',
            'destination_balance_after',
            'transaction_type_CASH_IN',
            'transaction_type_CASH_OUT',
            'transaction_type_DEBIT',
            'transaction_type_PAYMENT',
            'transaction_type_TRANSFER'
        ])

        prediction = int(model.predict(X)[0])
        result = "Fraud" if prediction == 1 else "Not Fraud"

        record = {
            **data,
            "origin_balance_after": origin_after,
            "destination_balance_after": dest_after,
            "model_prediction": result,
            "investigation_status": "Pending" if result == "Fraud" else "Not Required",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        cleaned = clean_record(record)

        append_row(RETRAIN_DATA, cleaned)

        if result == "Fraud":
            append_row(FRAUD_LOG, cleaned)
        else:
            append_row(NORMAL_LOG, cleaned)

        session.pop("transaction_data", None)

    return render_template("predict.html", data=data, result=result, error_message=error_message)


@app.route("/fraud-records")
def fraud_records():
    data = load_records(FRAUD_LOG)
    return render_template("fraud_records.html", data=data)


@app.route("/normal-records")
def normal_records():
    data = load_records(NORMAL_LOG)
    return render_template("normal_records.html", data=data)


if __name__ == "__main__":
    app.run(debug=True)