import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

# -------------------
# Load environment variables
# -------------------
load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
MODEL_PATH = os.getenv("MODEL_PATH", "fraud_detection_model.pkl")

# Validate required env variables
required_vars = ["EMAIL_USER", "EMAIL_PASS", "MODEL_PATH"]
for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"Environment variable '{var}' not set in .env file")

# -------------------
# Flask App Setup
# -------------------
app = Flask(__name__)
#app.secret_key = os.getenv("FLASK_SECRET_KEY", "fraud_secret_key")  # optional env override

# -------------------
# Paths and CSV files
# -------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

FRAUD_LOG = os.path.join(DATA_DIR, "fraud_transactions.csv")
NORMAL_LOG = os.path.join(DATA_DIR, "normal_transactions.csv")
RETRAIN_DATA = os.path.join(DATA_DIR, "retraining_dataset.csv")

columns = [
    "timestamp",
    "sender_account",
    "receiver_account",
    "transaction_type",
    "transaction_amount",
    "origin_balance_before",
    "origin_balance_after",
    "destination_balance_before",
    "destination_balance_after",
    "transaction_hour",
    "latitude",
    "longitude",
    "model_prediction",
    "investigation_status"
]

for file in [FRAUD_LOG, NORMAL_LOG, RETRAIN_DATA]:
    if not os.path.exists(file):
        pd.DataFrame(columns=columns).to_csv(file, index=False)

# -------------------
# Load ML Model
# -------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ML model not found at path: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# -------------------
# Helper functions
# -------------------
def save_record(file_path, record):
    df = pd.DataFrame([record])
    df.to_csv(file_path, mode="a", header=False, index=False)

def load_records(file_path):
    if not os.path.exists(file_path):
        return []
    df = pd.read_csv(file_path)
    return df.to_dict(orient="records")

# -------------------
# Email Alert
# -------------------
def send_fraud_alert(record, receiver_email):
    subject = "🚨 Fraud Transaction Detected"
    body = f"""
Fraud Transaction Alert

Sender Account: {record['sender_account']}
Receiver Account: {record['receiver_account']}
Amount: ₦{record['transaction_amount']}
Transaction Type: {record['transaction_type']}

Location:
Latitude: {record['latitude']}
Longitude: {record['longitude']}

Time:
{record['timestamp']}

Please investigate immediately.
"""

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = receiver_email

    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, receiver_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print("Email error:", e)
        return False

# -------------------
# Routes
# -------------------
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
            "longitude": float(request.form.get("longitude", 0)),
            "investigator_email": request.form["investigator_email"]
        }
        session["transaction_data"] = data
        return redirect(url_for("predict"))

    return render_template("data_entry.html")

@app.route("/predict")
def predict():
    data = session.get("transaction_data")
    if not data:
        return redirect(url_for("data_entry"))

    # Validate transaction
    if data["transaction_amount"] > data["origin_balance_before"]:
        return render_template(
            "predict.html",
            data=data,
            result="Invalid",
            error_message="Transaction rejected: sender balance is lower than transaction amount",
            alert_message=None
        )

    origin_after = data["origin_balance_before"] - data["transaction_amount"]
    destination_after = data["destination_balance_before"] + data["transaction_amount"]

    tx = {"CASH_IN": 0, "CASH_OUT": 0, "DEBIT": 0, "PAYMENT": 0, "TRANSFER": 0}
    tx[data["transaction_type"]] = 1

    X = pd.DataFrame([[ 
        data["transaction_hour"],
        data["transaction_amount"],
        data["origin_balance_before"],
        origin_after,
        data["destination_balance_before"],
        destination_after,
        tx["CASH_IN"],
        tx["CASH_OUT"],
        tx["DEBIT"],
        tx["PAYMENT"],
        tx["TRANSFER"]
    ]], columns=[
        "transaction_hour",
        "transaction_amount",
        "origin_balance_before",
        "origin_balance_after",
        "destination_balance_before",
        "destination_balance_after",
        "transaction_type_CASH_IN",
        "transaction_type_CASH_OUT",
        "transaction_type_DEBIT",
        "transaction_type_PAYMENT",
        "transaction_type_TRANSFER"
    ])

    prediction = int(model.predict(X)[0])
    result = "Fraud" if prediction == 1 else "Not Fraud"

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sender_account": data["sender_account"],
        "receiver_account": data["receiver_account"],
        "transaction_type": data["transaction_type"],
        "transaction_amount": data["transaction_amount"],
        "origin_balance_before": data["origin_balance_before"],
        "origin_balance_after": origin_after,
        "destination_balance_before": data["destination_balance_before"],
        "destination_balance_after": destination_after,
        "transaction_hour": data["transaction_hour"],
        "latitude": data["latitude"],
        "longitude": data["longitude"],
        "model_prediction": result,
        "investigation_status": "Pending" if result == "Fraud" else "Not Required"
    }

    save_record(RETRAIN_DATA, record)

    alert_message = None
    if result == "Fraud":
        save_record(FRAUD_LOG, record)
        email_sent = send_fraud_alert(record, data["investigator_email"])
        if email_sent:
            alert_message = f"Fraud alert email sent to {data['investigator_email']}"
        else:
            alert_message = "Fraud detected but email could not be delivered. Check the email address."
    else:
        save_record(NORMAL_LOG, record)

    session.pop("transaction_data", None)

    return render_template("predict.html", data=data, result=result, alert_message=alert_message, error_message=None)

@app.route("/fraud-records")
def fraud_records():
    records = load_records(FRAUD_LOG)
    return render_template("fraud_records.html", data=records)

@app.route("/normal-records")
def normal_records():
    records = load_records(NORMAL_LOG)
    return render_template("normal_records.html", data=records)

if __name__ == "__main__":

    app.run(debug=True)




