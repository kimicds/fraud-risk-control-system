import csv
import os

def init_csv(path, headers):
    if not os.path.exists(path):
        with open(path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

def append_row(path, row):
    with open(path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)
