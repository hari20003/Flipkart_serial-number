import pandas as pd
import json

file_path = "Error_data_Bhilai.xlsx"

clean_df = pd.read_excel(file_path, sheet_name="Clean Data")
error_df = pd.read_excel(file_path, sheet_name="Errors")

clean_records = []
error_records = []

for _, row in clean_df.iterrows():
    clean_records.append({
        "fsn": row["FSN"],
        "information": row["Information"],
        "description": row["Description"],
        "label": "clean"
    })

for _, row in error_df.iterrows():
    error_records.append({
        "fsn": row["FSN"],
        "information": row["Information"],
        "description": row["Description"],
        "label": "error",
        "error_type": row["Error"],
        "l1": row["L1"],
        "l2": row["L2"]
    })

with open("clean_data.jsonl", "w", encoding="utf-8") as f:
    for r in clean_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open("error_data.jsonl", "w", encoding="utf-8") as f:
    for r in error_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Dataset prepared successfully.")