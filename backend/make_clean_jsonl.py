import pandas as pd
import json

xlsx_path = "Error_data_Bhilai.xlsx"   # your excel file name
out_path = "clean_data.jsonl"

df = pd.read_excel(xlsx_path, sheet_name="Clean Data")

with open(out_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        rec = {
            "fsn": str(row.get("FSN", "")).strip(),
            "information": str(row.get("Information", "")).strip(),
            "description": str(row.get("Description", "")).strip(),
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("✅ Created:", out_path, "rows =", len(df))