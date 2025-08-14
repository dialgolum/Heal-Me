import pandas as pd
import json
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "conditions.json"

def import_from_csv(csv_path: Path, output_path: Path = OUTPUT_PATH):
    print(f"Importing KB from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Map your dataset's columns to our standard format
    col_mapping = {
        "label": "disease",
        "text": "symptoms"
    }
    df = df.rename(columns=col_mapping)

    # Ensure columns exist
    required = {"disease", "symptoms"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns (or mappable to): {required}")
    
    kb = []
    for i, row in df.iterrows():
        cond = {
            "id": f"cond_{i+1}",
            "name": row["disease"].strip(),
            "symptoms": [s.strip().lower() for s in row["symptoms"].split(",")],
            "category": row.get("category", "Unknown"),
            "risk_level": row.get("risk_level", "unknown").lower(),
            "advice": row.get("advice", "Consult a healthcare professional.")
        }
        kb.append(cond)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(kb)} conditions to {output_path}")

if __name__ == "__main__":
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {RAW_DIR}")
    elif len(csv_files) == 1:
        import_from_csv(csv_files[0])
    else:
        print("Multiple CSV files found:")
        for idx, file in enumerate(csv_files, start=1):
            print(f"{idx}. {file.name}")
        choice = input("Select file number: ")
        try:
            choice_idx = int(choice) - 1
            import_from_csv(csv_files[choice_idx])
        except Exception:
            print("❌ Invalid choice")