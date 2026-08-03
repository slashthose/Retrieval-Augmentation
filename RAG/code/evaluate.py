"""Local evaluation against labeled sample_messages.csv, when supplied."""
import argparse, csv
from pathlib import Path
from router import predict_rows

def main(dataset: Path):
    sample=dataset/"sample_messages.csv"
    if not sample.exists(): raise FileNotFoundError("sample_messages.csv is required for evaluation")
    gold=list(csv.DictReader(sample.open(encoding="utf-8-sig")))
    pred={r["message_id"]:r for r in predict_rows(dataset, gold)}
    rows=[(pred.get(g["message_id"],{}),g) for g in gold]
    for field in ("action","message_type"):
        print(f"{field}_accuracy: {sum(p.get(field)==g.get(field) for p,g in rows)/max(1,len(rows)):.3f}")
    print(f"evaluated_rows: {len(rows)}")
if __name__=="__main__":
    a=argparse.ArgumentParser(); a.add_argument("--dataset",type=Path,default=Path("dataset")); main(a.parse_args().dataset)
