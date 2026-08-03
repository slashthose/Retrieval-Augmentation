"""Fail fast if a generated CSV cannot meet the organizer's output contract."""
import argparse, csv
from pathlib import Path
from router import OUT_COLUMNS

def main(path: Path, dataset: Path):
    rows=list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    expected={r["message_id"] for r in csv.DictReader((dataset/"messages.csv").open(encoding="utf-8-sig"))}
    assert rows and list(rows[0]) == OUT_COLUMNS, "wrong output columns or ordering"
    assert len(rows)==len(expected) and {r["message_id"] for r in rows}==expected, "message IDs do not match messages.csv"
    assert all(r["action"] in {"notify","digest","mute"} for r in rows), "invalid action"
    assert all(0 <= float(r["confidence"]) <= 1 for r in rows), "confidence outside [0,1]"
    assert all(r["evidence_message_ids"] for r in rows), "missing evidence field"
    print(f"VALID: {len(rows)} predictions with the required schema")
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,default=Path("output.csv")); p.add_argument("--dataset",type=Path,default=Path("dataset")); a=p.parse_args(); main(a.output,a.dataset)
