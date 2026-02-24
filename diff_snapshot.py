from __future__ import annotations

"""
diff_snapshot.py

Purpose
Compare two run snapshot JSON files and print a human readable JSON diff.

What is a snapshot
A snapshot is a compact summary of a run written to:
app/output/snapshots/snapshot_<run_id>.json

Why this script exists
- The YAML report is large and harder to compare across runs.
- Snapshots store just what you need for diffs: KPI scores, confidence, and citations.
- This script lets you diff two runs from the terminal without opening the dashboard.

Example usage
python3 diff_snapshot.py \
  --old app/output/snapshots/snapshot_test1.json \
  --new app/output/snapshots/snapshot_test2.json

Optional output file
python3 diff_snapshot.py --old ... --new ... --out diff.json
"""

import argparse
import json

# These functions come from app/snapshots.py
# load_snapshot reads a snapshot JSON file into a Python dict
# diff_snapshots computes KPI level changes, citation adds or removes, and overall deltas
from app.snapshots import load_snapshot, diff_snapshots


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Diff two Vitelis snapshot json files")
    parser.add_argument("--old", required=True, help="Path to old snapshot json")
    parser.add_argument("--new", required=True, help="Path to new snapshot json")
    parser.add_argument("--out", default="", help="Optional output path for diff json")
    args = parser.parse_args()

    # Load both snapshot JSON files
    old_snapshot = load_snapshot(args.old)
    new_snapshot = load_snapshot(args.new)

    # Compute a diff summary dict
    diff = diff_snapshots(old_snapshot, new_snapshot)

    # Pretty print the diff as JSON
    text = json.dumps(diff, indent=2, sort_keys=False)

    # Either write to a file or print to stdout
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote diff to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()