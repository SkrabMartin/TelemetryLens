"""Write a synthetic telemetry CSV and its ground-truth fault list.

python scripts/generate_data.py --machines 5 --hours 8 --seed 42 --out data/sample_telemetry.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telemetrylens import generator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machines", type=int, default=5)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", default="2026-10-22T06:00:00Z")
    ap.add_argument("--clean", action="store_true", help="Do not inject faults.")
    ap.add_argument("--out", type=Path, default=Path("data/sample_telemetry.csv"))
    args = ap.parse_args()

    ds = generator.generate(args.machines, args.hours, args.start, args.seed, inject_faults=not args.clean)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(generator.to_csv_bytes(ds.data))
    truth = args.out.with_name(args.out.stem + "_ground_truth.csv")
    ds.ground_truth.to_csv(truth, index=False)
    print(f"Wrote {len(ds.data):,} rows to {args.out} and {len(ds.ground_truth)} faults to {truth}")


if __name__ == "__main__":
    main()
