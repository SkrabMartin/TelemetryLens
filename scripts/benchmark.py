"""Time each pipeline stage for growing synthetic files (5 machines).

python scripts/benchmark.py --hours 1 8 24
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telemetrylens import cycles, generator, ingest, insights, quality, schema  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, nargs="+", default=[1, 8, 24])
    args = ap.parse_args()
    print("| hours | rows | CSV MB | parse s | quality s | clean+cycles s | insights s | total s |")
    print("|---|---|---|---|---|---|---|---|")
    for h in args.hours:
        content = generator.to_csv_bytes(generator.generate(5, h, seed=42).data)
        t = [time.perf_counter()]
        p = ingest.parse(content)
        t.append(time.perf_counter())
        quality.quality_report(p.data, p.sensors)
        t.append(time.perf_counter())
        df, _ = cycles.add_cycle_columns(quality.clean(p.data, p.sensors).data)
        t.append(time.perf_counter())
        insights.threshold_summary(df, p.sensors, schema.default_thresholds())
        insights.flag_anomalies(df, p.sensors)
        insights.state_summary(df, p.sensors)
        cycles.cycle_table(df, p.sensors)
        t.append(time.perf_counter())
        d = [b - a for a, b in zip(t, t[1:], strict=False)]
        print(
            f"| {h:g} | {len(p.raw):,} | {len(content) / 1e6:.1f} | "
            + " | ".join(f"{x:.2f}" for x in d)
            + f" | {sum(d):.2f} |"
        )


if __name__ == "__main__":
    main()
