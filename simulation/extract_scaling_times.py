#!/usr/bin/env python3
"""
Extract GEOS timing data from scaling-test output.log files.

Walks an output directory, finds every output.log, parses the timing block
that GEOS prints at the end, and writes a tidy CSV. Also prints a small
summary (mean / min / max / std of run time per rank count).

Directory names are expected to look like:
    kgd_benchmark_16_procs_run3            (strong scaling)
    kgd_benchmark_weak_16_procs_run3       (weak scaling)

Usage:
    python extract_scaling_times.py <output_dir> [-o results.csv] [--weak]

    <output_dir>   Top-level dir containing the per-run subdirectories
    -o / --output  CSV output path (default: scaling_times.csv in cwd)
    --weak         Treat dir names as weak-scaling (only affects the
                   'type' column; parsing still auto-detects the 'weak' tag)
"""

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

# --- regexes -----------------------------------------------------------------

# "run time              00h28m24s (1704.838308872 s)"  -> capture the seconds
RE_TIMES = {
    "total_time_s": re.compile(r"^total time\s+\S+\s+\(([\d.]+)\s*s\)", re.M),
    "init_time_s": re.compile(r"^initialization time\s+\S+\s+\(([\d.]+)\s*s\)", re.M),
    "run_time_s": re.compile(r"^run time\s+\S+\s+\(([\d.]+)\s*s\)", re.M),
}
RE_RANKS = re.compile(r"^Num ranks:\s*(\d+)", re.M)
RE_STARTED = re.compile(r"^Started at\s+(.+)$", re.M)
RE_FINISHED = re.compile(r"^Finished at\s+(.+)$", re.M)

# procs + run number from the directory name; 'weak' tag optional
RE_DIRNAME = re.compile(r"(?P<weak>weak_)?(?P<procs>\d+)_procs_run(?P<run>\d+)")


def parse_log(log_path: Path) -> dict:
    """Parse a single output.log. Returns a dict of extracted fields."""
    text = log_path.read_text(errors="replace")

    row = {
        "path": str(log_path),
        "num_ranks": None,
        "total_time_s": None,
        "init_time_s": None,
        "run_time_s": None,
        "started_at": None,
        "finished_at": None,
        "complete": False,
    }

    m = RE_RANKS.search(text)
    if m:
        row["num_ranks"] = int(m.group(1))

    for key, rx in RE_TIMES.items():
        m = rx.search(text)
        if m:
            row[key] = float(m.group(1))

    m = RE_STARTED.search(text)
    if m:
        row["started_at"] = m.group(1).strip()
    m = RE_FINISHED.search(text)
    if m:
        row["finished_at"] = m.group(1).strip()

    # a run is "complete" if we found the final run time line
    row["complete"] = row["run_time_s"] is not None
    return row


def parse_dirname(log_path: Path) -> dict:
    """Pull procs / run number / weak flag from the parent directory name."""
    info = {"procs": None, "run": None, "type": None}
    m = RE_DIRNAME.search(log_path.parent.name)
    if m:
        info["procs"] = int(m.group("procs"))
        info["run"] = int(m.group("run"))
        info["type"] = "weak" if m.group("weak") else "strong"
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", type=Path, help="Top-level scaling output dir")
    ap.add_argument("-o", "--output", type=Path, default=Path("scaling_times.csv"),
                    help="CSV output path (default: scaling_times.csv)")
    ap.add_argument("--weak", action="store_true",
                    help="Force 'type' column to weak when dir names lack the tag")
    args = ap.parse_args()

    if not args.output_dir.is_dir():
        sys.exit(f"error: {args.output_dir} is not a directory")

    log_files = sorted(args.output_dir.rglob("output*.log"))
    if not log_files:
        sys.exit(f"error: no output*.log files found under {args.output_dir}")

    rows = []
    for log_path in log_files:
        row = parse_log(log_path)
        row.update(parse_dirname(log_path))
        if args.weak and row["type"] is None:
            row["type"] = "weak"
        rows.append(row)

    # sort by type, procs, run for a tidy table
    rows.sort(key=lambda r: (r["type"] or "", r["procs"] or 0, r["run"] or 0))

    fieldnames = ["type", "procs", "run", "num_ranks",
                  "run_time_s", "total_time_s", "init_time_s",
                  "started_at", "finished_at", "complete", "path"]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    # --- summary -------------------------------------------------------------
    n_total = len(rows)
    n_complete = sum(1 for r in rows if r["complete"])
    n_incomplete = n_total - n_complete

    print(f"Parsed {n_total} log(s): {n_complete} complete, "
          f"{n_incomplete} incomplete/cancelled.")
    print(f"Wrote {args.output}\n")

    if n_incomplete:
        print("Incomplete runs (no final 'run time' line -- likely cancelled):")
        for r in rows:
            if not r["complete"]:
                print(f"  {r['path']}")
        print()

    # group completed runs by (type, procs) and report stats on run_time_s
    groups = {}
    for r in rows:
        if r["complete"] and r["procs"] is not None:
            groups.setdefault((r["type"], r["procs"]), []).append(r["run_time_s"])

    if groups:
        print(f"{'type':<7} {'procs':>5} {'n':>3} "
              f"{'mean_s':>12} {'min_s':>12} {'max_s':>12} {'std_s':>10}")
        for (typ, procs) in sorted(groups, key=lambda k: (k[0] or "", k[1])):
            vals = groups[(typ, procs)]
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"{typ or '':<7} {procs:>5} {len(vals):>3} "
                  f"{mean:>12.3f} {min(vals):>12.3f} {max(vals):>12.3f} {std:>10.3f}")


if __name__ == "__main__":
    main()
