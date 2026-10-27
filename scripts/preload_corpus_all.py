"""Download Deep corpus=all train H5s (TIMIT / MOCHA / The Moth) via pnpl.

Sherlock is already on the farm. This only fetches missing runs.
Does not train. Safe to re-run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pnpl.datasets import LibriBrain100Word

DATA_PATH = ROOT / "data" / "libribrain_word"


def main() -> None:
    t0 = time.perf_counter()
    print("preload LibriBrain100Word corpus=all train…", flush=True)
    ds = LibriBrain100Word(
        data_path=str(DATA_PATH),
        partition="train",
        subjects="deep",
        corpus="all",
        tmin=0.0,
        tmax=1.0,
        standardize=False,
        include_info=False,
        preload_files=False,
        download=True,
    )
    tasks = {}
    for s in ds.samples:
        tasks[str(s[2])] = tasks.get(str(s[2]), 0) + 1
    print(f"runs={len(ds.run_keys)}  samples={len(ds.samples)}", flush=True)
    print("task counts:", dict(sorted(tasks.items())), flush=True)
    print(f"wall={time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
