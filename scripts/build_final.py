"""One command from member checkpoints to the final Deep CSV (+ reproducible bundle).

Pipeline (every step is deterministic given the checkpoints and the config):

  1. holdout inference per checkpoint  (submit_tcn_hybrid_holdout.py, blend head, full mode)
  2. log-space average of the member probabilities + few-shot LM fusion  (lm_fuse.py)
  3. transductive holdout-prior correction                              (prior_correct.py)
  4. bundle: checkpoints, member CSVs, final CSV, config, sha256 manifest (bundle_best.py)

    python scripts/build_final.py --ckpts a.ckpt b.ckpt c.ckpt \\
        --name final_v2re --lm Qwen/Qwen2.5-1.5B --fewshot --lam 0.3 --alpha 0.5 [--public 0.72]

Re-running with the same arguments reproduces the CSV (member CSVs are cached
in <work>/members/; pass --force to recompute them).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", type=Path, nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--lm", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--fewshot", action="store_true")
    ap.add_argument("--fewshot-file", type=Path, default=None)
    ap.add_argument("--prefix-file", type=Path, default=None, help="verbatim LM prompt (lm_fuse --prefix-file)")
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--pre-alpha", type=float, default=0.0)
    ap.add_argument("--iso-shrink", type=float, default=0.0,
                    help="isolated rows' own prior shrunk toward uniform (prior_correct --iso-shrink)")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--head", default="blend")
    ap.add_argument("--public", type=float, default=float("nan"))
    ap.add_argument("--work", type=Path, default=ROOT / "submissions" / "final")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-bundle", action="store_true")
    args = ap.parse_args()

    work = args.work / args.name
    (work / "members").mkdir(parents=True, exist_ok=True)
    members = []
    for ck in args.ckpts:
        with open(ck, "rb") as f:
            short = hashlib.sha256(f.read()).hexdigest()[:8]
        tag = f"{ck.stem}__{short}"
        out = work / "members" / f"{tag}.csv"
        if args.force or not out.exists():
            run([PY, "scripts/submit_tcn_hybrid_holdout.py", "--ckpt", str(ck), "--head", args.head,
                 "--mode", "full", "--out", str(out)])
        members.append(out)

    prefix = work / "fused"
    cmd = [PY, "scripts/lm_fuse.py", "--members", *map(str, members), "--lm", args.lm,
           "--lam", str(args.lam), "--beam", str(args.beam), "--sweeps", str(args.sweeps),
           "--dtype", args.dtype, "--batch", "128", "--out-prefix", str(prefix)]
    if args.prefix_file:
        cmd += ["--prefix-file", str(args.prefix_file)]
    elif args.fewshot_file:
        cmd += ["--fewshot-file", str(args.fewshot_file)]
    elif args.fewshot:
        cmd += ["--fewshot"]
    if args.pre_alpha:
        cmd += ["--pre-alpha", str(args.pre_alpha)]
    run(cmd)
    fused = Path(f"{prefix}_lam{args.lam}_g0.0.csv")
    run([PY, "scripts/prior_correct.py", str(fused), "--alpha", str(args.alpha), "--iso-shrink", str(args.iso_shrink),
         "--out-dir", str(work)])
    final = work / f"{fused.stem}_pc{args.alpha}.csv"
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["ckpts"] = [str(c) for c in args.ckpts]
    (work / "config.json").write_text(json.dumps(config, indent=2))
    print(f"FINAL: {final}", flush=True)

    if not args.no_bundle:
        recipe = "python scripts/build_final.py " + " ".join(sys.argv[1:])
        extra = [work / "config.json"] + [f for f in (args.fewshot_file, args.prefix_file) if f]
        run([PY, "scripts/bundle_best.py", "--name", args.name, "--public", str(args.public),
             "--final", str(final), "--members", *map(str, members), "--ckpts", *map(str, args.ckpts),
             "--extra", *map(str, extra), "--recipe", recipe])


if __name__ == "__main__":
    main()
