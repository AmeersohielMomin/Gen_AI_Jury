"""CLI helper to build and verify artifacts/jury_eval_cache.json."""

from __future__ import annotations

import time

import pandas as pd

from src.constants import ARTIFACT_DIR, CACHE_PATH
from src.evaluation_cache import build_evaluation_cache


def main() -> None:
    start = time.time()
    print("Building GenAI Jury evaluation cache from saved checkpoints...", flush=True)
    cache = build_evaluation_cache(CACHE_PATH)
    elapsed = time.time() - start
    print(f"Saved {CACHE_PATH} in {elapsed:.1f}s.", flush=True)

    ref_csv = ARTIFACT_DIR / "classifier_comparison.csv"
    if ref_csv.exists():
        ref = pd.read_csv(ref_csv).set_index("arm")
        for arm in cache["arm_order"]:
            m = cache["arms"][arm]["metrics"]
            diff = abs(m["target_f1"] - round(float(ref.loc[arm, "target_f1"]), 4))
            print(
                f"  {arm:18s} | SCC F1={m['target_f1']:.4f} (CI {m['target_f1_ci_low']:.3f}-{m['target_f1_ci_high']:.3f}) "
                f"| PR-AUC={m['target_pr_auc']:.4f} | Macro-F1={m['macro_f1']:.4f} | diff={diff:.5f}",
                flush=True,
            )
    print(f"Detector AUC: {cache['cgan_audit']['synthetic_detector_auc']}", flush=True)
    print(f"Curated samples indexed: {len(cache['samples'])}", flush=True)


if __name__ == "__main__":
    main()
