"""Precompute and load verified evaluation analytics for the GenAI Jury UI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)

from src.constants import (
    ACTIVE_CLASS_INDICES,
    ACTIVE_CLASS_NAMES,
    ARM_METADATA,
    ARM_ORDER,
    ARTIFACT_DIR,
    CACHE_PATH,
    CGAN_DIR,
    DIAGNOSIS_FULL_NAMES,
    N_CLASSES,
    PROJECT_DIR,
    SEED,
    SYNTHETIC_DIR,
    TARGET_CLASS,
    TARGET_LABEL,
    VALID_GAN_EPOCHS,
)
from src.medical_cgan import (
    balanced_loader,
    classifier_transforms,
    evaluate_classifier,
    group_bootstrap_target_f1,
    make_classifier,
    select_device,
    synthetic_detector_auc,
)


def _split_summary(df: pd.DataFrame) -> dict[str, Any]:
    counts = df["diagnosis"].value_counts().to_dict()
    ordered = {name: int(counts.get(name, 0)) for name in ACTIVE_CLASS_NAMES}
    return {
        "images": int(len(df)),
        "lesions": int(df["group_id"].nunique()),
        "counts": ordered,
    }


def _downsample_pr_curve(y_binary: np.ndarray, probs: np.ndarray, max_points: int = 85) -> list[dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_binary, probs)
    if len(recall) > max_points:
        indices = np.linspace(0, len(recall) - 1, max_points, dtype=int)
        # Ensure endpoints are preserved
        indices = np.unique(np.append(indices, [0, len(recall) - 1]))
    else:
        indices = np.arange(len(recall))
    points = []
    for idx in indices:
        thresh = float(thresholds[idx]) if idx < len(thresholds) else 1.0
        points.append({
            "recall": round(float(recall[idx]), 4),
            "precision": round(float(precision[idx]), 4),
            "threshold": round(thresh, 4),
        })
    return points


def _calibration_bins(y_binary: np.ndarray, probs: np.ndarray, n_bins: int = 8) -> dict[str, Any]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        mask = (probs >= low) & (probs <= high if i == n_bins - 1 else probs < high)
        count = int(mask.sum())
        bins.append({
            "range": f"{low:.2f}-{high:.2f}",
            "count": count,
            "mean_pred": round(float(probs[mask].mean()), 4) if count else None,
            "empirical_pos_rate": round(float(y_binary[mask].mean()), 4) if count else None,
        })
    return {
        "mean_prob_true_scc": round(float(probs[y_binary == 1].mean()), 4),
        "mean_prob_non_scc": round(float(probs[y_binary == 0].mean()), 4),
        "bins": bins,
    }


def _per_class_rows(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, Any]]:
    labels = list(ACTIVE_CLASS_INDICES)
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    rows = []
    for idx, name in enumerate(ACTIVE_CLASS_NAMES):
        rows.append({
            "index": idx,
            "diagnosis": name,
            "full_name": DIAGNOSIS_FULL_NAMES[name],
            "precision": round(float(p[idx]), 4),
            "recall": round(float(r[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(sup[idx]),
        })
    return rows


def _phash_histogram(series: pd.Series, bin_width: int = 2) -> list[dict[str, Any]]:
    vals = series.dropna().to_numpy(dtype=int)
    if len(vals) == 0:
        return []
    max_val = min(int(vals.max()), 32)
    bins = []
    for start in range(0, max_val + 1, bin_width):
        end = start + bin_width - 1
        count = int(((vals >= start) & (vals <= end)).sum())
        bins.append({"label": f"{start}-{end}", "start": start, "end": end, "count": count})
    overflow = int((vals > max_val).sum())
    if overflow > 0:
        bins.append({"label": f">{max_val}", "start": max_val + 1, "end": 64, "count": overflow})
    return bins


def _select_curated_samples(
    test_df: pd.DataFrame,
    outputs_by_arm: dict[str, dict[str, np.ndarray]],
    manifest_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Select representative held-out real test cases and synthetic SCC candidates."""
    samples: list[dict[str, Any]] = []
    used_indices: set[int] = set()

    real_preds = outputs_by_arm["real_only"]["y_pred"]
    real_probs = outputs_by_arm["real_only"]["probabilities"]
    cgan_preds = outputs_by_arm["cgan_scc"]["y_pred"]
    y_true = outputs_by_arm["real_only"]["y_true"]

    def add_real(idx: int, sample_id: str, category: str, note: str) -> None:
        if idx in used_indices:
            return
        row = test_df.iloc[idx]
        img_path = Path(str(row["image_path"]))
        if not img_path.exists():
            return
        used_indices.add(idx)
        samples.append({
            "sample_id": sample_id,
            "source": "real_test",
            "category": category,
            "note": note,
            "image_name": str(row["image"]),
            "rel_path": str(img_path.relative_to(PROJECT_DIR)),
            "ground_truth": str(row["diagnosis"]),
            "ground_truth_label": int(row["label"]),
            "lesion_id": str(row["group_id"]),
            "age_approx": None if pd.isna(row.get("age_approx")) else int(row["age_approx"]),
            "sex": None if pd.isna(row.get("sex")) else str(row["sex"]),
            "anatom_site": None if pd.isna(row.get("anatom_site_general")) else str(row["anatom_site_general"]),
            "real_only_pred": ACTIVE_CLASS_NAMES[int(real_preds[idx])],
            "cgan_scc_pred": ACTIVE_CLASS_NAMES[int(cgan_preds[idx])],
            "real_only_scc_prob": round(float(real_probs[idx, TARGET_LABEL]), 4),
        })

    # 1. SCC True Positives sorted by real_only confidence
    scc_tp = [i for i in range(len(test_df)) if y_true[i] == TARGET_LABEL and real_preds[i] == TARGET_LABEL]
    scc_tp.sort(key=lambda i: float(real_probs[i, TARGET_LABEL]), reverse=True)
    for rank, idx in enumerate(scc_tp[:4], start=1):
        add_real(idx, f"real_scc_tp_{rank:02d}", "SCC True Positive", "Held-out SCC lesion correctly identified by baseline.")

    # 2. SCC Disagreements / Confusions (e.g. missed by cgan_scc or confused with AK/BCC)
    scc_disagree = [
        i for i in range(len(test_df))
        if y_true[i] == TARGET_LABEL and (real_preds[i] != cgan_preds[i] or real_preds[i] != TARGET_LABEL)
    ]
    scc_disagree.sort(key=lambda i: abs(float(real_probs[i, TARGET_LABEL]) - 0.35))
    for rank, idx in enumerate(scc_disagree[:4], start=1):
        pred_lbl = ACTIVE_CLASS_NAMES[int(real_preds[idx])]
        add_real(
            idx,
            f"real_scc_edge_{rank:02d}",
            "SCC Challenge Case",
            f"Held-out SCC lesion (baseline predicted {pred_lbl}); highlights boundary ambiguity.",
        )

    # 3. Representative lesions from the other 7 active ISIC diagnoses
    for cls_idx, diag in enumerate(ACTIVE_CLASS_NAMES):
        if cls_idx == TARGET_LABEL:
            continue
        cls_correct = [i for i in range(len(test_df)) if y_true[i] == cls_idx and real_preds[i] == cls_idx]
        cls_correct.sort(key=lambda i: float(real_probs[i, cls_idx]), reverse=True)
        chosen = cls_correct[:2] if len(cls_correct) >= 2 else [i for i in range(len(test_df)) if y_true[i] == cls_idx][:2]
        for rank, idx in enumerate(chosen, start=1):
            add_real(
                idx,
                f"real_{diag.lower()}_{rank:02d}",
                f"Held-out {diag}",
                f"Real test {DIAGNOSIS_FULL_NAMES[diag]} image.",
            )

    # 4. Accepted synthetic SCC candidates across diverse pHash distances
    accepted = manifest_df[manifest_df["accepted"] == True].copy()  # noqa: E712
    if not accepted.empty:
        pick_positions = np.linspace(0, min(len(accepted) - 1, 599), 8, dtype=int)
        for rank, pos in enumerate(np.unique(pick_positions), start=1):
            row = accepted.iloc[int(pos)]
            img_path = Path(str(row["image_path"]))
            if not img_path.exists():
                continue
            samples.append({
                "sample_id": f"synth_scc_{rank:02d}",
                "source": "synthetic_cgan",
                "category": "Synthetic cGAN SCC",
                "note": (
                    f"128px cGAN candidate (real pHash dist={int(row['min_real_phash_distance'])}, "
                    f"synth dist={int(row['min_synthetic_phash_distance'])})."
                ),
                "image_name": img_path.stem,
                "rel_path": str(img_path.relative_to(PROJECT_DIR)),
                "ground_truth": "SCC (Synthetic)",
                "ground_truth_label": TARGET_LABEL,
                "lesion_id": "synthetic_generator",
                "age_approx": None,
                "sex": None,
                "anatom_site": None,
                "real_only_pred": None,
                "cgan_scc_pred": None,
                "real_only_scc_prob": None,
            })
    return samples


def build_evaluation_cache(output_path: Path = CACHE_PATH) -> dict[str, Any]:
    """Evaluate saved checkpoints on the held-out real test split and save UI cache JSON."""
    train_df = pd.read_csv(ARTIFACT_DIR / "train_split.csv")
    val_df = pd.read_csv(ARTIFACT_DIR / "val_split.csv")
    test_df = pd.read_csv(ARTIFACT_DIR / "test_split.csv")
    manifest_df = pd.read_csv(SYNTHETIC_DIR / "screening_manifest.csv")
    cgan_loss_df = pd.read_csv(CGAN_DIR / "loss_history.csv")

    device = select_device()
    _, eval_transform = classifier_transforms(224)
    test_loader = balanced_loader(test_df, eval_transform, batch_size=32, seed=SEED, shuffle=False)

    outputs_by_arm: dict[str, dict[str, np.ndarray]] = {}
    arm_results: dict[str, dict[str, Any]] = {}
    test_binary = (test_df["label"].to_numpy() == TARGET_LABEL).astype(int)

    for arm in ARM_ORDER:
        meta = ARM_METADATA[arm]
        ckpt_path = ARTIFACT_DIR / meta["checkpoint"]
        hist_path = ARTIFACT_DIR / meta["history"]
        model = make_classifier(n_classes=N_CLASSES, pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        metrics, outputs = evaluate_classifier(model, test_loader, device, TARGET_LABEL)
        ci_low, ci_high = group_bootstrap_target_f1(
            outputs, test_df["group_id"].to_numpy(), TARGET_LABEL, rounds=1000, seed=SEED
        )
        metrics["target_f1_ci_low"] = float(ci_low)
        metrics["target_f1_ci_high"] = float(ci_high)
        outputs_by_arm[arm] = outputs

        cm = confusion_matrix(
            outputs["y_true"], outputs["y_pred"], labels=list(ACTIVE_CLASS_INDICES)
        ).tolist()
        hist_df = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
        arm_results[arm] = {
            **meta,
            "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
            "per_class": _per_class_rows(outputs["y_true"], outputs["y_pred"]),
            "confusion_matrix": cm,
            "pr_curve": _downsample_pr_curve(test_binary, outputs["probabilities"][:, TARGET_LABEL]),
            "calibration": _calibration_bins(test_binary, outputs["probabilities"][:, TARGET_LABEL]),
            "training_history": hist_df.round(4).to_dict(orient="records"),
        }
        if device.type == "mps":
            torch.mps.empty_cache()

    real_scc_paths = train_df.query("label == @TARGET_LABEL")["image_path"].tolist()
    accepted_fake_paths = manifest_df.query("accepted")["image_path"].tolist()[:600]
    detector_auc = synthetic_detector_auc(real_scc_paths, accepted_fake_paths, seed=SEED)

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "target_class": TARGET_CLASS,
        "target_label": TARGET_LABEL,
        "class_names": list(ACTIVE_CLASS_NAMES),
        "diagnosis_full_names": {k: DIAGNOSIS_FULL_NAMES[k] for k in ACTIVE_CLASS_NAMES},
        "splits": {
            "train": _split_summary(train_df),
            "val": _split_summary(val_df),
            "test": _split_summary(test_df),
        },
        "arms": arm_results,
        "arm_order": list(ARM_ORDER),
        "cgan_audit": {
            "epochs_completed": int(cgan_loss_df["epoch"].max()),
            "valid_sample_epochs": list(VALID_GAN_EPOCHS),
            "loss_history": cgan_loss_df.round(5).to_dict(orient="records"),
            "candidates_total": int(len(manifest_df)),
            "candidates_accepted": int(manifest_df["accepted"].sum()),
            "candidates_rejected": int((~manifest_df["accepted"]).sum()),
            "phash_threshold": 4,
            "synthetic_detector_auc": round(float(detector_auc), 4),
            "mean_real_phash": round(float(manifest_df["min_real_phash_distance"].mean()), 2),
            "mean_synthetic_phash": round(float(manifest_df["min_synthetic_phash_distance"].mean()), 2),
            "real_phash_hist": _phash_histogram(manifest_df["min_real_phash_distance"]),
            "synthetic_phash_hist": _phash_histogram(manifest_df["min_synthetic_phash_distance"]),
        },
        "samples": _select_curated_samples(test_df, outputs_by_arm, manifest_df),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return payload


def load_evaluation_cache(cache_path: Path = CACHE_PATH) -> dict[str, Any]:
    """Load precomputed cache from disk, building it if absent."""
    if not cache_path.exists():
        return build_evaluation_cache(cache_path)
    return json.loads(cache_path.read_text())
