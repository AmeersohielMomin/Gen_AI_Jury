"""Train Arm 3 (cgan_scc) and compute unified 3-arm evaluation for Topic 146.

Preserves existing checkpoints:
  - artifacts/cgan/cgan_latest.pt
  - artifacts/real_only_efficientnet_b0.pt
  - artifacts/repeated_real_scc_efficientnet_b0.pt
"""

import copy
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

from src.medical_cgan import (
    balanced_loader,
    build_training_arms,
    classifier_transforms,
    configure_mps_memory,
    evaluate_classifier,
    export_blinded_realism_panel,
    group_bootstrap_target_f1,
    make_classifier,
    seed_everything,
    select_device,
    train_classifier,
)

# ----------------- Configuration -----------------
SEED = 146
TARGET_LABEL = 7  # 'SCC' in ISIC 2019
TARGET_CLASS = "SCC"
N_SYNTHETIC_SCC = 600
CLASSIFIER_BATCH = 8
CLASSIFIER_EPOCHS = 12
LEARNING_RATE = 2e-4

PROJECT_DIR = Path.cwd()
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
SYNTHETIC_DIR = ARTIFACT_DIR / "synthetic_scc"
MANIFEST_PATH = SYNTHETIC_DIR / "screening_manifest.csv"

def main():
    print(f"=== Topic 146: Arm 3 Training & 3-Arm Evaluation ===", flush=True)
    seed_everything(SEED)
    device = select_device()
    configure_mps_memory(device, fraction=0.68)
    print(f"Device: {device}", flush=True)

    # 1. Load splits
    print("Loading leakage-safe splits...", flush=True)
    train_df = pd.read_csv(ARTIFACT_DIR / "train_split.csv")
    val_df = pd.read_csv(ARTIFACT_DIR / "val_split.csv")
    test_df = pd.read_csv(ARTIFACT_DIR / "test_split.csv")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}", flush=True)

    # 2. Build training arms
    print("Building training arms...", flush=True)
    synthetic_manifest = pd.read_csv(MANIFEST_PATH)
    arms = build_training_arms(train_df, synthetic_manifest, TARGET_LABEL, N_SYNTHETIC_SCC, SEED)
    arm3_frame = arms["cgan_scc"]
    print(f"Arm 3 (cgan_scc) dataset size: {len(arm3_frame)} (includes {N_SYNTHETIC_SCC} screened cGAN images)", flush=True)

    # 3. Create loaders
    train_transform, eval_transform = classifier_transforms(224)
    val_loader = balanced_loader(val_df, eval_transform, CLASSIFIER_BATCH, SEED, shuffle=False)
    test_loader = balanced_loader(test_df, eval_transform, CLASSIFIER_BATCH, SEED, shuffle=False)
    arm3_loader = balanced_loader(arm3_frame, train_transform, CLASSIFIER_BATCH, SEED, shuffle=True)

    # 4. Train Arm 3 (cgan_scc)
    cgan_scc_ckpt = ARTIFACT_DIR / "cgan_scc_efficientnet_b0.pt"
    cgan_scc_hist_path = ARTIFACT_DIR / "cgan_scc_history.csv"

    if cgan_scc_ckpt.exists() and cgan_scc_hist_path.exists():
        print(f"Found existing {cgan_scc_ckpt.name}, skipping re-training.", flush=True)
        cgan_scc_model = make_classifier(n_classes=9, pretrained=False).to(device)
        cgan_scc_model.load_state_dict(torch.load(cgan_scc_ckpt, map_location=device))
    else:
        print(f"\n--- Starting Arm 3 (cgan_scc) Training for {CLASSIFIER_EPOCHS} epochs ---", flush=True)
        seed_everything(SEED)
        cgan_scc_model = make_classifier(n_classes=9, pretrained=True)
        start_time = time.time()
        cgan_scc_model, cgan_scc_history = train_classifier(
            cgan_scc_model,
            arm3_loader,
            val_loader,
            device,
            TARGET_LABEL,
            CLASSIFIER_EPOCHS,
            learning_rate=LEARNING_RATE,
        )
        elapsed = time.time() - start_time
        print(f"\nArm 3 training completed in {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours).", flush=True)
        cgan_scc_history.to_csv(cgan_scc_hist_path, index=False)
        torch.save(cgan_scc_model.state_dict(), cgan_scc_ckpt)
        print(f"Saved checkpoint: {cgan_scc_ckpt}", flush=True)

    # 5. Evaluate all 3 arms on held-out test set
    print("\n=== Evaluating All 3 Arms on Held-Out Test Set (3,664 real lesions) ===", flush=True)
    arm_models = {
        "real_only": ARTIFACT_DIR / "real_only_efficientnet_b0.pt",
        "repeated_real_scc": ARTIFACT_DIR / "repeated_real_scc_efficientnet_b0.pt",
        "cgan_scc": cgan_scc_ckpt,
    }

    results = []
    outputs_by_arm = {}

    for arm_name, ckpt_path in arm_models.items():
        print(f"Evaluating {arm_name}...", flush=True)
        model = make_classifier(n_classes=9, pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        metrics, outputs = evaluate_classifier(model, test_loader, device, TARGET_LABEL)
        low, high = group_bootstrap_target_f1(outputs, test_df["group_id"].to_numpy(), TARGET_LABEL)
        metrics.update({
            "arm": arm_name,
            "target_f1_ci_low": low,
            "target_f1_ci_high": high,
        })
        results.append(metrics)
        outputs_by_arm[arm_name] = outputs
        print(f"  {arm_name} -> SCC Recall: {metrics['target_recall']:.3f}, SCC F1: {metrics['target_f1']:.3f} [{low:.3f}, {high:.3f}], PR-AUC: {metrics['target_pr_auc']:.3f}, Macro-F1: {metrics['macro_f1']:.3f}", flush=True)

    results_df = pd.DataFrame(results).set_index("arm").sort_index()
    comparison_csv = ARTIFACT_DIR / "classifier_comparison.csv"
    results_df.to_csv(comparison_csv)
    print(f"\nSaved final comparison table to {comparison_csv}:", flush=True)
    print(results_df[["target_recall", "target_precision", "target_f1", "target_pr_auc", "macro_f1", "balanced_accuracy"]].round(3).to_string(), flush=True)

    # 6. Generate presentation figures
    print("\nGenerating evaluation figures...", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PR curve
    ax_pr = axes[0]
    test_binary = (test_df["label"].to_numpy() == TARGET_LABEL).astype(int)
    for arm_name, outputs in outputs_by_arm.items():
        probs = outputs["probabilities"][:, TARGET_LABEL]
        PrecisionRecallDisplay.from_predictions(test_binary, probs, ax=ax_pr, name=arm_name)
    ax_pr.set_title("Test-Set Precision-Recall Curves (SCC Detection)")
    ax_pr.grid(True, linestyle="--", alpha=0.5)

    # F1 with Bootstrap CI
    ax_bar = axes[1]
    arms_list = list(results_df.index)
    f1s = results_df["target_f1"].values
    err_low = f1s - results_df["target_f1_ci_low"].values
    err_high = results_df["target_f1_ci_high"].values - f1s
    yerr = np.vstack([err_low, err_high])
    colors = ["#4c78a8", "#72b7b2", "#f58518"]
    ax_bar.bar(arms_list, f1s, yerr=yerr, capsize=6, color=colors, alpha=0.85)
    ax_bar.set_ylabel("SCC F1-Score (Held-out Test)")
    ax_bar.set_title("Target SCC F1-Score with 95% Group-Bootstrap CI")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = ARTIFACT_DIR / "jury_classifier_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved comparison figure to {plot_path}", flush=True)

    # 7. Export blinded realism panel
    print("\nExporting blinded classroom realism survey panel...", flush=True)
    real_scc_paths = train_df.query("label == @TARGET_LABEL")["image_path"].tolist()
    accepted_paths = synthetic_manifest.query("accepted")["image_path"].tolist()[:N_SYNTHETIC_SCC]
    realism_dir = ARTIFACT_DIR / "realism_survey"
    answer_key = export_blinded_realism_panel(real_scc_paths, accepted_paths, realism_dir, per_source=12, seed=SEED)
    print(f"Blinded panel images saved in {realism_dir}. Answer key: {answer_key}", flush=True)

    print("\n=== ALL TOPIC 146 EXPERIMENTS COMPLETE! ===", flush=True)

if __name__ == "__main__":
    main()
