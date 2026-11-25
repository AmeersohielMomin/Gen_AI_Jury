"""Shared constants and path definitions for the GenAI Jury application."""

from __future__ import annotations

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
CGAN_DIR = ARTIFACT_DIR / "cgan"
SYNTHETIC_DIR = ARTIFACT_DIR / "synthetic_scc"
REALISM_DIR = ARTIFACT_DIR / "realism_survey"
CACHE_PATH = ARTIFACT_DIR / "jury_eval_cache.json"

SEED = 146
N_CLASSES = 9
TARGET_CLASS = "SCC"
TARGET_LABEL = 7

# Matches ISIC_2019_Training_GroundTruth.csv column order (index 8 is UNK, unused in training splits)
CLASS_NAMES: tuple[str, ...] = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK")
ACTIVE_CLASS_INDICES: tuple[int, ...] = tuple(range(8))
ACTIVE_CLASS_NAMES: tuple[str, ...] = CLASS_NAMES[:8]

DIAGNOSIS_FULL_NAMES: dict[str, str] = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion",
    "SCC": "Squamous Cell Carcinoma (Target)",
    "UNK": "Unknown",
}

ARM_METADATA: dict[str, dict[str, str]] = {
    "real_only": {
        "label": "Real Only (Baseline)",
        "short": "Real Only",
        "checkpoint": "real_only_efficientnet_b0.pt",
        "history": "real_only_history.csv",
        "description": "Class-balanced sampling over the 18,623 real training images (429 real SCC).",
    },
    "repeated_real_scc": {
        "label": "Repeated Real SCC (Control)",
        "short": "Repeated Real",
        "checkpoint": "repeated_real_scc_efficientnet_b0.pt",
        "history": "repeated_real_scc_history.csv",
        "description": "Adds 600 resampled real SCC records to isolate repetition frequency from diversity.",
    },
    "cgan_scc": {
        "label": "cGAN SCC (Augmented)",
        "short": "cGAN SCC",
        "checkpoint": "cgan_scc_efficientnet_b0.pt",
        "history": "cgan_scc_history.csv",
        "description": "Adds 600 pHash-screened 128px cGAN-generated SCC images alongside real training data.",
    },
}

ARM_ORDER: tuple[str, ...] = ("real_only", "repeated_real_scc", "cgan_scc")
VALID_GAN_EPOCHS: tuple[int, ...] = (1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80)
