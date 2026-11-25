# Topic 146 — GenAI Jury: Conditional GAN for Skin-Lesion Augmentation

This project tests whether cGAN-generated **squamous cell carcinoma (SCC)** images improve a skin-lesion classifier on a held-out set of real lesions, packaged with an interactive **FastAPI Jury Studio UI**, live **Grad-CAM explainability**, and a **Future Healthcare Imaging Roadmap**.

## Controlled 3-Arm Experiment & Verified Results

Evaluated on **3,664 held-out real ISIC 2019 lesions** (8 active diagnoses, zero lesion-group leakage via `StratifiedGroupKFold` seed `146`):

| Arm | Extra SCC Images | SCC Recall | SCC Precision | SCC F1 [95% CI] | SCC PR-AUC | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Real only (Baseline)** | 0 | 0.311 | 0.410 | **0.354** [0.230, 0.466] | **0.340** | **0.523** |
| **Repeated real SCC (Control)** | 600 resampled real | 0.282 | 0.337 | 0.307 [0.194, 0.413] | 0.246 | 0.500 |
| **cGAN SCC (Augmented)** | 600 screened fakes | 0.272 | 0.354 | 0.308 [0.193, 0.422] | 0.258 | 0.505 |

### cGAN Diagnostic Audit

- **Why cGAN v1 did not beat the baseline:** Over 80 epochs (`artifacts/cgan/loss_history.csv`), discriminator hinge loss pinned near `2.000` from epoch 3 onward, and `ConvTranspose2d` layers produced visible 4×4 checkerboard artifacts (`artifacts/cgan/samples_epoch_080.png`).
- **Domain shift signal:** Although 963/1,200 candidates passed perceptual-hash deduplication (`pHash > 4`), a simple 12-feature color/edge `LogisticRegression` detector separates real vs. synthetic SCC with **0.928 ROC-AUC**.
- **Future roadmap (`v2`):** Upgrade `ConvTranspose2d` to `Upsample + Conv2d`, add `DiffAugment` + R1 regularization, evaluate across ≥3 seeds, and expand toward multi-modal MRI/CT/X-ray/pathology workflows (`GenAI_Jury_Professional_Report.docx`).

## Dataset

Download the ISIC 2019 training images and ground-truth CSV from the [ISIC Challenge data page](https://challenge.isic-archive.com/data/) (~9.1 GB, licensed CC-BY-NC):

```text
data/isic2019/
├── ISIC_2019_Training_Input/
│   ├── ISIC_0000000.jpg
│   └── ...
├── ISIC_2019_Training_GroundTruth.csv
└── ISIC_2019_Training_Metadata.csv
```

## Launch the GenAI Jury Web UI

1. Install dependencies:
   ```bash
   python3 -m pip install -r requirements.txt
   ```
2. Build or verify the precomputed evaluation cache (if not already present):
   ```bash
   python3 -m scripts.build_eval_cache
   ```
3. Start the interactive Jury Studio:
   ```bash
   uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
   Open `http://127.0.0.1:8000` in your browser to explore:
   - **Overview**: Class imbalance & 3-arm verdict
   - **Live Analyzer & Grad-CAM**: Upload or test held-out lesions across all 3 EfficientNet-B0 models with spatial heatmaps & printable preliminary reports
   - **Interactive Jury Lab**: PR curves, bootstrap CIs, 8×8 confusion matrices, Epoch 1–80 scrubber, live cGAN conditional sampler, and a blinded 24-panel realism quiz
   - **Future Roadmap**: cGAN v1 → v2 architecture diff, MRI/CT/X-ray/Pathology workflow specs, hardware sizing, and Responsible AI controls

4. Run the automated verification suite:
   ```bash
   python3 -m pytest tests/test_app.py -v
   ```

## Notebook & Training Scripts

- `topic_146_cgan_skin_lesions.ipynb` — Full reproducible notebook pipeline.
- `train_arm3_and_evaluate.py` — Standalone Arm 3 (`cgan_scc`) + 3-arm evaluation runner.

## Safety and Interpretation

This is a classroom research prototype, **not a diagnostic medical device**. A visual-realism survey and perceptual-hash checks are quality screens, not clinical validation. Generated images can encode acquisition artifacts and domain shifts; any clinical deployment requires external validation, regulatory review, and qualified clinician oversight.
