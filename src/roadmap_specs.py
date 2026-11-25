"""Structured Future Healthcare Imaging & Architecture Roadmap data."""

from __future__ import annotations

from typing import Any


def get_roadmap_payload() -> dict[str, Any]:
    """Return structured technical specs for the GenAI Jury Future Things Hub."""
    return {
        "cgan_upgrade_diff": {
            "title": "From cGAN v1 (Trained Baseline) to GenAI v2 (Next Iteration)",
            "summary": (
                "Audit of the 80-epoch cGAN revealed discriminator hinge loss saturation (~2.000 from epoch 3), "
                "4x4 ConvTranspose2d checkerboard artifacts, and a 0.928 simple-stats real-vs-fake detector AUC. "
                "Below is the exact engineering blueprint to resolve domain shift in future runs."
            ),
            "comparisons": [
                {
                    "dimension": "Spatial Upsampling",
                    "v1_current": "nn.ConvTranspose2d(kernel=4, stride=2) — creates periodic 4x4 grid checkerboards",
                    "v2_proposed": "Bilinear Upsample(scale_factor=2) + 3x3 Conv2d + GroupNorm/BatchNorm",
                    "expected_impact": "Eliminates high-frequency grid artifacts that trigger trivial synthetic detection.",
                },
                {
                    "dimension": "Discriminator Regularization",
                    "v1_current": "Global Avg Pool + spectral_norm with N(0, 0.02) init; D-loss pinned at 2.000",
                    "v2_proposed": "Multi-scale PatchGAN / ResNet D + DiffAugment (color/cutout/translation) + R1 gradient penalty",
                    "expected_impact": "Prevents discriminator memorization & collapse under small minority class N=429.",
                },
                {
                    "dimension": "Target Resolution & Conditioning",
                    "v1_current": "128x128 RGB cropped patches upscaled to 224x224 for EfficientNet-B0",
                    "v2_proposed": "256x256 conditional StyleGAN2-ADA or Latent Diffusion (SD/UNet fine-tune) with EMA weights",
                    "expected_impact": "Preserves dermoscopic pigment networks, vascular structures, and crisp lesion borders.",
                },
                {
                    "dimension": "Validation Protocol",
                    "v1_current": "Single seed (146), single synthetic ratio (600 SCC), pHash <= 4 filter",
                    "v2_proposed": ">= 3 random seeds, FID/KID + detector AUC <= 0.65 gate before classifier mixing, external test cohort",
                    "expected_impact": "Statistically rigorous evidence before claiming minority-class clinical benefit.",
                },
            ],
        },
        "modalities": [
            {
                "id": "dermoscopy",
                "name": "Dermoscopy (Skin Lesions)",
                "status": "Active Trained Prototype",
                "badge": "LIVE",
                "formats": ["JPEG", "PNG (ISIC 2019)"],
                "preprocessing": "256px resize + 224px center crop, ImageNet normalization, lesion_id group-safe splits",
                "models": "EfficientNet-B0 (3 controlled arms) + Conditional DCGAN (128px)",
                "explainability": "Grad-CAM spatial activation overlay + 8-class calibrated probabilities",
                "target_metrics": "SCC F1 (0.354 baseline), SCC PR-AUC (0.340), Macro-F1 (0.523), 95% Group-Bootstrap CI",
                "next_milestone": "Deploy Upsample+Conv DiffAugment cGAN v2 at 256px and validate across >= 3 seeds.",
            },
            {
                "id": "xray",
                "name": "Chest & Musculoskeletal X-Ray",
                "status": "Stage 3 Planned Expansion",
                "badge": "ROADMAP",
                "formats": ["DICOM (.dcm)", "16-bit PNG"],
                "preprocessing": "pydicom windowing, CLAHE contrast normalization, PHI header scrubbing, patient-ID group split",
                "models": "DenseNet-121 / ConvNeXt multi-label classifier + Class-conditioned Diffusion/GAN augmentation",
                "explainability": "Grad-CAM++ ROI bounding boxes + uncertainty intervals + structured technical summary",
                "target_metrics": "Rare finding Sensitivity/Specificity, AUROC, Expected Calibration Error (ECE)",
                "next_milestone": "Integrate MONAI + pydicom ingestion pipeline with de-identification audit gate.",
            },
            {
                "id": "mri",
                "name": "Brain & Body MRI",
                "status": "Stage 3-4 Multi-Modal Vision",
                "badge": "ROADMAP",
                "formats": ["NIfTI (.nii/.nii.gz)", "DICOM Series"],
                "preprocessing": "SimpleITK / MONAI bias-field correction, isotropic 3D resampling, intensity z-normalization",
                "models": "3D Swin-UNETR / SegResNet segmentation + slice-conditioned synthesis for rare tumor subtypes",
                "explainability": "3D volumetric uncertainty maps, Dice boundary overlays, multi-sequence attention",
                "target_metrics": "Dice Similarity Coefficient (DSC), Hausdorff Distance 95%, false-positive volume rate",
                "next_milestone": "Prototype 2D slice browser + MONAI 3D inference worker on GPU node.",
            },
            {
                "id": "ct",
                "name": "Thoracic & Abdominal CT",
                "status": "Stage 3-4 Multi-Modal Vision",
                "badge": "ROADMAP",
                "formats": ["DICOM CT Volume", "NIfTI"],
                "preprocessing": "Hounsfield Unit (HU) multi-windowing (lung/mediastinal/bone), 3D patch tiling",
                "models": "3D nnU-Net / MONAI bundle nodule detector + minority-lesion generative stress testing",
                "explainability": "Axial/coronal/sagittal ROI slice cards + confidence score + structured preliminary report",
                "target_metrics": "FROC sensitivity at fixed false positives/scan, volumetric IoU",
                "next_milestone": "Establish multi-scanner domain generalization benchmark across hospital sites.",
            },
            {
                "id": "pathology",
                "name": "Digital Histopathology",
                "status": "Stage 4 Future Exploration",
                "badge": "FUTURE",
                "formats": ["Whole Slide Image (SVS/TIFF)", "Tile PNG"],
                "preprocessing": "Tissue foreground masking, stain normalization (Macenko), multi-instance patch tiling",
                "models": "Multiple Instance Learning (MIL) + Vision Transformer patch encoder",
                "explainability": "Top-k diagnostic tile gallery + whole-slide attention heatmap",
                "target_metrics": "Slide-level PR-AUC, inter-scanner stain invariance",
                "next_milestone": "Add federated learning simulation so sensitive WSI data never leaves institutional servers.",
            },
        ],
        "methodology_phases": [
            {"phase": "1. Discover", "activity": "Define clinical/research problem, target minority class, and benefit.", "artifact": "Problem Charter"},
            {"phase": "2. Specify", "activity": "Lock group-safe split policy, input schemas, and minority-class KPIs.", "artifact": "Experimental Protocol"},
            {"phase": "3. Prepare", "activity": "De-identify, clean, deduplicate, and enforce zero lesion-group leakage.", "artifact": "Train/Val/Test Splits"},
            {"phase": "4. Design", "activity": "Architect controlled arms (Real vs. Repeated Real vs. Synthetic).", "artifact": "Model & Ablation Spec"},
            {"phase": "5. Train", "activity": "Run hardware-capped training, checkpoint best validation Macro-F1.", "artifact": "Checkpoints & Loss Logs"},
            {"phase": "6. Validate", "activity": "Evaluate held-out real test set with 95% group-bootstrap CIs & detector AUC.", "artifact": "Verified Audit Table"},
            {"phase": "7. Explain", "activity": "Attach Grad-CAM spatial overlays, calibration bins, and failure cases.", "artifact": "Interpretable UI Cards"},
            {"phase": "8. Implement", "activity": "Package FastAPI + interactive Jury Lab + preliminary report generator.", "artifact": "Demonstrable Web App"},
            {"phase": "9. Improve", "activity": "Upgrade cGAN v2 (Upsample+DiffAugment), multi-seed runs, external validation.", "artifact": "Next-Release Roadmap"},
        ],
        "hardware_tiers": [
            {
                "tier": "Current Prototype (8 GB Mac MPS)",
                "specs": "Apple Silicon Unified Memory (5.33 GB recommended working set, 68% cap), Batch 8",
                "capabilities": "Fast Grad-CAM inference (<0.3s/image), lightweight 128px prototyping",
                "limitations": "80-epoch cGAN took ~8.7 hrs; 3-arm classifiers took ~2.5-5.5 hrs each; batch size constrained",
            },
            {
                "tier": "Workstation (NVIDIA RTX 16-24 GB VRAM)",
                "specs": "CUDA + cuDNN + AMP (fp16/bf16), Batch 32-64",
                "capabilities": "256px DiffAugment cGAN v2 or Diffusion LoRA in <2 hrs, 3-seed validation sweeps",
                "limitations": "Requires local driver/CUDA setup and thermal headroom",
            },
            {
                "tier": "Cloud / Colab GPU (A100 / L4)",
                "specs": "24-80 GB VRAM, fast NVMe caching",
                "capabilities": "High-res multi-modal MRI/CT 3D training, large hyperparameter sweeps",
                "limitations": "Public/unencrypted cloud nodes must never receive non-anonymized patient imaging",
            },
            {
                "tier": "Hospital / Institutional On-Prem",
                "specs": "Air-gapped / HIPAA-GDPR compliant GPU cluster + Docker + Federated Learning",
                "capabilities": "External multi-site validation, clinician-in-the-loop PACS pilot",
                "limitations": "Requires IRB/ethics approval, IT integration, and regulatory clearance",
            },
        ],
        "responsible_ai_controls": [
            {"area": "Zero Data Leakage", "status": "Verified Active", "detail": "StratifiedGroupKFold by lesion_id prevents same-lesion images crossing train/val/test."},
            {"area": "No Synthetic Eval Contamination", "status": "Verified Active", "detail": "Validation (3,044) and test (3,664) splits contain strictly 100% real lesions."},
            {"area": "Shortcut & Domain-Shift Detection", "status": "Verified Active", "detail": "Simple-stats real-vs-fake detector flagged 0.928 AUC and triggered honest rejection of cGAN v1 benefit."},
            {"area": "Visual Explainability & Uncertainty", "status": "Active in UI", "detail": "Grad-CAM spatial overlays + multi-arm agreement + confidence score shown on every inference."},
            {"area": "Non-Diagnostic Guardrail", "status": "Active in UI", "detail": "Persistent research disclaimer and mandatory qualified clinician review requirement on all reports."},
            {"area": "Demographic & Site Bias Audit", "status": "Planned v2", "detail": "Stratified error analysis by age_approx, sex, and anatom_site_general across external cohorts."},
        ],
    }
