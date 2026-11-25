"""Live EfficientNet-B0 inference, Grad-CAM explainability, and cGAN sampling."""

from __future__ import annotations

import base64
import io
import threading
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.constants import (
    ACTIVE_CLASS_INDICES,
    ACTIVE_CLASS_NAMES,
    ARM_METADATA,
    ARM_ORDER,
    ARTIFACT_DIR,
    CGAN_DIR,
    DIAGNOSIS_FULL_NAMES,
    N_CLASSES,
    TARGET_LABEL,
)
from src.medical_cgan import (
    ConditionalGenerator,
    classifier_transforms,
    make_classifier,
    select_device,
)


def _pil_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_cam_images(base_rgb: Image.Image, cam_map: np.ndarray, alpha: float = 0.45) -> dict[str, str]:
    """Render pure Grad-CAM heatmap and blended overlay PNGs from a [0, 1] spatial map."""
    cmap = matplotlib.colormaps["inferno"]
    rgba = (cmap(np.clip(cam_map, 0.0, 1.0)) * 255.0).astype(np.uint8)
    heatmap_rgb = Image.fromarray(rgba[:, :, :3], mode="RGB")
    overlay_rgb = Image.blend(base_rgb, heatmap_rgb, alpha=alpha)
    return {
        "heatmap_png": _pil_to_base64_png(heatmap_rgb),
        "overlay_png": _pil_to_base64_png(overlay_rgb),
    }


def _center_crop_for_display(image: Image.Image, size: int = 224) -> Image.Image:
    """Match classifier_transforms(224) evaluation crop so Grad-CAM aligns pixel-for-pixel."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    if w <= h:
        new_w = 256
        new_h = max(256, int(round(h * (256 / w))))
    else:
        new_h = 256
        new_w = max(256, int(round(w * (256 / h))))
    resized = rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return resized.crop((left, top, left + size, top + size))


def _compute_gradcam_map(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run forward + backward with retain_grad() on EfficientNet-B0 final feature block."""
    model.zero_grad(set_to_none=True)
    features = model.features(input_tensor)
    features.retain_grad()
    pooled = model.avgpool(features)
    flattened = torch.flatten(pooled, 1)
    logits = model.classifier(flattened)
    probs = torch.softmax(logits, dim=1)

    score = logits[0, target_class]
    score.backward()

    grads = features.grad
    if grads is None:
        raise RuntimeError("Failed to capture gradients on EfficientNet-B0 features.")

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * features).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam_np = cam.squeeze().detach().cpu().numpy()
    max_val = float(cam_np.max())
    if max_val > 1e-8:
        cam_np = (cam_np - float(cam_np.min())) / (max_val - float(cam_np.min()) + 1e-8)
    else:
        cam_np = np.zeros((224, 224), dtype=np.float32)
    return probs[0].detach().cpu().numpy(), cam_np


class ModelRegistry:
    """Lazy thread-safe registry for EfficientNet-B0 classifiers and ConditionalGenerator."""

    def __init__(self, artifact_dir: Path = ARTIFACT_DIR, cgan_dir: Path = CGAN_DIR) -> None:
        self.artifact_dir = artifact_dir
        self.cgan_dir = cgan_dir
        self.device = select_device()
        _, self.eval_transform = classifier_transforms(224)
        self._lock = threading.Lock()
        self._classifiers: dict[str, torch.nn.Module] = {}
        self._generator: ConditionalGenerator | None = None

    def get_classifier(self, arm: str) -> torch.nn.Module:
        if arm not in ARM_METADATA:
            raise ValueError(f"Unknown arm: {arm}")
        if arm not in self._classifiers:
            ckpt = self.artifact_dir / ARM_METADATA[arm]["checkpoint"]
            model = make_classifier(n_classes=N_CLASSES, pretrained=False).to(self.device)
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.eval()
            self._classifiers[arm] = model
        return self._classifiers[arm]

    def get_generator(self) -> ConditionalGenerator:
        if self._generator is None:
            ckpt = torch.load(self.cgan_dir / "cgan_latest.pt", map_location=self.device)
            gen = ConditionalGenerator(n_classes=N_CLASSES, z_dim=128, base=32).to(self.device)
            gen.load_state_dict(ckpt["generator"])
            gen.eval()
            self._generator = gen
        return self._generator

    def analyze_image(
        self,
        image: Image.Image,
        arms: list[str] | None = None,
        gradcam_mode: str = "predicted",
    ) -> dict[str, Any]:
        """Run multi-arm inference and Grad-CAM attention visualization on one image."""
        chosen_arms = [a for a in (arms or list(ARM_ORDER)) if a in ARM_METADATA]
        if not chosen_arms:
            chosen_arms = list(ARM_ORDER)

        rgb = image.convert("RGB")
        display_crop = _center_crop_for_display(rgb, size=224)
        original_b64 = _pil_to_base64_png(display_crop)
        tensor = self.eval_transform(rgb).unsqueeze(0).to(self.device)

        arm_outputs: dict[str, Any] = {}
        with self._lock:
            for arm in chosen_arms:
                model = self.get_classifier(arm)
                with torch.no_grad():
                    logits = model(tensor)
                    probs_arr = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

                active_probs = probs_arr[list(ACTIVE_CLASS_INDICES)]
                pred_idx = int(np.argmax(active_probs))
                cam_target = TARGET_LABEL if gradcam_mode == "scc" else pred_idx

                _, cam_map = _compute_gradcam_map(model, tensor, target_class=cam_target)
                cam_imgs = _render_cam_images(display_crop, cam_map)

                ranked = [
                    {
                        "diagnosis": ACTIVE_CLASS_NAMES[i],
                        "full_name": DIAGNOSIS_FULL_NAMES[ACTIVE_CLASS_NAMES[i]],
                        "probability": round(float(probs_arr[i]), 4),
                    }
                    for i in np.argsort(-active_probs)
                ]
                arm_outputs[arm] = {
                    "arm": arm,
                    "label": ARM_METADATA[arm]["label"],
                    "short": ARM_METADATA[arm]["short"],
                    "predicted_diagnosis": ACTIVE_CLASS_NAMES[pred_idx],
                    "predicted_full_name": DIAGNOSIS_FULL_NAMES[ACTIVE_CLASS_NAMES[pred_idx]],
                    "predicted_confidence": round(float(probs_arr[pred_idx]), 4),
                    "scc_probability": round(float(probs_arr[TARGET_LABEL]), 4),
                    "gradcam_target": ACTIVE_CLASS_NAMES[cam_target],
                    "probabilities": ranked,
                    **cam_imgs,
                }
            if self.device.type == "mps":
                torch.mps.empty_cache()

        preds = [v["predicted_diagnosis"] for v in arm_outputs.values()]
        consensus = len(set(preds)) == 1
        max_scc_prob = max(v["scc_probability"] for v in arm_outputs.values())
        return {
            "device": str(self.device),
            "gradcam_mode": gradcam_mode,
            "original_png": original_b64,
            "arms": arm_outputs,
            "arm_consensus": consensus,
            "consensus_diagnosis": preds[0] if consensus else None,
            "max_scc_probability": round(float(max_scc_prob), 4),
        }

    def sample_cgan(self, diagnosis: str, seed: int = 146, count: int = 4) -> dict[str, Any]:
        """Generate conditional 128x128 synthetic samples from the saved 80-epoch cGAN."""
        diag_upper = diagnosis.upper()
        if diag_upper not in ACTIVE_CLASS_NAMES:
            raise ValueError(f"Unsupported diagnosis '{diagnosis}'. Choose from {ACTIVE_CLASS_NAMES}.")
        label_idx = ACTIVE_CLASS_NAMES.index(diag_upper)
        n_samples = max(1, min(int(count), 8))

        with self._lock:
            gen = self.get_generator()
            rng_device = "cpu"
            rng = torch.Generator(device=rng_device).manual_seed(int(seed))
            noise = torch.randn((n_samples, gen.z_dim), generator=rng, device=rng_device).to(self.device)
            labels = torch.full((n_samples,), label_idx, dtype=torch.long, device=self.device)
            with torch.no_grad():
                imgs = gen(noise, labels).detach().cpu()

        rendered = []
        for idx, tensor_img in enumerate(imgs):
            arr = ((tensor_img.clamp(-1.0, 1.0) + 1.0) * 0.5).permute(1, 2, 0).numpy()
            pil_img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")
            edge_h = float(np.abs(np.diff(arr, axis=1)).mean())
            edge_v = float(np.abs(np.diff(arr, axis=0)).mean())
            rendered.append({
                "index": idx + 1,
                "image_png": _pil_to_base64_png(pil_img),
                "mean_rgb": [round(float(x), 3) for x in arr.mean(axis=(0, 1))],
                "checkerboard_gradient_score": round((edge_h + edge_v) / 2.0, 4),
            })

        return {
            "diagnosis": diag_upper,
            "full_name": DIAGNOSIS_FULL_NAMES[diag_upper],
            "seed": int(seed),
            "resolution": "128x128",
            "samples": rendered,
        }
