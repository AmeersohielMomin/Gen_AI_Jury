"""Reusable, MPS-friendly helpers for the Topic 146 skin-lesion experiment.

The module deliberately separates split creation, cGAN training, synthetic-image
screening, and classifier evaluation. That makes the comparison auditable: fakes
cannot enter validation/test data, and the repeated-real-SCC control can be run
under identical classifier settings.
"""

from __future__ import annotations

import copy
import csv
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch import Tensor, nn
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm


def seed_everything(seed: int = 146) -> None:
    """Set reproducible random seeds for the notebook's experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def select_device() -> torch.device:
    """Prefer Metal/MPS, then CUDA, then CPU; print the chosen backend."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def configure_mps_memory(device: torch.device, fraction: float = 0.68) -> None:
    """Reserve unified memory for macOS rather than filling an 8 GB machine."""
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(fraction)
        max_mem = torch.mps.recommended_max_memory() / 1024**3
        print(f"MPS recommended working set: {max_mem:.2f} GB; cap: {fraction:.0%}")


def validate_isic_layout(image_dir: Path, labels_csv: Path, metadata_csv: Path) -> None:
    missing = [str(p) for p in (image_dir, labels_csv, metadata_csv) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "ISIC files are missing. Download the ISIC 2019 training image archive, "
            "ground-truth CSV, and metadata CSV, then update the notebook paths:\n- "
            + "\n- ".join(missing)
        )


def load_isic_2019(
    image_dir: Path, labels_csv: Path, metadata_csv: Path
) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
    """Load one-hot ISIC labels and attach safe image and lesion-group fields."""
    labels = pd.read_csv(labels_csv)
    metadata = pd.read_csv(metadata_csv)
    if "image" not in labels or "image" not in metadata:
        raise ValueError("Both ISIC CSVs must contain an 'image' column.")

    class_columns = [c for c in labels.columns if c != "image"]
    if not class_columns:
        raise ValueError("No diagnosis columns found in the ground-truth CSV.")

    frame = labels.merge(metadata, on="image", how="left", validate="one_to_one")
    frame["diagnosis"] = frame[class_columns].idxmax(axis=1)
    frame["label"] = frame["diagnosis"].map({name: i for i, name in enumerate(class_columns)})
    frame["image_path"] = frame["image"].map(lambda name: str(image_dir / f"{name}.jpg"))
    missing_images = (~frame["image_path"].map(lambda p: Path(p).exists())).sum()
    if missing_images:
        raise FileNotFoundError(f"{missing_images} listed image files were not found in {image_dir}.")

    # lesion_id is preferable. When unavailable, an image is its own group: safe but weaker.
    group_field = "lesion_id" if "lesion_id" in frame.columns else "image"
    frame["group_id"] = frame[group_field].fillna(frame["image"]).astype(str)
    label_to_index = {name: i for i, name in enumerate(class_columns)}
    index_to_label = {i: name for name, i in label_to_index.items()}
    return frame, label_to_index, index_to_label


def group_safe_split(frame: pd.DataFrame, seed: int = 146) -> dict[str, pd.DataFrame]:
    """Make approximately 71/14/14 stratified splits without lesion-group leakage."""
    def one_fold(data: pd.DataFrame, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
        cv = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=split_seed)
        return next(cv.split(data, y=data["label"], groups=data["group_id"]))

    train_val_idx, test_idx = one_fold(frame, seed)
    train_val = frame.iloc[train_val_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)
    train_idx, val_idx = one_fold(train_val, seed + 1)
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)
    assert_no_group_leakage(train, val, test)
    return {"train": train, "val": val, "test": test}


def assert_no_group_leakage(*frames: pd.DataFrame) -> None:
    names = ["train", "validation", "test"]
    groups = [set(frame["group_id"].astype(str)) for frame in frames]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            overlap = groups[i] & groups[j]
            if overlap:
                raise AssertionError(f"Group leakage between {names[i]} and {names[j]}: {len(overlap)} groups")


def class_count_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["diagnosis", "label"]).size().rename("images").reset_index()
        .sort_values("images", ascending=False)
    )


class ImageFrameDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
            return self.transform(image), int(row["label"])


def gan_transform(size: int = 128) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def classifier_transforms(size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    train = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(size, scale=(0.82, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    evaluate = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return train, evaluate


def balanced_loader(
    frame: pd.DataFrame,
    transform: transforms.Compose,
    batch_size: int,
    seed: int,
    shuffle: bool = False,
) -> DataLoader:
    """Balance labels in all classifier arms, isolating diversity from frequency."""
    dataset = ImageFrameDataset(frame, transform)
    if not shuffle:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    count = frame["label"].value_counts()
    weights = frame["label"].map(lambda label: 1.0 / count[label]).to_numpy(dtype=float)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False, num_workers=0)


class ConditionalGenerator(nn.Module):
    """Compact 128px conditional DCGAN generator suitable for small MPS memory."""
    def __init__(self, n_classes: int, z_dim: int = 128, base: int = 32, embed_dim: int = 32):
        super().__init__()
        self.z_dim = z_dim
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.project = nn.Sequential(
            nn.Linear(z_dim + embed_dim, base * 16 * 4 * 4),
            nn.BatchNorm1d(base * 16 * 4 * 4),
            nn.ReLU(True),
        )
        self.net = nn.Sequential(
            nn.Unflatten(1, (base * 16, 4, 4)),
            nn.ConvTranspose2d(base * 16, base * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(base * 8), nn.ReLU(True),
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(base * 4), nn.ReLU(True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(base * 2), nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False), nn.BatchNorm2d(base), nn.ReLU(True),
            nn.ConvTranspose2d(base, 3, 4, 2, 1, bias=False), nn.Tanh(),
        )

    def forward(self, noise: Tensor, labels: Tensor) -> Tensor:
        conditioned = torch.cat([noise, self.embedding(labels)], dim=1)
        return self.net(self.project(conditioned))


class ProjectionDiscriminator(nn.Module):
    """Spectral-normalised projection discriminator with label conditioning."""
    def __init__(self, n_classes: int, base: int = 32):
        super().__init__()
        feature_dim = base * 16
        self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(3, base, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base, base * 2, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base * 2, base * 4, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base * 4, base * 8, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base * 8, feature_dim, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True),
        )
        self.unconditional = spectral_norm(nn.Linear(feature_dim, 1))
        self.embedding = nn.Embedding(n_classes, feature_dim)

    def forward(self, image: Tensor, labels: Tensor) -> Tensor:
        features = self.features(image).mean(dim=(2, 3))
        return self.unconditional(features).squeeze(1) + (features * self.embedding(labels)).sum(dim=1)


def init_gan_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def train_cgan(
    generator: ConditionalGenerator,
    discriminator: ProjectionDiscriminator,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    output_dir: Path,
    sample_labels: Iterable[int],
    sample_every: int = 5,
    lr: float = 2e-4,
) -> pd.DataFrame:
    """Train a conditional hinge-loss GAN and save reproducible image grids/checkpoints."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generator, discriminator = generator.to(device), discriminator.to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.9))
    fixed_labels = torch.tensor(list(sample_labels), dtype=torch.long, device=device)
    fixed_noise = torch.randn(len(fixed_labels), generator.z_dim, device=device)
    rows: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        g_losses, d_losses = [], []
        for real, labels in tqdm(loader, desc=f"cGAN {epoch}/{epochs}", leave=False):
            real, labels = real.to(device), labels.to(device)
            batch_size = real.shape[0]

            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake = generator(torch.randn(batch_size, generator.z_dim, device=device), labels)
            d_real = discriminator(real, labels)
            d_fake = discriminator(fake, labels)
            d_loss = torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            fake = generator(torch.randn(batch_size, generator.z_dim, device=device), labels)
            g_loss = -discriminator(fake, labels).mean()
            g_loss.backward()
            opt_g.step()
            d_losses.append(float(d_loss.detach().cpu()))
            g_losses.append(float(g_loss.detach().cpu()))

        rows.append({"epoch": epoch, "generator_loss": np.mean(g_losses), "discriminator_loss": np.mean(d_losses)})
        if epoch == 1 or epoch % sample_every == 0 or epoch == epochs:
            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_noise, fixed_labels).detach().cpu()
            save_image(samples, output_dir / f"samples_epoch_{epoch:03d}.png", nrow=min(8, len(samples)), normalize=True, value_range=(-1, 1))
            torch.save({"epoch": epoch, "generator": generator.state_dict(), "discriminator": discriminator.state_dict()}, output_dir / "cgan_latest.pt")
        if device.type == "mps":
            torch.mps.empty_cache()
    return pd.DataFrame(rows)


def generate_candidates(
    generator: ConditionalGenerator,
    target_label: int,
    number: int,
    device: torch.device,
    output_dir: Path,
    batch_size: int = 8,
) -> list[Path]:
    """Generate labelled candidates; quality screening happens separately."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.eval().to(device)
    paths: list[Path] = []
    with torch.no_grad():
        for start in tqdm(range(0, number, batch_size), desc="Generating SCC candidates"):
            current = min(batch_size, number - start)
            labels = torch.full((current,), target_label, dtype=torch.long, device=device)
            images = generator(torch.randn(current, generator.z_dim, device=device), labels).detach().cpu()
            for offset, image in enumerate(images):
                path = output_dir / f"synthetic_scc_{start + offset:05d}.png"
                save_image(image, path, normalize=True, value_range=(-1, 1))
                paths.append(path)
    return paths


def _phash(path: str | Path) -> imagehash.ImageHash:
    with Image.open(path) as image:
        return imagehash.phash(image.convert("RGB"))


def screen_near_duplicates(
    real_paths: Iterable[str | Path], candidate_paths: Iterable[str | Path], threshold: int = 4
) -> pd.DataFrame:
    """Reject candidates close to real SCC or previously accepted synthetic images."""
    real_hashes = [_phash(path) for path in tqdm(list(real_paths), desc="Hashing real SCC")]
    records = []
    accepted_hashes: list[imagehash.ImageHash] = []
    for path in tqdm(list(candidate_paths), desc="Screening synthetic SCC"):
        candidate_hash = _phash(path)
        real_distance = min(candidate_hash - real_hash for real_hash in real_hashes)
        synthetic_distance = min((candidate_hash - accepted_hash for accepted_hash in accepted_hashes), default=64)
        accepted = real_distance > threshold and synthetic_distance > threshold
        if accepted:
            accepted_hashes.append(candidate_hash)
        records.append(
            {
                "image_path": str(path),
                "min_real_phash_distance": real_distance,
                "min_synthetic_phash_distance": synthetic_distance,
                "accepted": accepted,
            }
        )
    return pd.DataFrame(records)


def _simple_image_features(paths: Iterable[str | Path]) -> np.ndarray:
    values = []
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
        mean, std = array.mean(axis=(0, 1)), array.std(axis=(0, 1))
        horizontal = np.abs(np.diff(array, axis=1)).mean(axis=(0, 1))
        vertical = np.abs(np.diff(array, axis=0)).mean(axis=(0, 1))
        values.append(np.concatenate([mean, std, horizontal, vertical]))
    return np.asarray(values)


def synthetic_detector_auc(real_paths: list[str], fake_paths: list[str], seed: int = 146) -> float:
    """Warning signal: high AUC means simple image statistics distinguish fake from real."""
    paths = real_paths + fake_paths
    labels = np.array([0] * len(real_paths) + [1] * len(fake_paths))
    if min(Counter(labels).values()) < 10:
        return float("nan")
    features = _simple_image_features(paths)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, stratify=labels, random_state=seed)
    model = LogisticRegression(max_iter=1000, random_state=seed).fit(x_train, y_train)
    return float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))


def build_training_arms(
    train: pd.DataFrame, synthetic_manifest: pd.DataFrame, target_label: int, n_synthetic: int, seed: int
) -> dict[str, pd.DataFrame]:
    """Return arms with matched extra-SCC counts for a fair diversity control."""
    baseline = train.copy().assign(source="real")
    real_target = baseline[baseline["label"] == target_label]
    if real_target.empty:
        raise ValueError("Target class is absent from the training split.")
    repeated = real_target.sample(n=n_synthetic, replace=n_synthetic > len(real_target), random_state=seed).copy()
    repeated["source"] = "repeated_real_scc"

    accepted = synthetic_manifest.query("accepted").head(n_synthetic).copy()
    if len(accepted) < n_synthetic:
        raise ValueError(f"Only {len(accepted)} quality-screened fakes available; need {n_synthetic}.")
    accepted["image"] = [Path(path).stem for path in accepted["image_path"]]
    accepted["diagnosis"] = baseline.loc[baseline["label"] == target_label, "diagnosis"].iloc[0]
    accepted["label"] = target_label
    accepted["group_id"] = [f"synthetic_{i:05d}" for i in range(len(accepted))]
    accepted["source"] = "cgan_scc"
    return {
        "real_only": baseline,
        "repeated_real_scc": pd.concat([baseline, repeated], ignore_index=True),
        # Synthetic records have no patient metadata; retain only fields available and
        # let pandas mark non-essential real-only metadata as missing.
        "cgan_scc": pd.concat([baseline, accepted.reindex(columns=baseline.columns)], ignore_index=True),
    }


def make_classifier(n_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    return model


def evaluate_classifier(
    model: nn.Module, loader: DataLoader, device: torch.device, target_label: int
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    y_true, y_pred, probabilities = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probability = torch.softmax(logits, dim=1).cpu().numpy()
            probabilities.append(probability)
            y_true.extend(labels.numpy())
            y_pred.extend(probability.argmax(axis=1))
    y_true_arr, y_pred_arr = np.asarray(y_true), np.asarray(y_pred)
    probs_arr = np.concatenate(probabilities)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_arr, y_pred_arr, labels=[target_label], zero_division=0)
    target_binary = (y_true_arr == target_label).astype(int)
    metrics = {
        "accuracy": float((y_true_arr == y_pred_arr).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "target_precision": float(precision[0]),
        "target_recall": float(recall[0]),
        "target_f1": float(f1[0]),
        "target_pr_auc": float(average_precision_score(target_binary, probs_arr[:, target_label])),
    }
    return metrics, {"y_true": y_true_arr, "y_pred": y_pred_arr, "probabilities": probs_arr}


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    target_label: int,
    epochs: int,
    learning_rate: float = 2e-4,
) -> tuple[nn.Module, pd.DataFrame]:
    """Fine-tune EfficientNet; choose the best epoch strictly by validation macro-F1."""
    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_state, best_score = None, -np.inf
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, labels in tqdm(train_loader, desc=f"Classifier {epoch}/{epochs}", leave=False):
            optimiser.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach().cpu()))
        validation, _ = evaluate_classifier(model, val_loader, device, target_label)
        history.append({"epoch": epoch, "train_loss": np.mean(losses), **validation})
        if validation["macro_f1"] > best_score:
            best_score, best_state = validation["macro_f1"], copy.deepcopy(model.state_dict())
        if device.type == "mps":
            torch.mps.empty_cache()
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def group_bootstrap_target_f1(
    outputs: dict[str, np.ndarray], groups: np.ndarray, target_label: int, rounds: int = 1000, seed: int = 146
) -> tuple[float, float]:
    """Group-resampling CI for target F1, preserving multi-image lesions together."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups).astype(str)
    unique_groups = np.unique(groups)
    scores = []
    for _ in range(rounds):
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
        y_true, y_pred = outputs["y_true"][indices], outputs["y_pred"][indices]
        if np.any(y_true == target_label):
            scores.append(f1_score(y_true == target_label, y_pred == target_label, zero_division=0))
    return tuple(np.quantile(scores, [0.025, 0.975]).tolist())


def export_blinded_realism_panel(
    real_paths: list[str], fake_paths: list[str], output_dir: Path, per_source: int = 12, seed: int = 146
) -> Path:
    """Export shuffled images and a separate answer key for a non-clinical class survey."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    real = rng.sample(real_paths, k=min(per_source, len(real_paths)))
    fake = rng.sample(fake_paths, k=min(per_source, len(fake_paths)))
    entries = [(path, "real") for path in real] + [(path, "synthetic") for path in fake]
    rng.shuffle(entries)
    key_path = output_dir / "PRIVATE_answer_key.csv"
    with key_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel_file", "source"])
        for index, (path, source) in enumerate(entries, start=1):
            destination = output_dir / f"panel_{index:02d}.png"
            shutil.copy2(path, destination)
            writer.writerow([destination.name, source])
    return key_path
