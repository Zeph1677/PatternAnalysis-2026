"""HipMRI NIfTI discovery, validation and patient-disjoint 3D data access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class VolumePair:
    """A validated image-mask pair stored as paths relative to the HipMRI root."""

    patient_id: str
    image_path: Path
    mask_path: Path
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    orientation: tuple[str, str, str]
    labels: tuple[int, ...]


def _nifti_paths(directory: Path) -> dict[Path, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"required directory does not exist: {directory}")
    paths = sorted(path for path in directory.rglob("*") if path.name.endswith((".nii", ".nii.gz")))
    return {path.relative_to(directory): path for path in paths}


def _three_dimensional(data: np.ndarray, path: Path) -> np.ndarray:
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path} must contain a 3D volume or a singleton-channel 4D volume; found {data.shape}")
    return data


def _patient_id(relative_path: Path) -> str:
    return relative_path.parts[0] if len(relative_path.parts) > 1 else relative_path.name.split(".")[0]


def discover_volume_pairs(data_root: Path | str) -> list[VolumePair]:
    """Discover and validate relative-path matched HipMRI semantic volumes."""

    root = Path(data_root).resolve()
    image_root = root / "semantic_MRs"
    mask_root = root / "semantic_labels_only"
    images, masks = _nifti_paths(image_root), _nifti_paths(mask_root)
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        fragments: list[str] = []
        if missing_masks:
            fragments.append("missing mask for " + ", ".join(str(path) for path in missing_masks[:5]))
        if missing_images:
            fragments.append("missing image for " + ", ".join(str(path) for path in missing_images[:5]))
        raise ValueError("; ".join(fragments))
    if not images:
        raise ValueError(f"no NIfTI pairs found below {root}")

    pairs: list[VolumePair] = []
    for relative_path in sorted(images):
        image_nii, mask_nii = nib.load(images[relative_path]), nib.load(masks[relative_path])
        image = _three_dimensional(image_nii.get_fdata(caching="unchanged"), images[relative_path])
        mask = _three_dimensional(mask_nii.get_fdata(caching="unchanged"), masks[relative_path])
        if image.shape != mask.shape:
            raise ValueError(f"shape mismatch for {relative_path}: image {image.shape}, mask {mask.shape}")
        if not np.isfinite(image).all():
            raise ValueError(f"non-finite intensity values in {images[relative_path]}")
        if not np.allclose(mask, np.rint(mask)):
            raise ValueError(f"non-integral labels in {masks[relative_path]}")
        if not np.allclose(image_nii.affine, mask_nii.affine, atol=1e-4):
            raise ValueError(f"affine mismatch for {relative_path}")
        pairs.append(
            VolumePair(
                patient_id=_patient_id(relative_path),
                image_path=images[relative_path].relative_to(root),
                mask_path=masks[relative_path].relative_to(root),
                shape=tuple(int(value) for value in image.shape),
                spacing=tuple(float(value) for value in image_nii.header.get_zooms()[:3]),
                orientation=tuple(str(value) for value in nib.aff2axcodes(image_nii.affine)),
                labels=tuple(int(value) for value in np.unique(mask)),
            )
        )
    if len({pair.patient_id for pair in pairs}) != len(pairs):
        raise ValueError("each HipMRI patient must map to exactly one matched 3D image-mask pair")
    return pairs


def build_patient_split_manifest(pairs: list[VolumePair], seed: int = 42) -> dict[str, list[str] | int]:
    """Create a deterministic 70/15/15 patient split with all groups non-empty."""

    patients = sorted(pair.patient_id for pair in pairs)
    if len(set(patients)) != len(patients):
        raise ValueError("cannot split duplicate patient IDs")
    if len(patients) < 3:
        raise ValueError("at least three patients are required for train, validation and test partitions")
    random.Random(seed).shuffle(patients)
    train_count = max(1, round(len(patients) * 0.70))
    validation_count = max(1, round(len(patients) * 0.15))
    while train_count + validation_count >= len(patients):
        if train_count > validation_count and train_count > 1:
            train_count -= 1
        else:
            validation_count -= 1
    manifest: dict[str, list[str] | int] = {
        "seed": seed,
        "train": sorted(patients[:train_count]),
        "validation": sorted(patients[train_count:train_count + validation_count]),
        "test": sorted(patients[train_count + validation_count:]),
    }
    validate_patient_split_manifest(manifest)
    return manifest


def validate_patient_split_manifest(manifest: dict[str, object]) -> None:
    """Reject empty or overlapping patient partitions."""

    groups = {name: set(manifest.get(name, [])) for name in ("train", "validation", "test")}
    if any(not values for values in groups.values()):
        raise ValueError("train, validation and test partitions must all be non-empty")
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
        raise ValueError("patient overlap between partitions")


class HipMRI3DDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load normalized image and integer mask tensors in `(C, D, H, W)` orientation."""

    def __init__(self, pairs: list[VolumePair], data_root: Path | str, patient_ids: set[str], num_classes: int) -> None:
        self.records = [pair for pair in pairs if pair.patient_id in patient_ids]
        self.root = Path(data_root).resolve()
        self.num_classes = num_classes
        if not self.records:
            raise ValueError("dataset partition contains no volumes")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.records[index]
        image = _three_dimensional(nib.load(self.root / pair.image_path).get_fdata(caching="unchanged"), self.root / pair.image_path)
        mask = _three_dimensional(nib.load(self.root / pair.mask_path).get_fdata(caching="unchanged"), self.root / pair.mask_path)
        if not np.allclose(mask, np.rint(mask)) or np.any(mask < 0) or np.any(mask >= self.num_classes):
            raise ValueError(f"mask labels for {pair.patient_id} are outside [0, {self.num_classes - 1}]")
        image = image.astype(np.float32)
        image = (image - image.mean()) / (image.std() + 1e-8)
        image_dhw, mask_dhw = np.transpose(image, (2, 1, 0)), np.transpose(mask.astype(np.int64), (2, 1, 0))
        return torch.from_numpy(image_dhw[None]), torch.from_numpy(mask_dhw)
