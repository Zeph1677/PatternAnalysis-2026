"""HipMRI NIfTI discovery, validation and patient-disjoint 3D data access."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class AugmentationConfig:
    """Probabilities and intensity ranges for training only, after normalization."""
    lr_flip: float = 0.5
    ap_flip: float = 0.5
    noise: float = 0.5
    blur: float = 0.25
    gamma: float = 0.25
    noise_std_max: float = 0.1
    blur_sigma_min: float = 0.5
    blur_sigma_max: float = 1.0
    gamma_min: float = 0.7
    gamma_max: float = 1.5

    def __post_init__(self):
        for name in ("lr_flip", "ap_flip", "noise", "blur", "gamma"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} probability must be in [0, 1]")
        if not all(np.isfinite(getattr(self, name)) for name in self.__dataclass_fields__):
            raise ValueError("augmentation values must be finite")
        if self.noise_std_max < 0 or not 0 < self.blur_sigma_min <= self.blur_sigma_max or not 0 < self.gamma_min <= self.gamma_max:
            raise ValueError("invalid augmentation intensity ranges")


def augment_training_pair(image, mask, config, orientation):
    """Map anatomical flips through NIfTI axis codes; intensity acts on MRI only."""
    for codes, probability in (({'L', 'R'}, config.lr_flip), ({'A', 'P'}, config.ap_flip)):
        axes = [2 - axis for axis, code in enumerate(orientation) if code in codes]
        if len(axes) != 1:
            raise ValueError("orientation must identify LR and AP axes uniquely")
        if torch.rand(()) < probability:
            image, mask = image.flip((axes[0]+1,)), mask.flip((axes[0],))
    if torch.rand(()) < config.noise:
        image = image + torch.randn_like(image) * (torch.rand(()).item() * config.noise_std_max)
    if torch.rand(()) < config.blur:
        sigma = config.blur_sigma_min + torch.rand(()).item() * (config.blur_sigma_max-config.blur_sigma_min)
        radius = int(np.ceil(3*sigma))
        coordinates = torch.arange(-radius,radius+1, dtype=image.dtype, device=image.device)
        kernel = torch.exp(-coordinates.square()/(2*sigma*sigma)); kernel /= kernel.sum()
        value = image[None]
        for axis in range(3):
            shape = [1,1,1,1,1]; shape[axis+2] = len(kernel)
            padding = [0,0,0]; padding[axis] = radius
            value = F.conv3d(F.pad(value, (padding[2],padding[2],padding[1],padding[1],padding[0],padding[0]), mode="replicate"),kernel.reshape(shape))
        image = value[0]
    if torch.rand(()) < config.gamma:
        exponent = config.gamma_min + torch.rand(()).item() * (config.gamma_max-config.gamma_min)
        low, high = image.min(), image.max()
        image = ((image-low)/(high-low).clamp_min(1e-8)).pow(exponent)*(high-low)+low
    return image.contiguous(), mask.contiguous()


@dataclass(frozen=True)
class VolumePair:
    """A validated image-mask pair stored as paths relative to the HipMRI root."""

    case_id: str
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


def _without_nifti_suffix(path: Path) -> Path:
    name = path.name
    if name.endswith(".nii.gz"):
        return path.with_name(name[:-7])
    if name.endswith(".nii"):
        return path.with_name(name[:-4])
    raise ValueError(f"not a NIfTI path: {path}")


def _case_key(relative_path: Path, expected_suffix: str) -> str:
    """Return a role-independent case key for conventional or HipMRI file names."""

    stem_path = _without_nifti_suffix(relative_path)
    match = re.fullmatch(r"(?P<case>.+)_(?P<role>LFOV|SEMANTIC)", stem_path.name)
    if match:
        if match.group("role") != expected_suffix:
            raise ValueError(f"expected _{expected_suffix} file name, found {relative_path}")
        stem_path = stem_path.with_name(match.group("case"))
    return stem_path.as_posix()


def _path_map_by_case_key(paths: dict[Path, Path], expected_suffix: str) -> dict[str, Path]:
    mapped: dict[str, Path] = {}
    for relative_path, absolute_path in paths.items():
        key = _case_key(relative_path, expected_suffix)
        if key in mapped:
            raise ValueError(f"duplicate case key {key!r} below {absolute_path.parent}")
        mapped[key] = absolute_path
    return mapped


def _three_dimensional(data: np.ndarray, path: Path) -> np.ndarray:
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path} must contain a 3D volume or a singleton-channel 4D volume; found {data.shape}")
    return data


def _patient_id(case_id: str, source_path: Path) -> str:
    """Use the HipMRI subject prefix when present, otherwise preserve nested layout IDs."""

    case_name = Path(case_id).name
    match = re.fullmatch(r"(?P<patient>[A-Za-z]+\d+)_Week\d+", case_name)
    if match:
        return match.group("patient")
    return source_path.parts[0] if len(source_path.parts) > 1 else case_name


def discover_volume_pairs(data_root: Path | str) -> list[VolumePair]:
    """Discover and validate relative-path matched HipMRI semantic volumes."""

    root = Path(data_root).resolve()
    image_root = root / "semantic_MRs"
    mask_root = root / "semantic_labels_only"
    images = _path_map_by_case_key(_nifti_paths(image_root), "LFOV")
    masks = _path_map_by_case_key(_nifti_paths(mask_root), "SEMANTIC")
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        fragments: list[str] = []
        if missing_masks:
                fragments.append("missing mask for " + ", ".join(missing_masks[:5]))
        if missing_images:
                fragments.append("missing image for " + ", ".join(missing_images[:5]))
        raise ValueError("; ".join(fragments))
    if not images:
        raise ValueError(f"no NIfTI pairs found below {root}")

    pairs: list[VolumePair] = []
    for case_id in sorted(images):
        image_path, mask_path = images[case_id], masks[case_id]
        image_nii, mask_nii = nib.load(image_path), nib.load(mask_path)
        image = _three_dimensional(image_nii.get_fdata(caching="unchanged"), image_path)
        mask = _three_dimensional(mask_nii.get_fdata(caching="unchanged"), mask_path)
        if image.shape != mask.shape:
            raise ValueError(f"shape mismatch for {case_id}: image {image.shape}, mask {mask.shape}")
        if not np.isfinite(image).all():
            raise ValueError(f"non-finite intensity values in {image_path}")
        if not np.allclose(mask, np.rint(mask)):
            raise ValueError(f"non-integral labels in {mask_path}")
        if not np.allclose(image_nii.affine, mask_nii.affine, atol=1e-4):
            raise ValueError(f"affine mismatch for {case_id}")
        pairs.append(
            VolumePair(
                case_id=case_id,
                patient_id=_patient_id(case_id, image_path.relative_to(image_root)),
                image_path=image_path.relative_to(root),
                mask_path=mask_path.relative_to(root),
                shape=tuple(int(value) for value in image.shape),
                spacing=tuple(float(value) for value in image_nii.header.get_zooms()[:3]),
                orientation=tuple(str(value) for value in nib.aff2axcodes(image_nii.affine)),
                labels=tuple(int(value) for value in np.unique(mask)),
            )
        )
    return pairs


def build_patient_split_manifest(pairs: list[VolumePair], seed: int = 42) -> dict[str, list[str] | int]:
    """Create a deterministic 70/15/15 patient split with all groups non-empty."""

    patients = sorted({pair.patient_id for pair in pairs})
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

    def __init__(
        self,
        pairs: list[VolumePair],
        data_root: Path | str,
        patient_ids: set[str],
        num_classes: int,
        target_shape: tuple[int, int, int] | None = None,
        training_patch_shape: tuple[int, int, int] | None = None,
        foreground_label: int | None = None,
        foreground_probability: float = 0.0,
        augmentation_config: AugmentationConfig | None = None,
        split: str = "validation",
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("invalid dataset split")
        if augmentation_config is not None and split != "train":
            raise ValueError("augmentation is permitted only for train")
        self.augmentation_config = augmentation_config
        self.records = [pair for pair in pairs if pair.patient_id in patient_ids]
        self.root = Path(data_root).resolve()
        self.num_classes = num_classes
        self.target_shape = target_shape
        self.training_patch_shape = training_patch_shape
        self.foreground_label = foreground_label
        self.foreground_probability = foreground_probability
        if not self.records:
            raise ValueError("dataset partition contains no volumes")
        if target_shape is not None and (len(target_shape) != 3 or any(size <= 0 or size % 8 for size in target_shape)):
            raise ValueError("target_shape must contain three positive dimensions divisible by 8")
        if training_patch_shape is not None:
            if target_shape is None:
                raise ValueError("training_patch_shape requires target_shape")
            if len(training_patch_shape) != 3 or any(size <= 0 or size % 8 for size in training_patch_shape):
                raise ValueError("training_patch_shape must contain three positive dimensions divisible by 8")
            if any(patch > target for patch, target in zip(training_patch_shape, target_shape)):
                raise ValueError("training_patch_shape cannot exceed target_shape")
        if foreground_label is not None and not 0 <= foreground_label < num_classes:
            raise ValueError("foreground_label is outside the declared label range")
        if not 0.0 <= foreground_probability <= 1.0:
            raise ValueError("foreground_probability must be in [0, 1]")
        if foreground_probability and (training_patch_shape is None or foreground_label is None):
            raise ValueError("foreground sampling requires a patch shape and declared foreground label")

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
        image_tensor, mask_tensor = torch.from_numpy(image_dhw[None]), torch.from_numpy(mask_dhw)
        if self.target_shape is not None:
            image_tensor = F.interpolate(image_tensor[None], size=self.target_shape, mode="trilinear", align_corners=False)[0]
            mask_tensor = F.interpolate(mask_tensor[None, None].float(), size=self.target_shape, mode="nearest")[0, 0].long()
        if self.training_patch_shape is not None:
            starts = self._sample_patch_starts(mask_tensor)
            slices = tuple(slice(start, start + size) for start, size in zip(starts, self.training_patch_shape))
            image_tensor, mask_tensor = image_tensor[(slice(None), *slices)], mask_tensor[slices]
        if self.augmentation_config is not None:
            image_tensor, mask_tensor = augment_training_pair(image_tensor, mask_tensor, self.augmentation_config, pair.orientation)
        return image_tensor, mask_tensor

    def _sample_patch_starts(self, mask: torch.Tensor) -> tuple[int, int, int]:
        """Choose a valid crop that contains foreground when requested."""

        assert self.training_patch_shape is not None
        coordinates = torch.empty((0, 3), dtype=torch.long)
        if self.foreground_label is not None and torch.rand(()) < self.foreground_probability:
            coordinates = torch.nonzero(mask == self.foreground_label, as_tuple=False)
        if len(coordinates):
            point = coordinates[torch.randint(len(coordinates), ())]
            starts = []
            for coordinate, size, patch in zip(point.tolist(), mask.shape, self.training_patch_shape):
                low, high = max(0, coordinate - patch + 1), min(coordinate, size - patch)
                starts.append(int(torch.randint(low, high + 1, ()).item()))
            return tuple(starts)
        return tuple(int(torch.randint(0, size - patch + 1, ()).item()) for size, patch in zip(mask.shape, self.training_patch_shape))
