"""Patch aggregation and multi-plane case figures for held-out HipMRI evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import HipMRI3DDataset
from modules import build_model
from train import load_audited_pairs


def patient_prostate_dice(prediction: torch.Tensor, target: torch.Tensor, prostate_label: int) -> float:
    """Compute Dice for one reconstructed patient volume, never pooled across patients."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    predicted = prediction == prostate_label
    actual = target == prostate_label
    actual_count = int(actual.sum())
    if not actual_count:
        raise ValueError("patient has no reference prostate voxels")
    denominator = int(predicted.sum()) + actual_count
    return 2 * int((predicted & actual).sum()) / denominator if denominator else 0.0


def mean_patient_prostate_dice(per_patient_scores: dict[str, float]) -> float:
    """Return the arithmetic mean of independently reconstructed patient Dice values."""

    if not per_patient_scores:
        raise ValueError("at least one patient prostate Dice is required")
    return sum(per_patient_scores.values()) / len(per_patient_scores)


def build_test_metrics(per_patient_scores: dict[str, float], per_class: dict[str, dict[str, float | None]], prostate_label: int) -> dict[str, object]:
    """Create the held-out metric contract without leaking background Dice into reports."""

    ranking = [{"patient_id": patient_id, "prostate_dice": score} for patient_id, score in sorted(per_patient_scores.items(), key=lambda item: item[1])]
    return {
        "prostate_label": prostate_label,
        "test_patient_count": len(per_patient_scores),
        "mean_patient_prostate_dice": mean_patient_prostate_dice(per_patient_scores),
        "per_patient_prostate_dice": per_patient_scores,
        "per_class": {label: value for label, value in per_class.items() if label != "0"},
        "failure_ranking": ranking,
    }


def load_checkpoint_model(checkpoint_path: Path | str, device: torch.device) -> tuple[torch.nn.Module, dict[str, object]]:
    """Restore exactly the architecture declared in a validation-selected checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise TypeError("checkpoint is missing experiment config")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, config


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a validation-selected HipMRI checkpoint on manifest test patients only.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--patch-size", required=True, type=int, nargs=3)
    parser.add_argument("--stride", required=True, type=int, nargs=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _class_metrics(prediction: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for label in range(num_classes):
        predicted, actual = prediction == label, target == label
        intersection, union = int((predicted & actual).sum()), int((predicted | actual).sum())
        denominator = int(predicted.sum() + actual.sum())
        result[str(label)] = {"dice": 2 * intersection / denominator if denominator else None, "iou": intersection / union if union else None}
    return result


def prostate_error_masks(
    label: torch.Tensor, prediction: torch.Tensor, prostate_label: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return prostate-specific false-positive and false-negative masks.

    A prostate voxel confused with another foreground organ remains a false
    negative, and another organ predicted as prostate remains a false positive.
    """

    if label.shape != prediction.shape:
        raise ValueError("label and prediction must have identical shapes")
    false_positive = (prediction == prostate_label) & (label != prostate_label)
    false_negative = (label == prostate_label) & (prediction != prostate_label)
    return false_positive, false_negative


def prostate_error_slice_indices(
    label: torch.Tensor, prediction: torch.Tensor, prostate_label: int
) -> tuple[int, int, int]:
    """Choose the prostate FP+FN maximum independently for axial/coronal/sagittal views.

    If a volume has no prostate error, use the centre of the reference prostate
    extent; if reference prostate is absent, use the geometric centre.
    """

    if label.ndim != 3 or prediction.ndim != 3:
        raise ValueError("prostate error slice selection requires 3D volumes")
    false_positive, false_negative = prostate_error_masks(label, prediction, prostate_label)
    error = false_positive | false_negative
    reference = label == prostate_label
    indices: list[int] = []
    for axis in range(3):
        reduce_axes = tuple(dimension for dimension in range(3) if dimension != axis)
        error_scores = error.sum(dim=reduce_axes)
        reference_scores = reference.sum(dim=reduce_axes)
        if int(error_scores.max()) > 0:
            indices.append(int(error_scores.argmax()))
        elif int(reference_scores.max()) > 0:
            occupied = torch.nonzero(reference_scores > 0, as_tuple=False).flatten()
            indices.append(int((occupied[0] + occupied[-1]).item() // 2))
        else:
            indices.append(label.shape[axis] // 2)
    return tuple(indices)


def _cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _overlap(patch_size: tuple[int, int, int], stride: tuple[int, int, int]) -> list[float]:
    if any(step <= 0 or step > patch for patch, step in zip(patch_size, stride, strict=True)):
        raise ValueError("stride dimensions must be positive and no greater than patch dimensions")
    return [1 - step / patch for patch, step in zip(patch_size, stride, strict=True)]


def run_prediction(args: argparse.Namespace) -> dict[str, object]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    device = torch.device(args.device)
    model, config = load_checkpoint_model(args.checkpoint, device)
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if config.get("manifest_sha256") != manifest_hash:
        raise ValueError("checkpoint manifest hash does not match supplied manifest")
    prostate_label = config.get("prostate_label")
    if not isinstance(prostate_label, int) or not str(config.get("prostate_label_source", "")).strip():
        raise ValueError("checkpoint lacks a cited prostate label mapping")
    pairs, split_ids, num_classes = load_audited_pairs(args.manifest)
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir = output / "predictions"
    prediction_dir.mkdir()
    patch_size, stride = tuple(args.patch_size), tuple(args.stride)
    overlap = _overlap(patch_size, stride)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    patient_scores: dict[str, list[float]] = {}
    patient_seconds: dict[str, list[float]] = {}
    representative_cases: dict[str, tuple[float, str, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    class_parts: dict[str, list[dict[str, float | None]]] = {str(label): [] for label in range(num_classes)}
    for pair in pairs:
        if pair.patient_id not in split_ids["test"]:
            continue
        image, target = HipMRI3DDataset([pair], args.data_root, {pair.patient_id}, num_classes, tuple(config["target_shape"]))[0]
        _cuda_synchronize(device)
        case_started = time.perf_counter()
        logits = sliding_window_logits(model, image.to(device), patch_size, stride)
        _cuda_synchronize(device)
        case_seconds = time.perf_counter() - case_started
        prediction = logits.argmax(0).cpu()
        torch.save({"case_id": pair.case_id, "patient_id": pair.patient_id, "image": image.cpu(), "target": target.cpu().long(), "prediction": prediction.long(), "manifest_sha256": manifest_hash, "prostate_label": prostate_label, "target_shape": config["target_shape"]}, prediction_dir / f"{pair.case_id}.pt")
        case_score = patient_prostate_dice(prediction, target, prostate_label)
        patient_scores.setdefault(pair.patient_id, []).append(case_score)
        patient_seconds.setdefault(pair.patient_id, []).append(case_seconds)
        representative = representative_cases.get(pair.patient_id)
        if representative is None or case_score < representative[0]:
            representative_cases[pair.patient_id] = (case_score, pair.case_id, image[0].cpu(), target.cpu(), prediction)
        for label, values in _class_metrics(prediction, target, num_classes).items():
            class_parts[label].append(values)
    per_patient = {patient: sum(scores) / len(scores) for patient, scores in patient_scores.items()}
    per_patient_seconds = {patient: sum(seconds) for patient, seconds in patient_seconds.items()}
    per_class = {label: {metric: (sum(value[metric] for value in values if value[metric] is not None) / sum(value[metric] is not None for value in values)) if any(value[metric] is not None for value in values) else None for metric in ("dice", "iou")} for label, values in class_parts.items()}
    metrics = build_test_metrics(per_patient, per_class, prostate_label)
    panel_dir = output / "failure_panels"
    panel_files = []
    failure_panel_selection: dict[str, dict[str, object]] = {}
    for failure in metrics["failure_ranking"]:
        patient_id = failure["patient_id"]
        _, case_id, image, target, prediction = representative_cases[patient_id]
        slice_indices = prostate_error_slice_indices(target, prediction, prostate_label)
        panel_path = panel_dir / f"{patient_id}.png"
        render_case_views(
            image,
            target,
            prediction,
            panel_path,
            title=f"{patient_id} {case_id}: patient prostate Dice={failure['prostate_dice']:.4f}",
            prostate_label=prostate_label,
            slice_indices=slice_indices,
        )
        panel_files.append(str(panel_path.relative_to(output)))
        failure_panel_selection[patient_id] = {
            "case_id": case_id,
            "slice_indices_dhw": list(slice_indices),
            "rule": "max_prostate_fp_plus_fn_per_plane; prostate_extent_centre_fallback",
        }
    metrics.update({"checkpoint": str(args.checkpoint), "manifest_sha256": manifest_hash, "prostate_label_source": config["prostate_label_source"], "target_shape": config["target_shape"], "patch_size": list(patch_size), "stride": list(stride), "overlap": overlap, "logit_merge_method": "overlap_averaged_logits", "per_patient_inference_seconds": per_patient_seconds, "total_inference_seconds": sum(per_patient_seconds.values()), "mean_patient_inference_seconds": sum(per_patient_seconds.values()) / len(per_patient_seconds), "peak_inference_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0, "failure_panels": panel_files, "failure_panel_selection": failure_panel_selection})
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _window_starts(length: int, patch: int, stride: int) -> list[int]:
    if patch <= 0 or stride <= 0:
        raise ValueError("patch dimensions and strides must be positive")
    if patch > length:
        raise ValueError(f"patch dimension {patch} exceeds volume dimension {length}")
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def sliding_window_logits(
    model: torch.nn.Module,
    volume: torch.Tensor,
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int],
) -> torch.Tensor:
    """Average overlapping patch logits and assert every output voxel was predicted."""

    if volume.ndim != 4:
        raise ValueError(f"volume must have shape (C, D, H, W), found {tuple(volume.shape)}")
    starts = [_window_starts(length, patch, step) for length, patch, step in zip(volume.shape[-3:], patch_size, stride, strict=True)]
    previous_training = model.training
    model.eval()
    logits_sum: torch.Tensor | None = None
    prediction_count = torch.zeros(volume.shape[-3:], dtype=torch.int32, device=volume.device)
    with torch.no_grad():
        for depth in starts[0]:
            for height in starts[1]:
                for width in starts[2]:
                    patch = volume[:, depth:depth + patch_size[0], height:height + patch_size[1], width:width + patch_size[2]]
                    output = model(patch[None])
                    logits = output[0] if isinstance(output, tuple) else output
                    if logits_sum is None:
                        logits_sum = torch.zeros((logits.shape[1], *volume.shape[-3:]), dtype=logits.dtype, device=logits.device)
                    logits_sum[:, depth:depth + patch_size[0], height:height + patch_size[1], width:width + patch_size[2]] += logits[0]
                    prediction_count[depth:depth + patch_size[0], height:height + patch_size[1], width:width + patch_size[2]] += 1
    model.train(previous_training)
    if logits_sum is None or torch.any(prediction_count == 0):
        raise RuntimeError("sliding-window inference did not produce a prediction for every voxel")
    return logits_sum / prediction_count.to(dtype=logits_sum.dtype).unsqueeze(0)


def _as_volume(value: torch.Tensor | np.ndarray) -> np.ndarray:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"case views require a 3D array, found {array.shape}")
    return array


def render_case_views(
    image: torch.Tensor | np.ndarray,
    label: torch.Tensor | np.ndarray,
    prediction: torch.Tensor | np.ndarray,
    output_path: Path | str,
    title: str,
    prostate_label: int,
    slice_indices: tuple[int, int, int] | None = None,
) -> None:
    """Render prostate-specific failure views at the selected error slices."""

    image_array, label_array, prediction_array = (_as_volume(value) for value in (image, label, prediction))
    if image_array.shape != label_array.shape or image_array.shape != prediction_array.shape:
        raise ValueError("image, label and prediction must have identical 3D shapes")
    selected = slice_indices or prostate_error_slice_indices(
        torch.as_tensor(label_array), torch.as_tensor(prediction_array), prostate_label
    )
    if len(selected) != 3 or any(index < 0 or index >= size for index, size in zip(selected, image_array.shape, strict=True)):
        raise ValueError("slice_indices must contain valid axial, coronal and sagittal indices")
    planes = (
        (f"axial z={selected[0]}", image_array[selected[0]], label_array[selected[0]], prediction_array[selected[0]]),
        (f"coronal y={selected[1]}", image_array[:, selected[1], :], label_array[:, selected[1], :], prediction_array[:, selected[1], :]),
        (f"sagittal x={selected[2]}", image_array[:, :, selected[2]], label_array[:, :, selected[2]], prediction_array[:, :, selected[2]]),
    )
    figure, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
    for row, (plane_name, image_plane, label_plane, prediction_plane) in enumerate(planes):
        false_positive, false_negative = prostate_error_masks(
            torch.as_tensor(label_plane), torch.as_tensor(prediction_plane), prostate_label
        )
        false_positive, false_negative = false_positive.numpy(), false_negative.numpy()
        panels = (("MRI", image_plane, "gray"), ("ground truth", label_plane, "tab20"), ("prediction", prediction_plane, "tab20"), ("FP / FN", false_positive.astype(int) + 2 * false_negative.astype(int), "coolwarm"))
        for column, (name, panel, colour_map) in enumerate(panels):
            axes[row, column].imshow(panel.T, cmap=colour_map, origin="lower")
            axes[row, column].set_title(f"{plane_name}: {name}")
            axes[row, column].axis("off")
    figure.suptitle(title)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    print(json.dumps(run_prediction(create_argument_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
