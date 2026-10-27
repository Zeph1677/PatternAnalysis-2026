"""Train and validate HipMRI models; predict.py performs held-out testing explicitly."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import AugmentationConfig, HipMRI3DDataset, VolumePair
from modules import build_model
from plot_training import render_learning_rate_curve, render_training_curves


def model_configuration(args: argparse.Namespace, num_classes: int) -> dict[str, object]:
    """Freeze an unambiguous architecture definition for every new run."""

    model_name = getattr(args, "model", None)
    version = getattr(args, "architecture_version", None)
    if model_name == "baseline":
        if version not in (None, "baseline_v1"):
            raise ValueError("baseline accepts only architecture-version baseline_v1")
        version = "baseline_v1"
    elif model_name == "improved":
        if version not in ("improved_v1", "improved_context_v2"):
            raise ValueError(
                "improved training requires --architecture-version "
                "improved_v1 or improved_context_v2"
            )
    else:
        raise ValueError("model must be 'baseline' or 'improved'")
    return {
        "model": model_name,
        "architecture_version": version,
        "num_classes": num_classes,
        "in_channels": 1,
        "base_channels": args.base_channels,
        "dropout_probability": getattr(args, "dropout_probability", 0.3),
    }


def build_experiment_config(
    manifest_path: Path | str,
    model_name: str,
    target_shape: tuple[int, int, int],
    seed: int,
    prostate_label: int,
    prostate_label_source: str,
) -> dict[str, object]:
    """Create the immutable metadata required before a formal training run."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    labels = manifest.get("summary", {}).get("labels")
    if model_name not in {"baseline", "improved"}:
        raise ValueError("model name must be 'baseline' or 'improved'")
    if not isinstance(labels, list) or not labels:
        raise ValueError("manifest must define at least one label")
    if len(target_shape) != 3 or any(size < 16 or size % 8 for size in target_shape):
        raise ValueError("target shape must contain dimensions of at least 16 divisible by 8")
    if isinstance(prostate_label, bool) or prostate_label not in labels:
        raise ValueError("prostate label is outside manifest labels")
    source = prostate_label_source.strip()
    if not source:
        raise ValueError("prostate label source is required")
    config: dict[str, object] = {
        "schema_version": 1,
        "model": model_name,
        "seed": seed,
        "target_shape": list(target_shape),
        "num_classes": len(labels),
        "prostate_label": prostate_label,
        "prostate_label_source": source,
        "manifest_path": str(path),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return config


def load_audited_pairs(manifest_path: Path | str) -> tuple[list[VolumePair], dict[str, set[str]], int]:
    """Restore only audited relative metadata and the fixed patient partitions."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    labels = manifest.get("summary", {}).get("labels")
    cases = manifest.get("cases")
    if not isinstance(labels, list) or labels != list(range(len(labels))) or not cases:
        raise ValueError("manifest must contain contiguous labels and at least one audited case")
    pairs = [
        VolumePair(
            case_id=case["case_id"], patient_id=case["patient_id"], image_path=Path(case["image_path"]), mask_path=Path(case["mask_path"]),
            shape=tuple(case["shape"]), spacing=tuple(case["spacing"]), orientation=tuple(case["orientation"]), labels=tuple(case["labels"]),
        )
        for case in cases
    ]
    split_ids = {split: {case["patient_id"] for case in cases if case["split"] == split} for split in ("train", "validation", "test")}
    if set().union(*split_ids.values()) != {pair.patient_id for pair in pairs}:
        raise ValueError("manifest has unassigned patients")
    if split_ids["train"] & split_ids["validation"] or split_ids["train"] & split_ids["test"] or split_ids["validation"] & split_ids["test"]:
        raise ValueError("manifest patient partitions overlap")
    return pairs, split_ids, len(labels)


def _soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(target, num_classes).movedim(-1, 1).float()
    overlap = (probabilities * one_hot).sum(dim=(0, 2, 3, 4))
    total = probabilities.sum(dim=(0, 2, 3, 4)) + one_hot.sum(dim=(0, 2, 3, 4))
    dice = (2 * overlap + 1e-6) / (total + 1e-6)
    return 1 - (dice * class_weights).sum() / class_weights.sum() if class_weights is not None else 1 - dice.mean()


def composite_3d_loss(output: torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]], target: torch.Tensor, num_classes: int, prostate_label: int | None = None, prostate_ce_weight: float = 1.0, prostate_dice_weight: float = 1.0) -> torch.Tensor:
    """Cross entropy plus soft Dice, with explicitly downsampled auxiliary targets."""

    if prostate_label is not None and (prostate_label < 0 or prostate_label >= num_classes or prostate_ce_weight <= 0 or prostate_dice_weight <= 0):
        raise ValueError("prostate label and loss weights must be positive and in range")
    final, auxiliaries = output if isinstance(output, tuple) else (output, ())
    ce_weights = torch.ones(num_classes, device=final.device)
    dice_weights = torch.ones(num_classes, device=final.device)
    if prostate_label is not None:
        ce_weights[prostate_label] = prostate_ce_weight
        dice_weights[prostate_label] = prostate_dice_weight
    loss = F.cross_entropy(final, target, weight=ce_weights, reduction="none").mean() + _soft_dice_loss(final, target, num_classes, dice_weights)
    for auxiliary in auxiliaries:
        scaled_target = F.interpolate(target[:, None].float(), size=auxiliary.shape[-3:], mode="nearest")[:, 0].long()
        loss = loss + 0.25 * (F.cross_entropy(auxiliary, scaled_target, weight=ce_weights, reduction="none").mean() + _soft_dice_loss(auxiliary, scaled_target, num_classes, dice_weights))
    return loss


def evaluate_3d_batches(model: torch.nn.Module, batches: Iterable[tuple[torch.Tensor, torch.Tensor]], device: torch.device, num_classes: int, prostate_label: int | None = None, patient_ids: Iterable[str] | None = None) -> dict[str, object]:
    """Compute hard held-out Dice/IoU and resource fields without model selection."""

    model.eval()
    prediction_parts, target_parts = [], []
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for images, masks in batches:
            output = model(images.to(device))
            logits = output[0] if isinstance(output, tuple) else output
            prediction_parts.append(logits.argmax(1).cpu()); target_parts.append(masks.cpu())
    if device.type == "cuda": torch.cuda.synchronize(device)
    prediction, target = torch.cat(prediction_parts), torch.cat(target_parts)
    per_class, dice_values, iou_values = {}, [], []
    for label in range(num_classes):
        predicted, actual = prediction == label, target == label
        intersection = int((predicted & actual).sum())
        union = int((predicted | actual).sum())
        denominator = int(predicted.sum() + actual.sum())
        dice = (2 * intersection / denominator) if denominator else None
        iou = (intersection / union) if union else None
        per_class[str(label)] = {"dice": dice, "iou": iou, "voxels": int(actual.sum())}
        if dice is not None: dice_values.append(dice)
        if iou is not None: iou_values.append(iou)
    result: dict[str, object] = {"per_class": per_class, "mean_dice": sum(dice_values) / len(dice_values), "mean_iou": sum(iou_values) / len(iou_values), "inference_seconds": time.perf_counter() - started, "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0}
    if (prostate_label is None) != (patient_ids is None):
        raise ValueError("prostate_label and patient_ids must be supplied together")
    if prostate_label is not None and patient_ids is not None:
        patient_scores: dict[str, list[float]] = {}
        false_positive = false_negative = 0
        identifiers = iter(patient_ids)
        for predicted, actual in zip(prediction, target):
            try:
                patient_id = next(identifiers)
            except StopIteration as error:
                raise ValueError("patient_ids has fewer entries than validation volumes") from error
            counts = prostate_diagnostics(predicted, actual, prostate_label)
            patient_scores.setdefault(patient_id, []).append(float(counts["prostate_dice"]))
            false_positive += int(counts["false_positive_voxels"])
            false_negative += int(counts["false_negative_voxels"])
        if next(identifiers, None) is not None:
            raise ValueError("patient_ids has more entries than validation volumes")
        per_patient = {patient_id: sum(scores) / len(scores) for patient_id, scores in patient_scores.items()}
        if not per_patient:
            raise ValueError("validation partition contains no patient prostate scores")
        result.update({
            "validation_patient_prostate_dice": sum(per_patient.values()) / len(per_patient),
            "validation_worst_patient_prostate_dice": min(per_patient.values()),
            "validation_prostate_false_positive_voxels": false_positive,
            "validation_prostate_false_negative_voxels": false_negative,
            "validation_patient_prostate_dice_by_patient": per_patient,
        })
    return result


def select_validation_checkpoint(history: list[dict[str, object]], selection_metric: str = "mean_dice") -> dict[str, object]:
    """Choose solely by saved validation Dice; test values cannot influence selection."""

    if not history: raise ValueError("cannot select checkpoint from empty history")
    if selection_metric == "mean_dice":
        return max(history, key=lambda row: float(row["validation_mean_dice"]))
    if selection_metric != "prostate_dice":
        raise ValueError(f"unsupported selection metric: {selection_metric}")
    required = ("validation_patient_prostate_dice", "validation_worst_patient_prostate_dice", "validation_prostate_false_positive_voxels", "validation_prostate_false_negative_voxels", "epoch")
    if any(any(key not in row for key in required) for row in history):
        raise ValueError("prostate_dice selection requires complete validation prostate metrics")
    return max(history, key=lambda row: (
        float(row["validation_patient_prostate_dice"]),
        float(row["validation_worst_patient_prostate_dice"]),
        -int(row["validation_prostate_false_positive_voxels"]) - int(row["validation_prostate_false_negative_voxels"]),
        -float(row["epoch"]),
    ))


def create_argument_parser() -> argparse.ArgumentParser:
    """Build the explicit command line contract for a reproducible training run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/training"))
    parser.add_argument("--model", choices=("baseline", "improved"), default="baseline")
    parser.add_argument("--architecture-version", choices=("improved_v1", "improved_context_v2"), help="Required for --model improved. Main historical experiment uses improved_v1.")
    parser.add_argument("--dropout-probability", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--augmentation", action="store_true", help="Enable train-only augmentation")
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument("--eta-min", type=float, default=1e-5)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--target-shape", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prostate-label", required=True, type=int)
    parser.add_argument("--prostate-label-source", required=True)
    parser.add_argument("--prostate-ce-weight", type=float, default=8.0)
    parser.add_argument("--prostate-dice-weight", type=float, default=8.0)
    parser.add_argument("--train-patch-shape", type=int, nargs=3)
    parser.add_argument("--foreground-patch-probability", type=float, default=0.0)
    parser.add_argument("--selection-metric", choices=("mean_dice", "prostate_dice"), default="mean_dice")
    parser.add_argument("--preflight-patient-id")
    parser.add_argument("--preflight-updates", type=int)
    parser.add_argument("--preflight-eval-every", type=int)
    return parser


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def _cuda_peak_mb(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0


def training_patch_options(args: argparse.Namespace, target_shape: tuple[int, int, int]) -> tuple[tuple[int, int, int] | None, float]:
    """Validate the train-only crop configuration retained in run metadata."""

    raw_shape = getattr(args, "train_patch_shape", None)
    probability = float(getattr(args, "foreground_patch_probability", 0.0))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("foreground patch probability must be in [0, 1]")
    if raw_shape is None:
        if probability:
            raise ValueError("foreground patch probability requires --train-patch-shape")
        return None, probability
    patch_shape = tuple(raw_shape)
    if len(patch_shape) != 3 or any(size < 16 or size % 8 for size in patch_shape):
        raise ValueError("train patch shape must contain dimensions of at least 16 divisible by 8")
    if any(patch > target for patch, target in zip(patch_shape, target_shape)):
        raise ValueError("train patch shape cannot exceed target shape")
    return patch_shape, probability


def prostate_diagnostics(prediction: torch.Tensor, target: torch.Tensor, prostate_label: int) -> dict[str, int | float]:
    """Return auditable prostate overlap counts for train-only preflight."""

    predicted, reference = prediction == prostate_label, target == prostate_label
    true_positive = int((predicted & reference).sum())
    predicted_count, reference_count = int(predicted.sum()), int(reference.sum())
    denominator = predicted_count + reference_count
    return {
        "reference_prostate_voxels": reference_count,
        "predicted_prostate_voxels": predicted_count,
        "true_positive_voxels": true_positive,
        "false_positive_voxels": predicted_count - true_positive,
        "false_negative_voxels": reference_count - true_positive,
        "prostate_dice": 2 * true_positive / denominator if denominator else 0.0,
    }


def _train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, num_classes: int, prostate_label: int, prostate_ce_weight: float, prostate_dice_weight: float) -> float:
    model.train()
    losses: list[float] = []
    for images, masks in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = composite_3d_loss(model(images.to(device)), masks.to(device), num_classes, prostate_label, prostate_ce_weight, prostate_dice_weight)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise ValueError("training partition yielded no batches")
    return sum(losses) / len(losses)


def run_overfit_preflight(args: argparse.Namespace) -> dict[str, object]:
    """Fit one declared train case only; this diagnostic is never reportable."""

    patient_id = str(getattr(args, "preflight_patient_id", "") or "").strip()
    if not patient_id:
        raise ValueError("preflight patient ID is required")
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0 or args.base_channels <= 0:
        raise ValueError("epochs, batch size, learning rate and base channels must be positive")
    updates = int(getattr(args, "preflight_updates", None) or args.epochs)
    evaluation_interval = int(getattr(args, "preflight_eval_every", None) or updates)
    if updates <= 0 or evaluation_interval <= 0:
        raise ValueError("preflight updates and evaluation interval must be positive")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    pairs, split_ids, num_classes = load_audited_pairs(args.manifest)
    if patient_id not in split_ids["train"]:
        raise ValueError("preflight patient must belong to the manifest train split")
    pair = next(pair for pair in pairs if pair.patient_id == patient_id)
    device = torch.device(args.device)
    target_shape = tuple(args.target_shape)
    training_patch_shape, foreground_probability = training_patch_options(args, target_shape)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    data = HipMRI3DDataset([pair], args.data_root, {patient_id}, num_classes, target_shape, training_patch_shape, args.prostate_label, foreground_probability)
    assessment_data = HipMRI3DDataset([pair], args.data_root, {patient_id}, num_classes, target_shape)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    assessment_loader = DataLoader(assessment_data, batch_size=1, shuffle=False)
    model = build_model(model_configuration(args, num_classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    for update in range(1, updates + 1):
        loss = _train_epoch(model, loader, optimizer, device, num_classes, args.prostate_label, getattr(args, "prostate_ce_weight", 8.0), getattr(args, "prostate_dice_weight", 8.0))
        if update % evaluation_interval == 0 or update == updates:
            metric = evaluate_3d_batches(model, assessment_loader, device, num_classes)
            history.append({"optimizer_updates": update, "train_loss": loss, "train_case_mean_dice": float(metric["mean_dice"])})
    image, target = assessment_data[0]
    with torch.no_grad():
        output_logits = model(image[None].to(device))
        logits = output_logits[0] if isinstance(output_logits, tuple) else output_logits
    prediction = logits.argmax(1).cpu()[0]
    diagnostics = prostate_diagnostics(prediction, target, args.prostate_label)
    summary = {"preflight_only": True, "patient_id": patient_id, "case_id": pair.case_id, "epochs": args.epochs, "optimizer_updates": updates, "evaluation_interval_updates": evaluation_interval, "history": history, "final_train_case_prostate_dice": diagnostics["prostate_dice"], "prostate_label": args.prostate_label, "prostate_label_source": args.prostate_label_source, "prostate_ce_weight": getattr(args, "prostate_ce_weight", 8.0), "prostate_dice_weight": getattr(args, "prostate_dice_weight", 8.0), "train_patch_shape": list(training_patch_shape) if training_patch_shape else None, "foreground_patch_probability": foreground_probability, "model": args.model, "seed": args.seed, "target_shape": list(target_shape), "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(), **diagnostics}
    summary.update(model_configuration(args, num_classes))
    _write_json(output / "preflight_summary.json", summary)
    return summary


def run_training(args: argparse.Namespace) -> dict[str, object]:
    """Train one declared architecture and persist validation-selected evidence only."""

    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0 or args.base_channels <= 0:
        raise ValueError("epochs, batch size, learning rate and base channels must be positive")
    selection_metric = getattr(args, "selection_metric", "mean_dice")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    target_shape = tuple(args.target_shape)
    training_patch_shape, foreground_probability = training_patch_options(args, target_shape)
    config = build_experiment_config(args.manifest, args.model, target_shape, args.seed, args.prostate_label, args.prostate_label_source)
    pairs, split_ids, num_classes = load_audited_pairs(args.manifest)
    device = torch.device(args.device)
    augmentation = AugmentationConfig() if getattr(args, "augmentation", False) else None
    scheduler_name = getattr(args, "scheduler", "none")
    eta_min = getattr(args, "eta_min", 1e-5)
    if scheduler_name not in {"none", "cosine"} or not 0 <= eta_min <= args.learning_rate:
        raise ValueError("invalid scheduler or eta_min")
    train_data = HipMRI3DDataset(pairs, args.data_root, split_ids["train"], num_classes, target_shape, training_patch_shape, args.prostate_label, foreground_probability, augmentation_config=augmentation, split="train")
    validation_data = HipMRI3DDataset(pairs, args.data_root, split_ids["validation"], num_classes, target_shape)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)
    config.update(model_configuration(args, num_classes))
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=eta_min) if scheduler_name == "cosine" else None
    output.mkdir(parents=True, exist_ok=True)
    config.update({"device": str(device), "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "base_channels": args.base_channels, "prostate_ce_weight": getattr(args, "prostate_ce_weight", 8.0), "prostate_dice_weight": getattr(args, "prostate_dice_weight", 8.0), "train_patch_shape": list(training_patch_shape) if training_patch_shape else None, "foreground_patch_probability": foreground_probability, "selection_metric": selection_metric})
    config.update({"augmentation_config": asdict(augmentation) if augmentation else None, "validation_augmentation": False, "test_augmentation": False, "scheduler": scheduler_name, "eta_min": eta_min, "scheduler_t_max": args.epochs, "lr_recording": "rate used for this epoch, scheduler steps after epoch"})
    _write_json(output / "experiment_config.json", config)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        _reset_cuda_peak(device)
        train_started = time.perf_counter()
        train_loss = _train_epoch(model, train_loader, optimizer, device, num_classes, args.prostate_label, getattr(args, "prostate_ce_weight", 8.0), getattr(args, "prostate_dice_weight", 8.0))
        if device.type == "cuda": torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - train_started
        train_peak_vram_mb = _cuda_peak_mb(device)
        validation = evaluate_3d_batches(model, validation_loader, device, num_classes, args.prostate_label, [pair.patient_id for pair in validation_data.records])
        row = {
            "epoch": float(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "train_seconds": train_seconds,
            "validation_mean_dice": float(validation["mean_dice"]),
            "validation_seconds": float(validation["inference_seconds"]),
            "train_peak_vram_mb": train_peak_vram_mb,
            "validation_peak_vram_mb": float(validation["peak_vram_mb"]),
            "validation_patient_prostate_dice": float(validation["validation_patient_prostate_dice"]),
            "validation_worst_patient_prostate_dice": float(validation["validation_worst_patient_prostate_dice"]),
            "validation_prostate_false_positive_voxels": float(validation["validation_prostate_false_positive_voxels"]),
            "validation_prostate_false_negative_voxels": float(validation["validation_prostate_false_negative_voxels"]),
            "validation_patient_prostate_dice_by_patient": validation["validation_patient_prostate_dice_by_patient"],
        }
        history.append(row)
        _write_json(output / "history.json", history)
        render_training_curves({args.model: history}, output / "training_curves.png", selection_metric)
        render_learning_rate_curve({args.model: history}, output / "learning_rate_schedule.png")
        if scheduler is not None:
            scheduler.step()
        if select_validation_checkpoint(history, selection_metric)["epoch"] == row["epoch"]:
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict() if scheduler else None, "epoch": epoch, "config": config}, output / "best.pt")
    selected = select_validation_checkpoint(history, selection_metric)
    summary = {
        "selected_epoch": selected["epoch"],
        "selection_metric": selection_metric,
        "selection_value": selected["validation_patient_prostate_dice"] if selection_metric == "prostate_dice" else selected["validation_mean_dice"],
        "training_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_train_vram_mb": max(row["train_peak_vram_mb"] for row in history),
        "peak_validation_vram_mb": max(row["validation_peak_vram_mb"] for row in history),
        "history": history,
    }
    _write_json(output / "training_summary.json", summary)
    return summary


def main() -> None:
    args = create_argument_parser().parse_args()
    runner = run_overfit_preflight if args.preflight_patient_id else run_training
    print(json.dumps(runner(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
