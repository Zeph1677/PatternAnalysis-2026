"""Create deterministic paired failure panels for matched formal predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import torch


_COMPATIBILITY_FIELDS = (
    "manifest_sha256",
    "prostate_label",
    "prostate_label_source",
    "target_shape",
    "patch_size",
    "stride",
)


def validate_comparison_inputs(
    baseline: dict[str, object], improved: dict[str, object]
) -> None:
    """Reject result sets that do not support a matched comparison."""

    for field in _COMPATIBILITY_FIELDS:
        if baseline.get(field) != improved.get(field):
            raise ValueError(f"incompatible {field}")
    baseline_patients = set(baseline.get("per_patient_prostate_dice", {}))
    improved_patients = set(improved.get("per_patient_prostate_dice", {}))
    if baseline_patients != improved_patients:
        raise ValueError("incompatible test patient sets")


def select_unique_patient_cases(
    ranking: list[dict[str, object]], max_patients: int
) -> list[dict[str, object]]:
    """Retain the highest-ranked scan for each patient."""

    if max_patients <= 0:
        raise ValueError("max_patients must be positive")
    selected: list[dict[str, object]] = []
    seen_patients: set[str] = set()
    for case in ranking:
        patient_id = str(case["patient_id"])
        if patient_id in seen_patients:
            continue
        selected.append(dict(case))
        seen_patients.add(patient_id)
        if len(selected) == max_patients:
            break
    return selected


def common_error_score(
    baseline_counts: dict[str, int], improved_counts: dict[str, int]
) -> float:
    """Return mean whole-volume FP+FN over both model predictions."""

    return (
        baseline_counts["false_positive_voxels"]
        + baseline_counts["false_negative_voxels"]
        + improved_counts["false_positive_voxels"]
        + improved_counts["false_negative_voxels"]
    ) / 2.0


def error_relevant_prostate_slice(
    target: torch.Tensor,
    baseline: torch.Tensor,
    improved: torch.Tensor,
    prostate_label: int,
) -> int:
    """Choose the highest paired-error depth slice inside the reference extent."""

    extent = torch.nonzero(
        (target == prostate_label).any(dim=(1, 2)), as_tuple=False
    ).flatten()
    if not len(extent):
        raise ValueError("reference mask contains no prostate voxels")
    scores = []
    for index in extent.tolist():
        reference = target[index] == prostate_label
        errors = ((baseline[index] == prostate_label) ^ reference).sum()
        errors += ((improved[index] == prostate_label) ^ reference).sum()
        scores.append((int(errors), int(index)))
    return max(scores)[1]


def prostate_extent_endpoints(
    target: torch.Tensor, prostate_label: int
) -> dict[str, int]:
    """Return mask-derived inferior/apex-side and superior/base-side slices."""

    extent = torch.nonzero(
        (target == prostate_label).any(dim=(1, 2)), as_tuple=False
    ).flatten()
    if not len(extent):
        raise ValueError("reference mask contains no prostate voxels")
    return {
        "apex_side": int(extent.min()),
        "base_side": int(extent.max()),
    }


def slice_error_counts(
    prediction: torch.Tensor,
    target: torch.Tensor,
    label: int,
    slice_index: int,
) -> dict[str, int]:
    """Count prostate false positives and false negatives on one depth slice."""

    predicted = prediction[slice_index] == label
    reference = target[slice_index] == label
    return {
        "false_positive_voxels": int((predicted & ~reference).sum()),
        "false_negative_voxels": int((~predicted & reference).sum()),
    }


def _counts(
    prediction: torch.Tensor, target: torch.Tensor, label: int
) -> dict[str, int]:
    predicted = prediction == label
    reference = target == label
    true_positive = int((predicted & reference).sum())
    return {
        "true_positive_voxels": true_positive,
        "false_positive_voxels": int(predicted.sum()) - true_positive,
        "false_negative_voxels": int(reference.sum()) - true_positive,
    }


def rank_failure_cases(
    baseline_predictions: dict[str, dict[str, object]],
    improved_predictions: dict[str, dict[str, object]],
    prostate_label: int,
) -> list[dict[str, object]]:
    """Rank matched scans by mean paired whole-volume prostate error."""

    if set(baseline_predictions) != set(improved_predictions):
        raise ValueError("incompatible prediction case sets")
    ranking: list[dict[str, object]] = []
    for case_id in sorted(baseline_predictions):
        baseline = baseline_predictions[case_id]
        improved = improved_predictions[case_id]
        for field in (
            "case_id",
            "patient_id",
            "manifest_sha256",
            "prostate_label",
            "target_shape",
        ):
            if baseline.get(field) != improved.get(field):
                raise ValueError(f"incompatible prediction {field} for {case_id}")
        if baseline.get("case_id") != case_id:
            raise ValueError(f"prediction case ID does not match filename for {case_id}")
        if baseline.get("prostate_label") != prostate_label:
            raise ValueError(
                f"prediction prostate label does not match comparison label for {case_id}"
            )
        if not torch.equal(baseline["target"], improved["target"]):
            raise ValueError(f"incompatible case payload for {case_id}")
        baseline_counts = _counts(
            baseline["prediction"], baseline["target"], prostate_label
        )
        improved_counts = _counts(
            improved["prediction"], improved["target"], prostate_label
        )
        ranking.append(
            {
                "case_id": case_id,
                "patient_id": baseline["patient_id"],
                "error_score": common_error_score(
                    baseline_counts, improved_counts
                ),
                "slice_index": error_relevant_prostate_slice(
                    baseline["target"],
                    baseline["prediction"],
                    improved["prediction"],
                    prostate_label,
                ),
                "baseline": baseline_counts,
                "improved": improved_counts,
            }
        )
    return sorted(
        ranking,
        key=lambda row: (
            -float(row["error_score"]),
            str(row["patient_id"]),
            str(row["case_id"]),
        ),
    )


def rank_endpoint_cases(
    baseline_predictions: dict[str, dict[str, object]],
    improved_predictions: dict[str, dict[str, object]],
    prostate_label: int,
    allowed_case_ids: set[str] | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Rank mask-derived apex-side and base-side slices by paired error."""

    if set(baseline_predictions) != set(improved_predictions):
        raise ValueError("incompatible prediction case sets")
    ranked: dict[str, list[dict[str, object]]] = {
        "apex_side": [],
        "base_side": [],
    }
    for case_id in sorted(baseline_predictions):
        if allowed_case_ids is not None and case_id not in allowed_case_ids:
            continue
        baseline = baseline_predictions[case_id]
        improved = improved_predictions[case_id]
        for field in (
            "case_id",
            "patient_id",
            "manifest_sha256",
            "prostate_label",
            "target_shape",
        ):
            if baseline.get(field) != improved.get(field):
                raise ValueError(f"incompatible prediction {field} for {case_id}")
        if baseline.get("case_id") != case_id:
            raise ValueError(f"prediction case ID does not match filename for {case_id}")
        if baseline.get("prostate_label") != prostate_label:
            raise ValueError(
                f"prediction prostate label does not match comparison label for {case_id}"
            )
        if not torch.equal(baseline["target"], improved["target"]):
            raise ValueError(f"incompatible case payload for {case_id}")
        endpoints = prostate_extent_endpoints(baseline["target"], prostate_label)
        for endpoint, slice_index in endpoints.items():
            baseline_counts = slice_error_counts(
                baseline["prediction"],
                baseline["target"],
                prostate_label,
                slice_index,
            )
            improved_counts = slice_error_counts(
                improved["prediction"],
                improved["target"],
                prostate_label,
                slice_index,
            )
            paired_error_voxels = sum(baseline_counts.values()) + sum(
                improved_counts.values()
            )
            ranked[endpoint].append(
                {
                    "endpoint": endpoint,
                    "case_id": case_id,
                    "patient_id": baseline["patient_id"],
                    "slice_index": slice_index,
                    "paired_error_voxels": paired_error_voxels,
                    "baseline": baseline_counts,
                    "improved": improved_counts,
                }
            )
    for endpoint in ranked:
        ranked[endpoint].sort(
            key=lambda row: (
                -int(row["paired_error_voxels"]),
                str(row["patient_id"]),
                str(row["case_id"]),
            )
        )
    return ranked


def render_comparative_prostate_panel(
    image: torch.Tensor,
    target: torch.Tensor,
    baseline: torch.Tensor,
    improved: torch.Tensor,
    prostate_label: int,
    output_path: Path | str,
    slice_index: int | None = None,
) -> int:
    """Write one aligned MRI/reference/prediction/FP-FN comparison panel."""

    index = (
        error_relevant_prostate_slice(target, baseline, improved, prostate_label)
        if slice_index is None
        else slice_index
    )
    source = image[0] if image.ndim == 4 else image
    if source.shape != target.shape:
        raise ValueError("image and segmentation shapes must match")
    reference = target[index] == prostate_label
    baseline_errors = torch.zeros_like(target[index], dtype=torch.uint8)
    improved_errors = torch.zeros_like(target[index], dtype=torch.uint8)
    baseline_errors[(baseline[index] == prostate_label) & ~reference] = 1
    baseline_errors[(baseline[index] != prostate_label) & reference] = 2
    improved_errors[(improved[index] == prostate_label) & ~reference] = 1
    improved_errors[(improved[index] != prostate_label) & reference] = 2

    figure, axes = plt.subplots(1, 6, figsize=(18, 3))
    label_norm = BoundaryNorm([value - 0.5 for value in range(7)], 20)
    error_cmap = ListedColormap(["#252525", "#ffffff", "#e31a1c"])
    error_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
    panels = (
        (source[index], "MRI", "gray"),
        (target[index], "Ground truth", "tab20"),
        (baseline[index], "Baseline", "tab20"),
        (improved[index], "Improved", "tab20"),
        (baseline_errors, "Baseline FP/FN", error_cmap),
        (improved_errors, "Improved FP/FN", error_cmap),
    )
    for number, (axis, (data, title, colour_map)) in enumerate(zip(axes, panels)):
        options = {} if number == 0 else {"norm": label_norm}
        if number >= 4:
            options = {"norm": error_norm}
        axis.imshow(data.numpy(), cmap=colour_map, interpolation="nearest", **options)
        axis.set_title(title)
        axis.axis("off")
    figure.legend(
        handles=[
            Patch(facecolor="#252525", label="No prostate error"),
            Patch(facecolor="#ffffff", edgecolor="black", label="FP"),
            Patch(facecolor="#e31a1c", label="FN"),
        ],
        loc="lower center",
        ncol=3,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return index


def build_selection_manifest(
    selected_cases: list[dict[str, object]],
    source_identity: str,
    endpoint_cases: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Record the deterministic rule and source identity with selected cases."""

    if not source_identity.strip():
        raise ValueError("selection manifest requires a source identity")
    manifest = {
        "schema_version": 1,
        "source_identity": source_identity,
        "selection_rule": (
            "rank scans by mean whole-volume prostate FP+FN across baseline and "
            "improved; retain the highest-ranked scan per patient; patient ID and "
            "case ID break ties; choose the highest paired-error slice inside the "
            "reference prostate extent"
        ),
        "selected_cases": selected_cases,
    }
    if endpoint_cases is not None:
        manifest["endpoint_definition"] = (
            "after verifying that tensor depth increases superiorly, the first "
            "non-empty reference-prostate slice is the apex-side endpoint and the "
            "last is the base-side endpoint; these are mask-derived boundaries, "
            "not independently annotated clinical landmarks"
        )
        manifest["endpoint_cases"] = endpoint_cases
    return manifest


def _load_predictions(directory: Path) -> dict[str, dict[str, object]]:
    return {
        path.stem: torch.load(path, map_location="cpu", weights_only=False)
        for path in sorted((directory / "predictions").glob("*.pt"))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--improved-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-patients", type=int, default=5)
    args = parser.parse_args()

    baseline_metrics = json.loads(
        (args.baseline_dir / "test_metrics.json").read_text(encoding="utf-8")
    )
    improved_metrics = json.loads(
        (args.improved_dir / "test_metrics.json").read_text(encoding="utf-8")
    )
    validate_comparison_inputs(baseline_metrics, improved_metrics)
    baseline_predictions = _load_predictions(args.baseline_dir)
    improved_predictions = _load_predictions(args.improved_dir)
    ranking = rank_failure_cases(
        baseline_predictions,
        improved_predictions,
        int(baseline_metrics["prostate_label"]),
    )
    selected = select_unique_patient_cases(ranking, args.max_patients)
    endpoint_ranking = rank_endpoint_cases(
        baseline_predictions,
        improved_predictions,
        int(baseline_metrics["prostate_label"]),
        allowed_case_ids={str(entry["case_id"]) for entry in selected},
    )
    endpoint_cases = {
        endpoint: dict(rows[0]) for endpoint, rows in endpoint_ranking.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for entry in selected:
        case_id = str(entry["case_id"])
        payload = baseline_predictions[case_id]
        panel = Path("comparison_panels") / f"{entry['patient_id']}__{case_id}.png"
        entry["panel"] = panel.as_posix()
        entry["slice_index"] = render_comparative_prostate_panel(
            payload["image"],
            payload["target"],
            payload["prediction"],
            improved_predictions[case_id]["prediction"],
            int(baseline_metrics["prostate_label"]),
            args.output_dir / panel,
        )
    for endpoint, entry in endpoint_cases.items():
        case_id = str(entry["case_id"])
        payload = baseline_predictions[case_id]
        panel = Path("endpoint_panels") / f"{endpoint}__{case_id}.png"
        entry["panel"] = panel.as_posix()
        render_comparative_prostate_panel(
            payload["image"],
            payload["target"],
            payload["prediction"],
            improved_predictions[case_id]["prediction"],
            int(baseline_metrics["prostate_label"]),
            args.output_dir / panel,
            slice_index=int(entry["slice_index"]),
        )
    source_identity = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = build_selection_manifest(selected, source_identity, endpoint_cases)
    (args.output_dir / "failure_selection.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
