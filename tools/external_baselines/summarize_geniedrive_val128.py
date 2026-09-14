#!/usr/bin/env python3
"""Print and save a compact summary of a GenieDrive val128 evaluation."""
import argparse
import json
from pathlib import Path


HORIZONS = ("1.0", "2.0", "3.0")
METRICS = (
    ("Overall mIoU", "overall"),
    ("Dynamic mIoU", "dynamic"),
    ("Moving-mIoU v2", "Moving-mIoU_v2"),
)


def read_json(path, required=True):
    path = Path(path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(str(path))
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric_values(model, metric_key):
    metric = model.get(metric_key)
    if not isinstance(metric, dict):
        return None
    per_horizon = metric.get("per_horizon") or {}
    values = []
    for horizon in HORIZONS:
        row = per_horizon.get(horizon) or per_horizon.get(str(float(horizon))) or {}
        values.append(row.get("mIoU"))
    values.append(metric.get("mIoU"))
    return values


def difference(left, right):
    if left is None or right is None:
        return None
    return [
        None if a is None or b is None else float(a) - float(b)
        for a, b in zip(left, right)
    ]


def fmt(value, signed=False):
    if value is None:
        return "--"
    return ("%+.3f" if signed else "%.3f") % float(value)


def metric_rows(report):
    baseline_names = [
        name for name in ("Strong-W2Det_baseline", "KTA_composed_baseline")
        if name in report
    ]
    baseline_name = baseline_names[0] if baseline_names else None
    rows = []
    compact = {}
    for metric_label, metric_key in METRICS:
        genie = metric_values(report["GenieDrive"], metric_key)
        if genie is None:
            continue
        compact.setdefault(metric_key, {})["GenieDrive"] = genie
        rows.append((metric_label, "GenieDrive", genie, False))
        if baseline_name:
            baseline = metric_values(report[baseline_name], metric_key)
            if baseline is not None:
                delta = difference(genie, baseline)
                compact[metric_key][baseline_name] = baseline
                compact[metric_key]["GenieDrive_minus_baseline"] = delta
                rows.append(("", baseline_name.replace("_baseline", ""), baseline, False))
                rows.append(("", "Delta", delta, True))
    return rows, compact, baseline_name


def render_table(rows):
    headers = ("Metric", "Model", "1s", "2s", "3s", "Mean")
    body = []
    for metric, model, values, signed in rows:
        body.append((metric, model) + tuple(fmt(value, signed=signed) for value in values))
    widths = [len(value) for value in headers]
    for row in body:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    lines = ["  ".join(value.ljust(widths[i]) for i, value in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    for row in body:
        lines.append("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))
    return "\n".join(lines)


def markdown_table(rows):
    lines = [
        "| Metric | Model | 1s | 2s | 3s | Mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for metric, model, values, signed in rows:
        cells = [fmt(value, signed=signed) for value in values]
        lines.append("| %s | %s | %s |" % (metric, model, " | ".join(cells)))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="/root/nas/occ/external_baselines/geniedrive_val128/results",
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()

    root = Path(args.results)
    manifest = read_json(root / "val128_manifest.json")
    selection = read_json(root / "selection_audit.json")
    native = read_json(root / "native_metric_sanity.json")
    inference = read_json(root / "inference_summary.json")
    report = read_json(root / "geniedrive_val128_moving_miou_v2.json")

    rows, metrics, baseline_name = metric_rows(report)
    protocol_passed = (
        manifest.get("num_samples") == 128
        and manifest.get("num_scenes") == 128
        and bool(selection.get("passed"))
        and selection.get("selected_tokens_found") == 128
        and selection.get("selected_occ_labels_found") == 128
    )
    native_passed = bool(native.get("passed"))
    inference_passed = (
        inference.get("num_samples") == 128
        and report.get("num_predictions_used") == 128
        and bool(report.get("complete_prediction_set"))
    )
    all_passed = protocol_passed and native_passed and inference_passed

    native_values = native.get("semantic_mIoU") or []
    elapsed = inference.get("elapsed_seconds")
    print("GenieDrive val128 final result")
    print("=" * 32)
    print(
        "Protocol:  %s | samples=%s scenes=%s | %s"
        % (
            "PASS" if protocol_passed else "FAIL",
            manifest.get("num_samples"),
            manifest.get("num_scenes"),
            manifest.get("selection_contract"),
        )
    )
    print(
        "Native:    %s | valid=%s | official-val mIoU=%s"
        % (
            "PASS" if native_passed else "FAIL",
            native.get("valid_samples"),
            "/".join(fmt(value) for value in native_values),
        )
    )
    print(
        "Inference: %s | saved=%s | elapsed=%s min"
        % (
            "PASS" if inference_passed else "FAIL",
            inference.get("num_samples"),
            fmt(float(elapsed) / 60.0) if elapsed is not None else "--",
        )
    )
    print()
    print(render_table(rows))

    compact = {
        "version": "swfm_geniedrive_val128_final_summary_v1",
        "all_gates_passed": all_passed,
        "gates": {
            "protocol": protocol_passed,
            "native_metric_reproduction": native_passed,
            "complete_inference": inference_passed,
        },
        "num_samples": manifest.get("num_samples"),
        "selection_contract": manifest.get("selection_contract"),
        "sample_ids_sha256": manifest.get("sample_ids_sha256"),
        "checkpoint_sha256": inference.get("checkpoint_sha256"),
        "geniedrive_revision": inference.get("geniedrive_revision"),
        "elapsed_seconds": elapsed,
        "baseline": baseline_name,
        "metrics": metrics,
    }
    json_output = Path(args.json_output) if args.json_output else root / "final_summary.json"
    markdown_output = (
        Path(args.markdown_output) if args.markdown_output else root / "final_summary.md"
    )
    json_output.write_text(json.dumps(compact, indent=2), encoding="utf-8")
    markdown = "# GenieDrive val128 final result\n\n"
    markdown += "All gates: **%s**\n\n" % ("PASS" if all_passed else "FAIL")
    markdown += markdown_table(rows) + "\n"
    markdown_output.write_text(markdown, encoding="utf-8")
    print()
    print("All gates: %s" % ("PASS" if all_passed else "FAIL"))
    print("Saved: %s" % json_output)
    print("Saved: %s" % markdown_output)
    if not all_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
