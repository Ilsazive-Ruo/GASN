"""Predict molecule pairs with saved 10-fold models and print synergistic hits."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dcp_common import (
    pair_features_from_frame,
    predict_proba_or_none,
    project_path,
    read_molecule_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True, help="directory created by train_10fold_models.py")
    parser.add_argument("--input", required=True, help="pair CSV with id-1/id-2 or smiles-1/smiles-2")
    parser.add_argument("--output", default="outputs/ten_fold_predictions.csv", help="prediction CSV path")
    parser.add_argument("--mol_file", default=None, help="optional molecule source for id-based pair files")
    parser.add_argument("--positive_label", default="synergistic", help="label to print in terminal")
    return parser.parse_args()


def load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata.json in {model_dir}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_model_dir(value) -> Path:
    raw_path = Path(value)
    candidates = [project_path(raw_path), raw_path]

    parts_lower = [part.lower() for part in raw_path.parts]
    if "scripts_outputs" in parts_lower:
        scripts_outputs_idx = parts_lower.index("scripts_outputs")
        tail = Path(*raw_path.parts[scripts_outputs_idx + 1 :])
        candidates.append(SCRIPT_DIR.parent / "scripts_outputs" / tail)

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if (candidate / "metadata.json").exists():
            return candidate

    checked_text = "\n  ".join(str(path) for path in checked)
    raise FileNotFoundError(
        "missing metadata.json. Checked these model directories:\n"
        f"  {checked_text}\n"
        "Use the directory created by train_10fold_models.py, for example: "
        "DrugCombination\\scripts_outputs\\ten_fold_models\\xgb_des"
    )


def align_proba(model, raw_proba: np.ndarray, class_count: int) -> np.ndarray:
    model_classes = getattr(model, "classes_", None)
    if model_classes is None and hasattr(model, "named_steps"):
        model_classes = getattr(model.named_steps["model"], "classes_", None)
    if model_classes is None:
        if raw_proba.shape[1] != class_count:
            raise ValueError("model probability columns cannot be aligned to label encoder classes")
        return raw_proba

    aligned = np.zeros((raw_proba.shape[0], class_count), dtype=float)
    for raw_col, class_id in enumerate(model_classes):
        class_id = int(class_id)
        if 0 <= class_id < class_count:
            aligned[:, class_id] = raw_proba[:, raw_col]
    return aligned


def choose_display_columns(result: pd.DataFrame, positive_label: str) -> list:
    preferred = [
        "id-1",
        "name-1",
        "smiles-1",
        "id-2",
        "name-2",
        "smiles-2",
        "pred_label",
        "score",
        "score_sd",
        f"proba_{positive_label}",
        f"proba_{positive_label}_sd",
    ]
    fold_columns = [
        column
        for column in result.columns
        if column.startswith("fold_")
        and (
            column.endswith("_pred_label")
            or column.endswith("_score")
            or column.endswith(f"_proba_{positive_label}")
        )
    ]
    return [column for column in preferred if column in result.columns] + fold_columns


def make_pair_labels(frame: pd.DataFrame) -> pd.Series:
    if {"name-1", "name-2"}.issubset(frame.columns):
        return frame["name-1"].astype(str) + " + " + frame["name-2"].astype(str)
    if {"id-1", "id-2"}.issubset(frame.columns):
        return frame["id-1"].astype(str) + " + " + frame["id-2"].astype(str)
    return pd.Series([f"pair_{idx + 1}" for idx in range(len(frame))], index=frame.index)


def save_graphpad_positive_table(
    result: pd.DataFrame,
    output: Path,
    positive_label: str,
) -> tuple:
    positive_rows = result[result["pred_label"] == positive_label].copy()
    csv_path = output.with_name(f"{output.stem}_{positive_label}_graphpad_scores.csv")
    tsv_path = output.with_name(f"{output.stem}_{positive_label}_graphpad_scores.tsv")

    if positive_rows.empty:
        graphpad_df = pd.DataFrame(
            columns=[
                "sample",
                f"mean_{positive_label}_score",
                f"sd_{positive_label}_score",
                "n_models",
            ]
        )
    else:
        fold_proba_columns = [
            column
            for column in positive_rows.columns
            if column.startswith("fold_") and column.endswith(f"_proba_{positive_label}")
        ]
        fold_output_columns = [column.replace(f"_proba_{positive_label}", "") for column in fold_proba_columns]
        graphpad_df = pd.DataFrame(
            {
                "sample": make_pair_labels(positive_rows),
                "pred_label": positive_rows["pred_label"],
                f"mean_{positive_label}_score": positive_rows[f"proba_{positive_label}"],
                f"sd_{positive_label}_score": positive_rows[f"proba_{positive_label}_sd"],
                "mean_predicted_class_score": positive_rows["score"],
                "sd_predicted_class_score": positive_rows["score_sd"],
                "n_models": positive_rows["n_models"],
            }
        )
        for source_column, output_column in zip(fold_proba_columns, fold_output_columns):
            graphpad_df[output_column] = positive_rows[source_column]
        graphpad_df = graphpad_df.sort_values(f"mean_{positive_label}_score", ascending=False)

    graphpad_df.to_csv(csv_path, index=False)
    graphpad_df.to_csv(tsv_path, index=False, sep="\t")
    return csv_path, tsv_path, graphpad_df


def main() -> None:
    args = parse_args()
    model_dir = resolve_model_dir(args.model_dir)
    metadata = load_metadata(model_dir)
    label_encoder = joblib.load(model_dir / "label_encoder.joblib")
    feature_type = metadata.get("feature_type", "des")
    fp_radius = int(metadata.get("fp_radius", 4))
    fp_bits = int(metadata.get("fp_bits", 2048))
    model_files = metadata.get("model_files") or sorted(path.name for path in model_dir.glob("fold_*_model.joblib"))
    if not model_files:
        raise ValueError(f"no fold models found in {model_dir}")

    pairs = pd.read_csv(project_path(args.input))
    mol_info = None
    mol_file = args.mol_file or metadata.get("mol_file")
    if {"id-1", "id-2"}.issubset(pairs.columns) and mol_file:
        mol_info = read_molecule_source(mol_file, feature_type, fp_radius=fp_radius, fp_bits=fp_bits)

    x, kept_pairs = pair_features_from_frame(
        pairs,
        feature_type=feature_type,
        mol_info=mol_info,
        fp_radius=fp_radius,
        fp_bits=fp_bits,
    )

    class_count = len(label_encoder.classes_)
    proba_sum = np.zeros((x.shape[0], class_count), dtype=float)
    used_models = 0
    fold_predictions = {}
    fold_score_arrays = []
    fold_positive_proba_arrays = []
    positive_class_idx = None
    if args.positive_label in label_encoder.classes_:
        positive_class_idx = int(np.where(label_encoder.classes_ == args.positive_label)[0][0])

    for fold_idx, model_file in enumerate(model_files, start=1):
        model = joblib.load(model_dir / model_file)
        raw_proba = predict_proba_or_none(model, x)
        if raw_proba is None:
            raise ValueError(f"{model_file} does not support predict_proba; cannot compute prediction scores")
        aligned_proba = align_proba(model, raw_proba, class_count)
        proba_sum += aligned_proba
        fold_pred_ids = aligned_proba.argmax(axis=1)
        fold_prefix = f"fold_{fold_idx:02d}"
        fold_predictions[f"{fold_prefix}_model_file"] = model_file
        fold_predictions[f"{fold_prefix}_pred_id"] = fold_pred_ids
        fold_predictions[f"{fold_prefix}_pred_label"] = label_encoder.inverse_transform(fold_pred_ids.astype(int))
        fold_predictions[f"{fold_prefix}_score"] = aligned_proba.max(axis=1)
        fold_score_arrays.append(aligned_proba.max(axis=1))
        if positive_class_idx is not None:
            positive_proba = aligned_proba[:, positive_class_idx]
            fold_predictions[f"{fold_prefix}_proba_{args.positive_label}"] = positive_proba
            fold_positive_proba_arrays.append(positive_proba)
        used_models += 1

    mean_proba = proba_sum / used_models
    pred_ids = mean_proba.argmax(axis=1)
    pred_labels = label_encoder.inverse_transform(pred_ids.astype(int))
    result = kept_pairs.copy()
    for column, values in fold_predictions.items():
        result[column] = values
    result["pred_id"] = pred_ids
    result["pred_label"] = pred_labels
    for idx, class_name in enumerate(label_encoder.classes_):
        result[f"proba_{class_name}"] = mean_proba[:, idx]
    result["score"] = mean_proba.max(axis=1)
    fold_score_matrix = np.vstack(fold_score_arrays).T
    result["score_sd"] = fold_score_matrix.std(axis=1, ddof=1) if used_models > 1 else 0.0
    if positive_class_idx is not None:
        fold_positive_proba_matrix = np.vstack(fold_positive_proba_arrays).T
        result[f"proba_{args.positive_label}_sd"] = (
            fold_positive_proba_matrix.std(axis=1, ddof=1) if used_models > 1 else 0.0
        )
    result["n_models"] = used_models

    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved predictions for {len(result)} pairs to {output}")
    graphpad_csv, graphpad_tsv, graphpad_df = save_graphpad_positive_table(result, output, args.positive_label)
    print(f"Saved GraphPad-friendly {args.positive_label} scores to {graphpad_csv}")
    print(f"Saved tab-delimited copy/paste table to {graphpad_tsv}")

    positive_rows = result[result["pred_label"] == args.positive_label].copy()
    print(f"\nPredicted {args.positive_label}: {len(positive_rows)} rows")
    if positive_rows.empty:
        return

    display_columns = choose_display_columns(positive_rows, args.positive_label)
    positive_rows = positive_rows.sort_values("score", ascending=False)
    print(positive_rows[display_columns].to_string(index=False))
    print(f"\nGraphPad-friendly {args.positive_label} score table (tab-delimited):")
    print(graphpad_df.to_csv(index=False, sep="\t").strip())


if __name__ == "__main__":
    main()
