"""Cold-start model comparison.

This script compares ADA, BNB, KNN, DT, and XGB under drug-level cold-start
splits. Each fold holds out a group of drugs. Training pairs contain no held-out
drugs; test pairs contain held-out drugs, so the metrics measure cold-start
generalization rather than ordinary pair-level cross-validation.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dcp_common import (
    MODEL_LABELS,
    MODEL_NAMES,
    add_common_data_args,
    build_pipeline,
    collect_curve_data,
    default_param_grid,
    encode_labels,
    ensure_dir,
    evaluate_predictions,
    load_json_arg,
    normalize_param_grids,
    parse_csv_paths,
    predict_proba_or_none,
    read_molecule_source,
    read_training_data,
    save_curve_outputs,
    serialize_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_data_args(parser)
    parser.add_argument(
        "--cold_start_mode",
        choices=("any", "single", "double"),
        default="double",
        help=(
            "test pair definition: any=at least one held-out drug, "
            "single=exactly one held-out drug, double=both drugs held out"
        ),
    )
    parser.add_argument("--grid_search", action="store_true", help="run GridSearchCV inside each training fold")
    parser.add_argument(
        "--param_grids",
        default=None,
        help="JSON string or JSON file mapping model names to GridSearchCV grids",
    )
    parser.add_argument("--inner_cv", type=int, default=3, help="inner GridSearchCV folds")
    parser.add_argument("--scoring", default="f1_macro", help="GridSearchCV scoring name")
    parser.add_argument("--n_jobs", type=int, default=-1, help="GridSearchCV parallel jobs")
    parser.add_argument("--k_folds", type=int, default=10, help="number of drug-level cold-start folds")
    parser.add_argument("--random_state", type=int, default=99, help="random seed")
    parser.add_argument("--out_dir", default="cold_start_double", help="output directory")
    parser.add_argument("--save_fold_models", action="store_true", help="save each fold model")
    return parser.parse_args()


def make_cold_start_splits(pair_rows, k_folds, random_state, mode):
    pair_ids = pair_rows[["id-1", "id-2"]].astype(str).reset_index(drop=True)
    unique_drugs = np.unique(pair_ids[["id-1", "id-2"]].values.ravel())
    splitter = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    splits = []

    for fold_idx, (known_drug_idx, cold_drug_idx) in enumerate(splitter.split(unique_drugs), start=1):
        cold_drugs = set(unique_drugs[cold_drug_idx])
        id_1_cold = pair_ids["id-1"].isin(cold_drugs).to_numpy()
        id_2_cold = pair_ids["id-2"].isin(cold_drugs).to_numpy()

        train_mask = ~(id_1_cold | id_2_cold)
        if mode == "any":
            test_mask = id_1_cold | id_2_cold
        elif mode == "single":
            test_mask = id_1_cold ^ id_2_cold
        elif mode == "double":
            test_mask = id_1_cold & id_2_cold
        else:
            raise ValueError(f"unsupported cold_start_mode={mode!r}")

        splits.append(
            {
                "fold": fold_idx,
                "train_idx": np.flatnonzero(train_mask),
                "test_idx": np.flatnonzero(test_mask),
                "n_cold_drugs": len(cold_drugs),
                "cold_drugs": sorted(cold_drugs),
            }
        )
    return splits


def fit_fold_model(args, model_name, x_train, y_train, out_dir, fold_idx, param_grids):
    base_model = build_pipeline(model_name, random_state=args.random_state)
    best_params = {}
    best_inner_score = None

    if args.grid_search:
        _, class_counts = np.unique(y_train, return_counts=True)
        inner_cv = min(args.inner_cv, int(class_counts.min()))
        if inner_cv < 2:
            base_model.fit(x_train, y_train)
            return base_model, best_params, best_inner_score

        fold_grid = param_grids.get(model_name) or default_param_grid(model_name)
        if not fold_grid:
            raise ValueError(f"No GridSearchCV grid is configured for model={model_name!r}")
        search = GridSearchCV(
            estimator=base_model,
            param_grid=fold_grid,
            scoring=args.scoring,
            cv=StratifiedKFold(
                n_splits=inner_cv,
                shuffle=True,
                random_state=args.random_state,
            ),
            n_jobs=args.n_jobs,
            refit=True,
            verbose=1,
        )
        search.fit(x_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        best_inner_score = float(search.best_score_)
        pd.DataFrame(search.cv_results_).to_csv(
            out_dir / f"{model_name}_fold_{fold_idx}_grid_results.csv",
            index=False,
        )
    else:
        model = base_model
        model.fit(x_train, y_train)

    return model, best_params, best_inner_score


def predict_proba_for_classes(model, x, class_ids):
    raw_proba = predict_proba_or_none(model, x)
    if raw_proba is None:
        return None

    model_classes = getattr(model, "classes_", None)
    if model_classes is None and hasattr(model, "named_steps"):
        model_classes = getattr(model.named_steps["model"], "classes_", None)
    if model_classes is None:
        return raw_proba

    aligned = np.zeros((raw_proba.shape[0], len(class_ids)))
    class_to_col = {int(class_id): idx for idx, class_id in enumerate(class_ids)}
    for raw_col, class_id in enumerate(model_classes):
        target_col = class_to_col.get(int(class_id))
        if target_col is not None:
            aligned[:, target_col] = raw_proba[:, raw_col]
    return aligned


def run_model_cold_start(args, x, y, pair_rows, label_encoder, model_name, splits, out_dir, param_grids):
    class_ids = np.arange(len(label_encoder.classes_))
    class_names = list(label_encoder.classes_)
    rows = []
    curve_data = []

    for split in splits:
        fold_idx = split["fold"]
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]

        if len(train_idx) == 0 or len(test_idx) == 0:
            print(f"  {model_name} fold {fold_idx}: skipped empty train/test split")
            continue
        if len(np.unique(y[train_idx])) < 2:
            print(f"  {model_name} fold {fold_idx}: skipped because training split has <2 classes")
            continue

        model, best_params, best_inner_score = fit_fold_model(
            args,
            model_name,
            x[train_idx],
            y[train_idx],
            out_dir,
            fold_idx,
            param_grids,
        )
        y_pred = model.predict(x[test_idx])
        y_proba = predict_proba_for_classes(model, x[test_idx], class_ids)
        collect_curve_data(
            curve_data,
            y[test_idx],
            y_proba,
            class_ids,
            class_names,
            fold_idx,
            {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "feature_type": args.feat,
                "cold_start_mode": args.cold_start_mode,
            },
        )

        row = {
            "fold": fold_idx,
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "feature_type": args.feat,
            "cold_start_mode": args.cold_start_mode,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "n_cold_drugs": int(split["n_cold_drugs"]),
            "grid_search": bool(args.grid_search),
            "inner_cv": int(args.inner_cv) if args.grid_search else None,
            "grid_scoring": args.scoring if args.grid_search else None,
            "best_inner_score": best_inner_score,
            "best_params": serialize_params(best_params),
        }
        row.update(evaluate_predictions(y[test_idx], y_pred, y_proba, class_ids, class_names))
        rows.append(row)

        pd.crosstab(
            pd.Series(label_encoder.inverse_transform(y[test_idx]), name="true"),
            pd.Series(label_encoder.inverse_transform(y_pred.astype(int)), name="pred"),
            dropna=False,
        ).to_csv(out_dir / f"{model_name}_fold_{fold_idx}_confusion_matrix.csv")

        if args.save_fold_models:
            joblib.dump(model, out_dir / f"{model_name}_fold_{fold_idx}_model.joblib")

        if args.grid_search and best_inner_score is not None:
            print(
                f"  {model_name} fold {fold_idx}: inner_{args.scoring}={best_inner_score:.4f}, "
                f"accuracy={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}, "
                f"train={len(train_idx)}, test={len(test_idx)}"
            )
        else:
            print(
                f"  {model_name} fold {fold_idx}: accuracy={row['accuracy']:.4f}, "
                f"macro_f1={row['macro_f1']:.4f}, train={len(train_idx)}, test={len(test_idx)}"
            )

    return rows, curve_data


def main() -> None:
    args = parse_args()
    mol_info = read_molecule_source(
        args.mol_file,
        args.feat,
        fp_radius=args.fp_radius,
        fp_bits=args.fp_bits,
    )
    x, labels, pair_rows = read_training_data(
        mol_info,
        train_files=parse_csv_paths(args.train_files),
        label_column=args.label_column,
    )
    y, label_encoder = encode_labels(labels)
    param_grids = normalize_param_grids(load_json_arg(args.param_grids)) if args.param_grids else {}
    out_dir = ensure_dir(args.out_dir)
    splits = make_cold_start_splits(pair_rows, args.k_folds, args.random_state, args.cold_start_mode)

    split_rows = [
        {
            "fold": split["fold"],
            "train_size": int(len(split["train_idx"])),
            "test_size": int(len(split["test_idx"])),
            "n_cold_drugs": int(split["n_cold_drugs"]),
            "cold_drugs": ";".join(split["cold_drugs"]),
        }
        for split in splits
    ]
    pd.DataFrame(split_rows).to_csv(out_dir / "cold_start_splits.csv", index=False)

    rows = []
    curve_data = []
    for model_name in MODEL_NAMES:
        print(f"Running {MODEL_LABELS[model_name]} ({model_name}) cold-start evaluation")
        model_rows, model_curve_data = run_model_cold_start(
            args,
            x,
            y,
            pair_rows,
            label_encoder,
            model_name,
            splits,
            out_dir,
            param_grids,
        )
        rows.extend(model_rows)
        curve_data.extend(model_curve_data)

    if not rows:
        raise ValueError("no valid cold-start folds were evaluated; check cold_start_mode and k_folds")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    metrics_df.describe(include="all").to_csv(out_dir / "metrics_summary.csv")
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    summary_df = metrics_df.groupby(["model", "model_label"])[numeric_cols].agg(["mean", "std"])
    summary_df.to_csv(out_dir / "metrics_by_model_summary.csv")
    save_curve_outputs(
        curve_data,
        out_dir,
        group_columns=["model", "model_label", "feature_type", "cold_start_mode"],
        file_prefix="cold_start",
    )

    print("\nMean cold-start accuracy by model:")
    cold_start_summary = {}
    for model_name, group in metrics_df.groupby("model", sort=False):
        cold_start_summary[model_name] = {
            "accuracy_mean": float(group["accuracy"].mean()),
            "accuracy_std": float(group["accuracy"].std()),
            "macro_f1_mean": float(group["macro_f1"].mean()),
            "macro_f1_std": float(group["macro_f1"].std()),
            "mcc_mean": float(group["mcc"].mean()),
            "mcc_std": float(group["mcc"].std()),
        }
        print(
            f"  {model_name}: accuracy={cold_start_summary[model_name]['accuracy_mean']:.4f}, "
            f"macro_f1={cold_start_summary[model_name]['macro_f1_mean']:.4f}, "
            f"mcc={cold_start_summary[model_name]['mcc_mean']:.4f}"
        )

    (out_dir / "classes.json").write_text(
        json.dumps(label_encoder.classes_.tolist(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "cold_start_summary.json").write_text(
        json.dumps(cold_start_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved cold-start evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
