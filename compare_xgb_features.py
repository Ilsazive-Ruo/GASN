"""Compare XGBoost performance across molecule descriptors and fingerprints."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dcp_common import (
    DEFAULT_MOL_FILE,
    DEFAULT_TRAIN_FILES,
    FEATURE_TYPES,
    add_common_data_args,
    build_pipeline,
    collect_curve_data,
    default_xgb_grid,
    encode_labels,
    ensure_dir,
    evaluate_predictions,
    load_json_arg,
    parse_csv_paths,
    predict_proba_or_none,
    read_molecule_source,
    read_training_data,
    save_curve_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mol_file", default=str(DEFAULT_MOL_FILE), help="molecule source CSV")
    parser.add_argument(
        "--train_files",
        default=",".join(str(path) for path in DEFAULT_TRAIN_FILES),
        help="comma-separated training CSV files",
    )
    parser.add_argument("--label_column", default="class", help="label column in training files")
    parser.add_argument(
        "--features",
        default=",".join(FEATURE_TYPES),
        help="comma-separated feature types to compare; use 'all' for des,fp,des_fp",
    )
    parser.add_argument("--fp_radius", type=int, default=4, help="Morgan fingerprint radius")
    parser.add_argument("--fp_bits", type=int, default=2048, help="Morgan fingerprint bit length")
    parser.add_argument("--k_folds", type=int, default=10, help="outer cross-validation folds")
    parser.add_argument("--random_state", type=int, default=99, help="random seed")
    parser.add_argument("--grid_search", action="store_true", help="run GridSearchCV for each feature type")
    parser.add_argument(
        "--param_grid",
        default=None,
        help="JSON string or JSON file for XGB GridSearchCV; keys may be estimator names or model__ names",
    )
    parser.add_argument("--grid_cv", type=int, default=3, help="inner GridSearchCV folds")
    parser.add_argument("--scoring", default="f1_macro", help="sklearn scoring name")
    parser.add_argument("--n_jobs", type=int, default=-1, help="GridSearchCV parallel jobs")
    parser.add_argument("--out_dir", default="xgb_feature_compare", help="output directory")
    return parser.parse_args()


def parse_feature_types(value):
    if value.strip().lower() == "all":
        return list(FEATURE_TYPES)
    feature_types = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(feature_types) - set(FEATURE_TYPES))
    if invalid:
        raise ValueError(f"unsupported feature types: {invalid}; expected one or more of {FEATURE_TYPES}")
    if not feature_types:
        raise ValueError("at least one feature type is required")
    return feature_types


def normalize_pipeline_grid(param_grid):
    normalized = {}
    for key, value in param_grid.items():
        if "__" in key:
            normalized[key] = value
        else:
            normalized[f"model__{key}"] = value
    return normalized


def cross_validate_xgb(x, y, label_encoder, feature_type, args, out_dir, param_grid=None):
    class_ids = np.arange(len(label_encoder.classes_))
    class_names = list(label_encoder.classes_)
    splitter = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_state)
    rows = []
    curve_data = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
        model = build_pipeline("xgb", random_state=args.random_state)
        best_params = {}
        best_inner_score = None
        if args.grid_search:
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid or default_xgb_grid(),
                scoring=args.scoring,
                cv=StratifiedKFold(
                    n_splits=args.grid_cv,
                    shuffle=True,
                    random_state=args.random_state,
                ),
                n_jobs=args.n_jobs,
                refit=True,
                verbose=1,
            )
            search.fit(x[train_idx], y[train_idx])
            model = search.best_estimator_
            best_params = search.best_params_
            best_inner_score = float(search.best_score_)
            pd.DataFrame(search.cv_results_).to_csv(
                out_dir / f"{feature_type}_fold_{fold_idx}_grid_results.csv",
                index=False,
            )
        else:
            model.fit(x[train_idx], y[train_idx])

        y_pred = model.predict(x[test_idx])
        y_proba = predict_proba_or_none(model, x[test_idx])
        collect_curve_data(
            curve_data,
            y[test_idx],
            y_proba,
            class_ids,
            class_names,
            fold_idx,
            {
                "feature_type": feature_type,
                "model": "xgb",
            },
        )
        row = {
            "feature_type": feature_type,
            "fold": fold_idx,
            "n_features": int(x.shape[1]),
            "grid_search": bool(args.grid_search),
            "best_inner_score": best_inner_score,
            "best_params": json.dumps(best_params, ensure_ascii=False),
        }
        row.update(evaluate_predictions(y[test_idx], y_pred, y_proba, class_ids, class_names))
        rows.append(row)
    return rows, curve_data


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    train_files = parse_csv_paths(args.train_files)
    all_rows = []
    all_curve_data = []
    param_grid = normalize_pipeline_grid(load_json_arg(args.param_grid)) if args.param_grid else None

    feature_types = parse_feature_types(args.features)
    print(f"Running feature comparison for: {', '.join(feature_types)}")

    for feature_type in feature_types:
        mol_info = read_molecule_source(
            args.mol_file,
            feature_type,
            fp_radius=args.fp_radius,
            fp_bits=args.fp_bits,
        )
        x, labels, _ = read_training_data(mol_info, train_files=train_files, label_column=args.label_column)
        y, label_encoder = encode_labels(labels)

        rows, curve_data = cross_validate_xgb(x, y, label_encoder, feature_type, args, out_dir, param_grid=param_grid)
        all_rows.extend(rows)
        all_curve_data.extend(curve_data)
        print(
            f"{feature_type}: mean accuracy={np.mean([r['accuracy'] for r in rows]):.4f}, "
            f"mean macro_f1={np.mean([r['macro_f1'] for r in rows]):.4f}"
        )

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(out_dir / "feature_compare_metrics.csv", index=False)
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    summary = metrics_df.groupby("feature_type")[numeric_cols].agg(["mean", "std"])
    summary.to_csv(out_dir / "feature_compare_summary.csv")
    save_curve_outputs(
        all_curve_data,
        out_dir,
        group_columns=["feature_type", "model"],
        file_prefix="feature_compare",
    )
    print(f"Saved XGB feature comparison outputs to {out_dir}")


if __name__ == "__main__":
    main()
