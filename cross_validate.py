"""Cross-validate drug-combination classifiers."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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
    ensure_dir,
    evaluate_predictions,
    load_feature_matrix_from_args,
    load_json_arg,
    normalize_param_grids,
    predict_proba_or_none,
    save_curve_outputs,
    serialize_params,
)


COMPARE_MODELS = MODEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_data_args(parser)
    parser.add_argument("--grid_search", action="store_true", help="run GridSearchCV inside each training fold")
    parser.add_argument(
        "--param_grids",
        default=None,
        help="JSON string or JSON file mapping model names to GridSearchCV grids",
    )
    parser.add_argument("--inner_cv", type=int, default=3, help="inner GridSearchCV folds")
    parser.add_argument("--scoring", default="f1_macro", help="GridSearchCV scoring name")
    parser.add_argument("--n_jobs", type=int, default=-1, help="GridSearchCV parallel jobs")
    parser.add_argument("--k_folds", type=int, default=10, help="number of stratified folds")
    parser.add_argument("--random_state", type=int, default=99, help="random seed")
    parser.add_argument("--exp", default="xgb_cv", help="experiment name")
    parser.add_argument("--out_dir", default="cross_validate", help="output directory")
    parser.add_argument("--save_fold_models", action="store_true", help="save each fold model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x, y, label_encoder, _ = load_feature_matrix_from_args(args)
    class_ids = np.arange(len(label_encoder.classes_))
    class_names = list(label_encoder.classes_)
    param_grids = normalize_param_grids(load_json_arg(args.param_grids)) if args.param_grids else {}

    out_dir = ensure_dir(Path(args.out_dir) / args.exp)
    splitter = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_state)
    rows = []
    curve_data = []

    for model_name in COMPARE_MODELS:
        print(f"Running {MODEL_LABELS[model_name]} ({model_name})")
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
            base_model = build_pipeline(model_name, random_state=args.random_state)
            best_params = {}
            best_inner_score = None

            if args.grid_search:
                fold_grid = param_grids.get(model_name) or default_param_grid(model_name)
                if not fold_grid:
                    raise ValueError(f"No GridSearchCV grid is configured for model={model_name!r}")
                search = GridSearchCV(
                    estimator=base_model,
                    param_grid=fold_grid,
                    scoring=args.scoring,
                    cv=StratifiedKFold(
                        n_splits=args.inner_cv,
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
                    out_dir / f"{model_name}_fold_{fold_idx}_grid_results.csv",
                    index=False,
                )
            else:
                model = base_model
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
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "feature_type": args.feat,
                },
            )
            row = {
                "fold": fold_idx,
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "feature_type": args.feat,
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "grid_search": bool(args.grid_search),
                "inner_cv": int(args.inner_cv) if args.grid_search else None,
                "grid_scoring": args.scoring if args.grid_search else None,
                "best_inner_score": best_inner_score,
                "best_params": serialize_params(best_params),
            }
            row.update(evaluate_predictions(y[test_idx], y_pred, y_proba, class_ids, class_names))
            rows.append(row)

            cm = pd.DataFrame(
                data=pd.crosstab(
                    pd.Series(label_encoder.inverse_transform(y[test_idx]), name="true"),
                    pd.Series(label_encoder.inverse_transform(y_pred), name="pred"),
                    dropna=False,
                )
            )
            cm.to_csv(out_dir / f"{model_name}_fold_{fold_idx}_confusion_matrix.csv")

            if args.save_fold_models:
                joblib.dump(model, out_dir / f"{model_name}_fold_{fold_idx}_model.joblib")

            if args.grid_search:
                print(
                    f"  {model_name} fold {fold_idx}: inner_{args.scoring}={best_inner_score:.4f}, "
                    f"accuracy={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}"
                )
            else:
                print(f"  {model_name} fold {fold_idx}: accuracy={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    metrics_df.describe(include="all").to_csv(out_dir / "metrics_summary.csv")
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    summary = metrics_df.groupby(["model", "model_label"])[numeric_cols].agg(["mean", "std"])
    summary.to_csv(out_dir / "metrics_by_model_summary.csv")
    save_curve_outputs(
        curve_data,
        out_dir,
        group_columns=["model", "model_label", "feature_type"],
        file_prefix="cross_validate",
    )
    print("\nMean accuracy by model:")
    for model_name, group in metrics_df.groupby("model", sort=False):
        print(f"  {model_name}: accuracy={group['accuracy'].mean():.4f}, macro_f1={group['macro_f1'].mean():.4f}")
    (out_dir / "classes.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved cross-validation outputs to {out_dir}")


if __name__ == "__main__":
    main()
