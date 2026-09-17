"""Run GridSearchCV hyperparameter optimization for XGBoost."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dcp_common import (
    add_common_data_args,
    default_xgb_grid,
    ensure_dir,
    load_feature_matrix_from_args,
    load_json_arg,
    run_grid_search,
    save_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_data_args(parser)
    parser.add_argument("--param_grid", default=None, help="JSON string or JSON file; defaults to a compact XGB grid")
    parser.add_argument("--cv", type=int, default=10, help="GridSearchCV folds")
    parser.add_argument("--scoring", default="f1_macro", help="sklearn scoring name")
    parser.add_argument("--random_state", type=int, default=99, help="random seed")
    parser.add_argument("--n_jobs", type=int, default=-1, help="GridSearchCV parallel jobs")
    parser.add_argument("--out_dir", default="scripts_outputs/grid_search_xgb/des", help="output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x, y, label_encoder, _ = load_feature_matrix_from_args(args)
    param_grid = load_json_arg(args.param_grid) if args.param_grid else default_xgb_grid()
    search = run_grid_search(
        x=x,
        y=y,
        param_grid=param_grid,
        cv=args.cv,
        scoring=args.scoring,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    out_dir = ensure_dir(args.out_dir)
    pd.DataFrame(search.cv_results_).to_csv(out_dir / "cv_results.csv", index=False)
    joblib.dump(search, out_dir / "grid_search.joblib")
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "grid_search_xgb",
        "model": "xgb",
        "feature_type": args.feat,
        "fp_radius": args.fp_radius,
        "fp_bits": args.fp_bits,
        "scoring": args.scoring,
        "cv": args.cv,
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "classes": label_encoder.classes_.tolist(),
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
    }
    (out_dir / "best_params.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_bundle(out_dir / "best_model", search.best_estimator_, label_encoder, metadata)
    print(f"Best {args.scoring}: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    print(f"Saved GridSearchCV outputs to {out_dir}")


if __name__ == "__main__":
    main()
