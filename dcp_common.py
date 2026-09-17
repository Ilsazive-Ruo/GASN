"""Shared utilities for drug-combination model scripts.

Run the entry-point scripts from the ``DrugCombination`` directory, for example:
    python scripts/cross_validate.py --model xgb --feat des
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_MOL_FILE = PROJECT_DIR / "structure_links.csv"
DEFAULT_TRAIN_FILES = (
    PROJECT_DIR / "data" / "antis_f.csv",
    PROJECT_DIR / "data" / "harms_f.csv",
    PROJECT_DIR / "data" / "others_f.csv",
    PROJECT_DIR / "data" / "syn_f.csv",
)


FEATURE_TYPES = ("des", "fp", "des_fp")
MODEL_NAMES = ("ada", "bnb", "knn", "dt", "xgb")
MODEL_LABELS = {
    "ada": "Adaptive Boosting",
    "bnb": "BernoulliNB",
    "knn": "K-Nearest Neighbors",
    "dt": "Decision Tree",
    "xgb": "Extreme Gradient Boosting",
}


def project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def ensure_dir(path):
    path = project_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_csv_paths(value):
    return [project_path(item.strip()) for item in value.split(",") if item.strip()]


def normalize_pair_key(id_1, id_2) -> Tuple[str, str]:
    pair = sorted((str(id_1).strip(), str(id_2).strip()))
    return pair[0], pair[1]


def load_json_arg(value):
    if not value:
        return {}
    maybe_path = Path(value)
    if maybe_path.exists():
        return json.loads(maybe_path.read_text(encoding="utf-8"))
    return json.loads(value)


def describe_mol(mol: Chem.Mol) -> List[float]:
    values: List[float] = []
    for _, calculator in Descriptors._descList:
        try:
            result = calculator(mol)
        except Exception:
            result = 0.0

        if not np.isfinite(result):
            values.append(0.0)
        elif result > np.finfo(np.float32).max:
            values.append(float(np.finfo(np.float32).max))
        else:
            values.append(float(np.float32(result)))
    return values


def mol_fp(mol: Chem.Mol, radius: int = 4, n_bits: int = 2048) -> List[float]:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return [float(bit) for bit in fp.ToBitString()]


def featurize_smiles(
    smiles: str,
    feature_type: str,
    fp_radius: int = 4,
    fp_bits: int = 2048,
) -> Optional[List[float]]:
    if pd.isna(smiles):
        return None
    smiles = str(smiles).strip()
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    if feature_type == "des":
        return describe_mol(mol)
    if feature_type == "fp":
        return mol_fp(mol, radius=fp_radius, n_bits=fp_bits)
    if feature_type == "des_fp":
        return mol_fp(mol, radius=fp_radius, n_bits=fp_bits) + describe_mol(mol)
    raise ValueError(f"SMILES featurization does not support feature_type={feature_type!r}")


def read_molecule_source(
    mol_file,
    feature_type: str,
    fp_radius: int = 4,
    fp_bits: int = 2048,
) -> Dict[str, List[float]]:
    if feature_type not in FEATURE_TYPES:
        raise ValueError(f"feature_type must be one of {FEATURE_TYPES}")

    source = pd.read_csv(project_path(mol_file))
    if "id" not in source.columns:
        raise ValueError("molecule source must contain an 'id' column")

    mol_info: Dict[str, List[float]] = {}
    skipped = 0
    for _, row in source.iterrows():
        mol_id = str(row["id"])
        if feature_type == "none":
            values = pd.to_numeric(row.iloc[6:], errors="coerce").fillna(0.0).astype(float).tolist()
        else:
            if "smiles" not in source.columns:
                raise ValueError("molecule source must contain a 'smiles' column for chemical features")
            values = featurize_smiles(row["smiles"], feature_type, fp_radius=fp_radius, fp_bits=fp_bits)
            if values is None:
                skipped += 1
                continue
        mol_info[mol_id] = values

    if not mol_info:
        raise ValueError(f"no molecules could be featurized from {mol_file}")
    if skipped:
        print(f"Skipped {skipped} molecules with invalid SMILES.")
    return mol_info


def read_training_data(
    mol_info: Dict[str, List[float]],
    train_files: Sequence = DEFAULT_TRAIN_FILES,
    label_column: str = "class",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    data = read_training_frames(train_files, label_column)
    data = resolve_duplicate_training_pairs(data, label_column)
    features: List[List[float]] = []
    labels: List[str] = []
    kept_rows = []

    for _, row in data.iterrows():
        id_1 = str(row["id-1"]).strip()
        id_2 = str(row["id-2"]).strip()
        if id_1 in mol_info and id_2 in mol_info:
            features.append(mol_info[id_1] + mol_info[id_2])
            labels.append(row[label_column])
            kept_rows.append(row)

    if not labels:
        raise ValueError("no training pairs matched the molecule source")
    return np.asarray(features, dtype=float), np.asarray(labels), pd.DataFrame(kept_rows)


def pair_features_from_frame(
    pairs: pd.DataFrame,
    feature_type: str,
    mol_info: Optional[Dict[str, List[float]]] = None,
    fp_radius: int = 4,
    fp_bits: int = 2048,
) -> Tuple[np.ndarray, pd.DataFrame]:
    features: List[List[float]] = []
    rows = []

    has_ids = {"id-1", "id-2"}.issubset(pairs.columns) and mol_info is not None
    has_smiles = {"smiles-1", "smiles-2"}.issubset(pairs.columns)
    if not has_ids and not has_smiles:
        raise ValueError("prediction file must contain id-1/id-2 with mol_info or smiles-1/smiles-2")

    for _, row in pairs.iterrows():
        values_1: Optional[List[float]]
        values_2: Optional[List[float]]
        if has_ids and str(row["id-1"]) in mol_info and str(row["id-2"]) in mol_info:
            values_1 = mol_info[str(row["id-1"])]
            values_2 = mol_info[str(row["id-2"])]
        elif has_smiles:
            values_1 = featurize_smiles(row["smiles-1"], feature_type, fp_radius=fp_radius, fp_bits=fp_bits)
            values_2 = featurize_smiles(row["smiles-2"], feature_type, fp_radius=fp_radius, fp_bits=fp_bits)
        else:
            continue
        if values_1 is None or values_2 is None:
            continue
        features.append(values_1 + values_2)
        rows.append(row)

    if not features:
        raise ValueError("no prediction pairs could be featurized")
    return np.asarray(features, dtype=float), pd.DataFrame(rows).reset_index(drop=True)


def encode_labels(labels: np.ndarray) -> Tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    return encoder.fit_transform(labels), encoder


def read_training_frames(train_files: Sequence, label_column: str) -> pd.DataFrame:
    frames = []
    for path in train_files:
        source_path = project_path(path)
        frame = pd.read_csv(source_path)
        frame["_source_file"] = source_path.name
        frame["_source_row"] = np.arange(1, len(frame) + 1)
        frames.append(frame)
    if not frames:
        raise ValueError("at least one training CSV file is required")
    data = pd.concat(frames, ignore_index=True)

    required = {"id-1", "id-2", label_column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"training data is missing columns: {sorted(missing)}")
    return data


def resolve_duplicate_training_pairs(data: pd.DataFrame, label_column: str) -> pd.DataFrame:
    data = data.copy()
    data["_id_1_normalized"] = data["id-1"].astype(str).str.strip()
    data["_id_2_normalized"] = data["id-2"].astype(str).str.strip()
    data["_label_normalized"] = data[label_column].astype(str).str.strip()
    data["_pair_key"] = [
        normalize_pair_key(id_1, id_2)
        for id_1, id_2 in zip(data["_id_1_normalized"], data["_id_2_normalized"])
    ]

    label_counts = data.groupby("_pair_key")["_label_normalized"].nunique(dropna=False)
    conflicting_keys = set(label_counts[label_counts > 1].index)
    conflict_rows = int(data["_pair_key"].isin(conflicting_keys).sum())

    if conflicting_keys:
        print(
            f"Removed {conflict_rows} rows from {len(conflicting_keys)} pairs with conflicting labels "
            f"(A-B and B-A are treated as the same pair)."
        )
        conflict_preview = (
            data[data["_pair_key"].isin(conflicting_keys)]
            .groupby("_pair_key")["_label_normalized"]
            .apply(lambda values: ",".join(sorted(set(values))))
            .head(10)
        )
        for pair_key, labels in conflict_preview.items():
            print(f"  conflict pair {pair_key[0]} | {pair_key[1]} labels={labels}")

    non_conflicting = data[~data["_pair_key"].isin(conflicting_keys)].copy()
    before_dedup = len(non_conflicting)
    non_conflicting = non_conflicting.drop_duplicates(subset=["_pair_key"], keep="first").copy()
    duplicate_rows = before_dedup - len(non_conflicting)
    if duplicate_rows:
        print(
            f"Removed {duplicate_rows} duplicate pair rows with consistent labels "
            f"(A-B and B-A are treated as the same pair)."
        )

    non_conflicting[label_column] = non_conflicting["_label_normalized"]
    helper_columns = ["_id_1_normalized", "_id_2_normalized", "_label_normalized", "_pair_key"]
    return non_conflicting.drop(columns=helper_columns).reset_index(drop=True)


def build_xgb(random_state: int = 99, n_jobs: int = -1, **params) -> XGBClassifier:
    return XGBClassifier(**params)


def build_estimator(model_name: str, random_state: int = 99, params: Optional[dict] = None):
    params = params or {}
    if model_name == "ada":
        return AdaBoostClassifier(**params)
    if model_name == "bnb":
        return BernoulliNB(**params)
    if model_name == "xgb":
        return build_xgb(random_state=random_state, **params)
    if model_name == "dt":
        return DecisionTreeClassifier(**params)
    if model_name == "knn":
        return KNeighborsClassifier(**params)
    raise ValueError(f"model_name must be one of {MODEL_NAMES}")


def build_pipeline(model_name: str, random_state: int = 99, params: Optional[dict] = None) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", build_estimator(model_name, random_state=random_state, params=params)),
        ]
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    class_ids: Sequence[int],
    class_names: Sequence[str],
) -> dict:
    result = {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "mcc": metrics.matthews_corrcoef(y_true, y_pred),
        "macro_f1": metrics.f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    f1 = metrics.f1_score(y_true, y_pred, average=None, labels=class_ids, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, average=None, labels=class_ids, zero_division=0)
    precision = metrics.precision_score(y_true, y_pred, average=None, labels=class_ids, zero_division=0)
    for idx, name in enumerate(class_names):
        result[f"f1_{name}"] = f1[idx]
        result[f"recall_{name}"] = recall[idx]
        result[f"precision_{name}"] = precision[idx]

    if y_proba is not None:
        y_bin = label_binarize(y_true, classes=list(class_ids))
        for idx, name in enumerate(class_names):
            try:
                result[f"ap_{name}"] = metrics.average_precision_score(y_bin[:, idx], y_proba[:, idx])
            except ValueError:
                result[f"ap_{name}"] = np.nan
            try:
                result[f"roc_auc_{name}"] = metrics.roc_auc_score(y_bin[:, idx], y_proba[:, idx])
            except ValueError:
                result[f"roc_auc_{name}"] = np.nan
    return result


def predict_proba_or_none(model: Pipeline, x: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)
    return None


def save_bundle(
    out_dir,
    model: Pipeline,
    label_encoder: LabelEncoder,
    metadata: dict,
) -> Path:
    out_dir = ensure_dir(out_dir)
    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(label_encoder, out_dir / "label_encoder.joblib")
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_dir


def load_bundle(model_dir):
    model_dir = project_path(model_dir)
    model = joblib.load(model_dir / "model.joblib")
    label_encoder = joblib.load(model_dir / "label_encoder.joblib")
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return model, label_encoder, metadata


def save_metrics_csv(rows, path):
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)
    return path


def collect_curve_data(
    curve_data: List[dict],
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray],
    class_ids: Sequence[int],
    class_names: Sequence[str],
    fold: int,
    group_values: dict,
) -> None:
    if y_proba is None:
        return
    curve_data.append(
        {
            **group_values,
            "fold": int(fold),
            "y_true": np.asarray(y_true),
            "y_proba": np.asarray(y_proba),
            "class_ids": list(class_ids),
            "class_names": list(class_names),
        }
    )


def _sanitize_filename(value) -> str:
    allowed = []
    for char in str(value):
        if char.isalnum() or char in ("-", "_"):
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "value"


def _group_stem(entry: dict, group_columns: Sequence[str]) -> str:
    return "__".join(f"{col}-{_sanitize_filename(entry.get(col, 'NA'))}" for col in group_columns)


def _has_binary_classes(y_binary: np.ndarray) -> bool:
    return np.unique(y_binary).size == 2


def _interp_pr_curve(recall: np.ndarray, precision: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(recall)
    recall_sorted = recall[order]
    precision_sorted = precision[order]
    unique_recall = np.unique(recall_sorted)
    unique_precision = np.array(
        [precision_sorted[recall_sorted == value].max() for value in unique_recall],
        dtype=float,
    )
    return np.interp(grid, unique_recall, unique_precision)


def _calibration_bin_rows(
    y_binary: np.ndarray,
    y_score: np.ndarray,
    n_bins: int,
) -> List[dict]:
    rows = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for bin_idx in range(n_bins):
        left = edges[bin_idx]
        right = edges[bin_idx + 1]
        if bin_idx == n_bins - 1:
            mask = (y_score >= left) & (y_score <= right)
        else:
            mask = (y_score >= left) & (y_score < right)
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin_index": bin_idx,
                "bin_start": float(left),
                "bin_end": float(right),
                "bin_center": float((left + right) / 2.0),
                "count": count,
                "mean_predicted": float(y_score[mask].mean()),
                "fraction_positive": float(y_binary[mask].mean()),
            }
        )
    return rows


def _plot_curve_series(
    series: Sequence[dict],
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    diagonal: bool = False,
) -> None:
    if not series:
        return
    plt.figure(figsize=(6, 5))
    for item in series:
        x = np.asarray(item["x"], dtype=float)
        y = np.asarray(item["y"], dtype=float)
        plt.plot(x, y, label=item["label"], linewidth=1.8)
        y_std = item.get("y_std")
        if y_std is not None:
            y_std = np.asarray(y_std, dtype=float)
            plt.fill_between(x, np.maximum(y - y_std, 0.0), np.minimum(y + y_std, 1.0), alpha=0.18)
    if diagonal:
        plt.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1.0, label="reference")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _std_or_zero(values: np.ndarray) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros(values.shape[1], dtype=float)
    return values.std(axis=0, ddof=1)


def _summarize_interpolated_curves(
    records: Sequence[dict],
    group_columns: Sequence[str],
    x_name: str,
    y_name: str,
    score_name: str,
) -> List[dict]:
    key_columns = list(group_columns) + ["class_id", "class_name"]
    grouped: Dict[Tuple, List[dict]] = {}
    for record in records:
        key = tuple(record[col] for col in key_columns)
        grouped.setdefault(key, []).append(record)

    rows = []
    for key, items in grouped.items():
        base = dict(zip(key_columns, key))
        x_grid = np.asarray(items[0]["x"], dtype=float)
        y_values = np.vstack([np.asarray(item["y"], dtype=float) for item in items])
        scores = np.asarray([item[score_name] for item in items], dtype=float)
        y_mean = y_values.mean(axis=0)
        y_std = _std_or_zero(y_values)
        score_mean = float(np.nanmean(scores))
        score_std = float(np.nanstd(scores, ddof=1)) if len(scores) > 1 else 0.0
        for point_idx, x_value in enumerate(x_grid):
            rows.append(
                {
                    **base,
                    "point": point_idx,
                    x_name: float(x_value),
                    f"{y_name}_mean": float(y_mean[point_idx]),
                    f"{y_name}_std": float(y_std[point_idx]),
                    "n_folds": int(len(items)),
                    f"{score_name}_mean": score_mean,
                    f"{score_name}_std": score_std,
                }
            )
    return rows


def _plot_summary_curves(
    summary_df: pd.DataFrame,
    group_columns: Sequence[str],
    output_dir: Path,
    curve_name: str,
    x_col: str,
    y_mean_col: str,
    y_std_col: str,
    xlabel: str,
    ylabel: str,
    diagonal: bool = False,
) -> None:
    if summary_df.empty:
        return
    for group_key, group_df in summary_df.groupby(list(group_columns), dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_values = dict(zip(group_columns, group_key))
        stem = "__".join(f"{col}-{_sanitize_filename(value)}" for col, value in group_values.items())
        series = []
        for class_name, class_df in group_df.groupby("class_name", sort=False):
            class_df = class_df.sort_values(x_col)
            series.append(
                {
                    "label": str(class_name),
                    "x": class_df[x_col].to_numpy(),
                    "y": class_df[y_mean_col].to_numpy(),
                    "y_std": class_df[y_std_col].to_numpy(),
                }
            )
        _plot_curve_series(
            series,
            output_dir / f"{stem}_{curve_name}_mean_std.png",
            f"{curve_name.upper()} mean +/- std",
            xlabel,
            ylabel,
            diagonal=diagonal,
        )


def save_curve_outputs(
    curve_data: Sequence[dict],
    out_dir,
    group_columns: Sequence[str],
    file_prefix: str = "",
    interpolation_points: int = 101,
    calibration_bins: int = 10,
) -> dict:
    if not curve_data:
        return {}

    out_dir = project_path(out_dir)
    curves_dir = out_dir / "curves"
    folds_dir = curves_dir / "folds"
    curves_dir.mkdir(parents=True, exist_ok=True)
    folds_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{file_prefix}_" if file_prefix else ""
    roc_grid = np.linspace(0.0, 1.0, interpolation_points)
    pr_grid = np.linspace(0.0, 1.0, interpolation_points)
    roc_fold_rows = []
    pr_fold_rows = []
    calibration_fold_rows = []
    roc_interp_records = []
    pr_interp_records = []

    for entry in curve_data:
        y_true = np.asarray(entry["y_true"])
        y_proba = np.asarray(entry["y_proba"])
        class_ids = entry["class_ids"]
        class_names = entry["class_names"]
        fold = int(entry["fold"])
        base = {col: entry.get(col) for col in group_columns}
        stem = _group_stem(entry, group_columns)
        n_classes = min(y_proba.shape[1], len(class_ids))
        fold_roc_series = []
        fold_pr_series = []
        fold_calibration_series = []

        for class_idx in range(n_classes):
            class_id = class_ids[class_idx]
            class_name = class_names[class_idx]
            y_binary = (y_true == class_id).astype(int)
            y_score = y_proba[:, class_idx]
            class_base = {
                **base,
                "fold": fold,
                "class_id": int(class_id),
                "class_name": class_name,
            }

            for cal_row in _calibration_bin_rows(y_binary, y_score, calibration_bins):
                calibration_fold_rows.append({**class_base, **cal_row})
            class_calibration_rows = [
                row
                for row in calibration_fold_rows
                if row["fold"] == fold
                and row["class_id"] == int(class_id)
                and all(row.get(col) == base.get(col) for col in group_columns)
            ]
            if class_calibration_rows:
                fold_calibration_series.append(
                    {
                        "label": class_name,
                        "x": [row["mean_predicted"] for row in class_calibration_rows],
                        "y": [row["fraction_positive"] for row in class_calibration_rows],
                    }
                )

            if not _has_binary_classes(y_binary):
                continue

            fpr, tpr, roc_thresholds = metrics.roc_curve(y_binary, y_score)
            roc_auc = float(metrics.auc(fpr, tpr))
            for point_idx, (x_value, y_value, threshold) in enumerate(zip(fpr, tpr, roc_thresholds)):
                roc_fold_rows.append(
                    {
                        **class_base,
                        "point": point_idx,
                        "fpr": float(x_value),
                        "tpr": float(y_value),
                        "threshold": float(threshold),
                        "auc": roc_auc,
                    }
                )
            roc_interp = np.interp(roc_grid, fpr, tpr)
            roc_interp[0] = 0.0
            roc_interp[-1] = 1.0
            roc_interp_records.append(
                {
                    **base,
                    "fold": fold,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "x": roc_grid,
                    "y": roc_interp,
                    "auc": roc_auc,
                }
            )
            fold_roc_series.append({"label": f"{class_name} AUC={roc_auc:.3f}", "x": fpr, "y": tpr})

            precision, recall, pr_thresholds = metrics.precision_recall_curve(y_binary, y_score)
            ap = float(metrics.average_precision_score(y_binary, y_score))
            for point_idx, (x_value, y_value) in enumerate(zip(recall, precision)):
                threshold = pr_thresholds[point_idx] if point_idx < len(pr_thresholds) else np.nan
                pr_fold_rows.append(
                    {
                        **class_base,
                        "point": point_idx,
                        "recall": float(x_value),
                        "precision": float(y_value),
                        "threshold": float(threshold),
                        "ap": ap,
                    }
                )
            pr_interp = _interp_pr_curve(recall, precision, pr_grid)
            pr_interp_records.append(
                {
                    **base,
                    "fold": fold,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "x": pr_grid,
                    "y": pr_interp,
                    "ap": ap,
                }
            )
            fold_pr_series.append({"label": f"{class_name} AP={ap:.3f}", "x": recall, "y": precision})

        _plot_curve_series(
            fold_roc_series,
            folds_dir / f"{stem}_fold_{fold}_roc.png",
            f"{stem} fold {fold} ROC",
            "False positive rate",
            "True positive rate",
            diagonal=True,
        )
        _plot_curve_series(
            fold_pr_series,
            folds_dir / f"{stem}_fold_{fold}_pr.png",
            f"{stem} fold {fold} PR",
            "Recall",
            "Precision",
        )
        _plot_curve_series(
            fold_calibration_series,
            folds_dir / f"{stem}_fold_{fold}_calibration.png",
            f"{stem} fold {fold} calibration",
            "Mean predicted probability",
            "Fraction positive",
            diagonal=True,
        )

    output_paths = {}
    if roc_fold_rows:
        roc_fold_path = curves_dir / f"{prefix}fold_roc_curves.csv"
        pd.DataFrame(roc_fold_rows).to_csv(roc_fold_path, index=False)
        output_paths["fold_roc"] = str(roc_fold_path)
    if pr_fold_rows:
        pr_fold_path = curves_dir / f"{prefix}fold_pr_curves.csv"
        pd.DataFrame(pr_fold_rows).to_csv(pr_fold_path, index=False)
        output_paths["fold_pr"] = str(pr_fold_path)
    if calibration_fold_rows:
        calibration_fold_path = curves_dir / f"{prefix}fold_calibration_curves.csv"
        pd.DataFrame(calibration_fold_rows).to_csv(calibration_fold_path, index=False)
        output_paths["fold_calibration"] = str(calibration_fold_path)

        calibration_summary_df = (
            pd.DataFrame(calibration_fold_rows)
            .groupby(list(group_columns) + ["class_id", "class_name", "bin_index", "bin_start", "bin_end", "bin_center"])
            .agg(
                mean_predicted_mean=("mean_predicted", "mean"),
                mean_predicted_std=("mean_predicted", "std"),
                fraction_positive_mean=("fraction_positive", "mean"),
                fraction_positive_std=("fraction_positive", "std"),
                n_folds=("fold", "nunique"),
                total_count=("count", "sum"),
            )
            .reset_index()
        )
        calibration_summary_df[["mean_predicted_std", "fraction_positive_std"]] = calibration_summary_df[
            ["mean_predicted_std", "fraction_positive_std"]
        ].fillna(0.0)
        calibration_summary_path = curves_dir / f"{prefix}calibration_curve_mean_std.csv"
        calibration_summary_df.to_csv(calibration_summary_path, index=False)
        output_paths["calibration_summary"] = str(calibration_summary_path)
        _plot_summary_curves(
            calibration_summary_df,
            group_columns,
            curves_dir,
            "calibration",
            "mean_predicted_mean",
            "fraction_positive_mean",
            "fraction_positive_std",
            "Mean predicted probability",
            "Fraction positive",
            diagonal=True,
        )

    roc_summary_rows = _summarize_interpolated_curves(
        roc_interp_records,
        group_columns,
        "fpr",
        "tpr",
        "auc",
    )
    if roc_summary_rows:
        roc_summary_df = pd.DataFrame(roc_summary_rows)
        roc_summary_path = curves_dir / f"{prefix}roc_curve_mean_std.csv"
        roc_summary_df.to_csv(roc_summary_path, index=False)
        output_paths["roc_summary"] = str(roc_summary_path)
        _plot_summary_curves(
            roc_summary_df,
            group_columns,
            curves_dir,
            "roc",
            "fpr",
            "tpr_mean",
            "tpr_std",
            "False positive rate",
            "True positive rate",
            diagonal=True,
        )

    pr_summary_rows = _summarize_interpolated_curves(
        pr_interp_records,
        group_columns,
        "recall",
        "precision",
        "ap",
    )
    if pr_summary_rows:
        pr_summary_df = pd.DataFrame(pr_summary_rows)
        pr_summary_path = curves_dir / f"{prefix}pr_curve_mean_std.csv"
        pr_summary_df.to_csv(pr_summary_path, index=False)
        output_paths["pr_summary"] = str(pr_summary_path)
        _plot_summary_curves(
            pr_summary_df,
            group_columns,
            curves_dir,
            "pr",
            "recall",
            "precision_mean",
            "precision_std",
            "Recall",
            "Precision",
        )

    return output_paths


def default_xgb_grid() -> dict:
    return {
        "model__n_estimators": [50, 100, 200, 400],
        "model__max_depth": [None, 3, 5, 10, 20, 40],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    }


def default_param_grid(model_name):
    if model_name == "xgb":
        return default_xgb_grid()
    if model_name == "ada":
        return {
            "model__estimator": [
                DecisionTreeClassifier(max_depth=1),
                DecisionTreeClassifier(max_depth=2),
                DecisionTreeClassifier(max_depth=3),
            ],
            "model__n_estimators": [50, 100, 200, 400],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        }
    if model_name == "bnb":
        return {
            "model__alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
            "model__binarize": [0.0, 0.25, 0.5, 0.75, 1.0],
            "model__fit_prior": [True, False],
        }
    if model_name == "dt":
        return {
            "model__criterion": ["gini", "entropy", "log_loss"],
            "model__max_depth": [None, 3, 5, 10, 20, 40],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__max_features": [None, "sqrt", "log2"],
        }
    if model_name == "knn":
        return {
            "model__n_neighbors": [1, 3, 5, 7, 9, 15, 21],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
            "model__leaf_size": [20, 30, 50],
        }
    return {}


def normalize_pipeline_grid(param_grid):
    normalized = {}
    for key, value in param_grid.items():
        if "__" in key:
            normalized[key] = value
        else:
            normalized[f"model__{key}"] = value
    return normalized


def normalize_param_grids(raw_grids):
    normalized = {}
    for model_name, grid in raw_grids.items():
        normalized[model_name.lower()] = normalize_pipeline_grid(grid)
    return normalized


def serialize_params(params):
    return json.dumps(params, ensure_ascii=False, default=str)


def run_grid_search(
    x: np.ndarray,
    y: np.ndarray,
    param_grid: Optional[dict],
    cv: int,
    scoring: str,
    random_state: int,
    n_jobs: int,
) -> GridSearchCV:
    pipeline = build_pipeline("xgb", random_state=random_state)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid or default_xgb_grid(),
        scoring=scoring,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
    )
    search.fit(x, y)
    return search


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mol_file", default=str(DEFAULT_MOL_FILE), help="molecule source CSV")
    parser.add_argument("--feat", choices=FEATURE_TYPES, default="des", help="molecule feature type")
    parser.add_argument(
        "--train_files",
        default=",".join(str(path) for path in DEFAULT_TRAIN_FILES),
        help="comma-separated training CSV files",
    )
    parser.add_argument("--label_column", default="class", help="label column in training files")
    parser.add_argument("--fp_radius", type=int, default=4, help="Morgan fingerprint radius")
    parser.add_argument("--fp_bits", type=int, default=2048, help="Morgan fingerprint bit length")


def load_feature_matrix_from_args(args) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, Dict[str, List[float]]]:
    mol_info = read_molecule_source(
        args.mol_file,
        args.feat,
        fp_radius=args.fp_radius,
        fp_bits=args.fp_bits,
    )
    x, labels, _ = read_training_data(
        mol_info,
        train_files=parse_csv_paths(args.train_files),
        label_column=args.label_column,
    )
    y, label_encoder = encode_labels(labels)
    return x, y, label_encoder, mol_info
