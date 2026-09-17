"""Compute LHT2-vs-training molecule similarity and distance matrices."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_LHT2_FILE = PROJECT_DIR / "data" / "ToBeScreened.csv"
DEFAULT_REFERENCE_FILES = (
    PROJECT_DIR / "data" / "antis_f.csv",
    PROJECT_DIR / "data" / "harms_f.csv",
    PROJECT_DIR / "data" / "others_f.csv",
    PROJECT_DIR / "data" / "syn_f.csv",
)


def project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def parse_csv_paths(value):
    return [project_path(item.strip()) for item in value.split(",") if item.strip()]


def molecule_key(name: str, smiles: str) -> str:
    name = "" if pd.isna(name) else str(name).strip()
    smiles = "" if pd.isna(smiles) else str(smiles).strip()
    if name:
        return name
    return smiles


def canonical_smiles(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None, None
    return Chem.MolToSmiles(mol, canonical=True), mol


def add_molecule(
    molecules: Dict[str, dict],
    name,
    smiles,
    source: str,
    molecule_id=None,
    role=None,
) -> None:
    smiles = "" if pd.isna(smiles) else str(smiles).strip()
    if not smiles:
        return

    canonical, mol = canonical_smiles(smiles)
    if canonical is None:
        return

    key = molecule_key(name, smiles)
    molecule = molecules.setdefault(
        canonical,
        {
            "matrix_label": key,
            "id": "" if molecule_id is None or pd.isna(molecule_id) else str(molecule_id).strip(),
            "name": "" if pd.isna(name) else str(name).strip(),
            "smiles": smiles,
            "canonical_smiles": canonical,
            "sources": set(),
            "roles": set(),
            "mol": mol,
        },
    )
    molecule["sources"].add(source)
    if role:
        molecule["roles"].add(role)
    if not molecule["id"] and molecule_id is not None and not pd.isna(molecule_id):
        molecule["id"] = str(molecule_id).strip()
    if not molecule["name"] and not pd.isna(name):
        molecule["name"] = str(name).strip()


def read_lht2_molecules(path) -> List[dict]:
    frame = pd.read_csv(project_path(path))
    molecules: Dict[str, dict] = {}
    for _, row in frame.iterrows():
        add_molecule(molecules, row.get("name-1"), row.get("smiles-1"), Path(path).name, role="left")
        add_molecule(molecules, row.get("name-2"), row.get("smiles-2"), Path(path).name, role="right")
    return sorted(molecules.values(), key=lambda item: item["matrix_label"].lower())


def read_reference_molecules(paths: Iterable) -> List[dict]:
    molecules: Dict[str, dict] = {}
    for path in paths:
        path = project_path(path)
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            add_molecule(
                molecules,
                row.get("name-1"),
                row.get("smiles-1"),
                path.name,
                molecule_id=row.get("id-1"),
                role="id-1",
            )
            add_molecule(
                molecules,
                row.get("name-2"),
                row.get("smiles-2"),
                path.name,
                molecule_id=row.get("id-2"),
                role="id-2",
            )
    return sorted(molecules.values(), key=lambda item: (item["id"], item["matrix_label"].lower()))


def read_reference_occurrences(paths: Iterable) -> Dict[str, dict]:
    occurrences: Dict[str, dict] = {}
    for path in paths:
        path = project_path(path)
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            for side in ("1", "2"):
                smiles = row.get(f"smiles-{side}")
                canonical, _ = canonical_smiles(smiles)
                if canonical is None:
                    continue
                occurrence = occurrences.setdefault(
                    canonical,
                    {
                        "ids": set(),
                        "names": set(),
                        "sources": set(),
                        "roles": set(),
                        "counts_by_file": {},
                        "total_occurrences": 0,
                    },
                )
                molecule_id = row.get(f"id-{side}")
                name = row.get(f"name-{side}")
                if molecule_id is not None and not pd.isna(molecule_id):
                    occurrence["ids"].add(str(molecule_id).strip())
                if name is not None and not pd.isna(name):
                    occurrence["names"].add(str(name).strip())
                occurrence["sources"].add(path.name)
                occurrence["roles"].add(f"id-{side}")
                occurrence["counts_by_file"][path.name] = occurrence["counts_by_file"].get(path.name, 0) + 1
                occurrence["total_occurrences"] += 1
    return occurrences


def morgan_fp(mol, radius: int, n_bits: int):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def descriptor_vector(mol) -> List[float]:
    values = []
    for _, calculator in Descriptors._descList:
        try:
            value = calculator(mol)
        except Exception:
            value = 0.0
        if not np.isfinite(value):
            value = 0.0
        elif value > np.finfo(np.float32).max:
            value = float(np.finfo(np.float32).max)
        values.append(float(np.float32(value)))
    return values


def make_unique_labels(molecules: List[dict], fallback_prefix: str) -> List[str]:
    seen = {}
    labels = []
    for idx, molecule in enumerate(molecules, start=1):
        base = molecule["matrix_label"] or molecule["id"] or f"{fallback_prefix}_{idx}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        labels.append(base if count == 1 else f"{base}_{count}")
    return labels


def save_molecule_table(molecules: List[dict], labels: List[str], path) -> None:
    rows = []
    for label, molecule in zip(labels, molecules):
        rows.append(
            {
                "matrix_label": label,
                "id": molecule["id"],
                "name": molecule["name"],
                "smiles": molecule["smiles"],
                "canonical_smiles": molecule["canonical_smiles"],
                "sources": ";".join(sorted(molecule["sources"])),
                "roles": ";".join(sorted(molecule["roles"])),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def save_occurrence_table(
    row_molecules: List[dict],
    row_labels: List[str],
    reference_occurrences: Dict[str, dict],
    reference_files: Iterable,
    path,
) -> pd.DataFrame:
    source_names = [Path(item).name for item in reference_files]
    rows = []
    for label, molecule in zip(row_labels, row_molecules):
        occurrence = reference_occurrences.get(molecule["canonical_smiles"])
        row = {
            "matrix_label": label,
            "name": molecule["name"],
            "smiles": molecule["smiles"],
            "canonical_smiles": molecule["canonical_smiles"],
            "appears_in_reference": occurrence is not None,
            "total_occurrences": occurrence["total_occurrences"] if occurrence else 0,
            "reference_ids": ";".join(sorted(occurrence["ids"])) if occurrence else "",
            "reference_names": ";".join(sorted(occurrence["names"])) if occurrence else "",
            "reference_sources": ";".join(sorted(occurrence["sources"])) if occurrence else "",
            "reference_roles": ";".join(sorted(occurrence["roles"])) if occurrence else "",
        }
        for source in source_names:
            row[f"occurrences_in_{source}"] = occurrence["counts_by_file"].get(source, 0) if occurrence else 0
        rows.append(row)
    occurrence_df = pd.DataFrame(rows)
    occurrence_df.to_csv(path, index=False)
    return occurrence_df


def save_tanimoto_topk_summary(
    tanimoto: np.ndarray,
    row_molecules: List[dict],
    col_molecules: List[dict],
    row_labels: List[str],
    col_labels: List[str],
    path,
) -> pd.DataFrame:
    rows = []
    for row_idx, (label, molecule) in enumerate(zip(row_labels, row_molecules)):
        scores = np.asarray(tanimoto[row_idx], dtype=float)
        ranked_idx = np.argsort(-scores, kind="mergesort")
        top1_idx = int(ranked_idx[0])
        top5_idx = ranked_idx[: min(5, len(ranked_idx))]
        top10_idx = ranked_idx[: min(10, len(ranked_idx))]
        top1_molecule = col_molecules[top1_idx]

        rows.append(
            {
                "matrix_label": label,
                "name": molecule["name"],
                "smiles": molecule["smiles"],
                "canonical_smiles": molecule["canonical_smiles"],
                "max_similarity": float(scores[top1_idx]),
                "top1_reference_label": col_labels[top1_idx],
                "top1_reference_id": top1_molecule["id"],
                "top1_reference_name": top1_molecule["name"],
                "top1_reference_sources": ";".join(sorted(top1_molecule["sources"])),
                "top5_mean_similarity": float(scores[top5_idx].mean()),
                "top10_mean_similarity": float(scores[top10_idx].mean()),
                "top5_reference_labels": ";".join(col_labels[idx] for idx in top5_idx),
                "top5_similarity_values": ";".join(f"{scores[idx]:.6g}" for idx in top5_idx),
                "top10_reference_labels": ";".join(col_labels[idx] for idx in top10_idx),
                "top10_similarity_values": ";".join(f"{scores[idx]:.6g}" for idx in top10_idx),
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(path, index=False)
    return summary_df


def compute_tanimoto_matrix(row_molecules: List[dict], col_molecules: List[dict], radius: int, n_bits: int) -> np.ndarray:
    row_fps = [morgan_fp(item["mol"], radius, n_bits) for item in row_molecules]
    col_fps = [morgan_fp(item["mol"], radius, n_bits) for item in col_molecules]
    return np.asarray(
        [[DataStructs.TanimotoSimilarity(row_fp, col_fp) for col_fp in col_fps] for row_fp in row_fps],
        dtype=float,
    )


def compute_descriptor_euclidean_matrix(row_molecules: List[dict], col_molecules: List[dict]) -> np.ndarray:
    row_descriptors = np.asarray([descriptor_vector(item["mol"]) for item in row_molecules], dtype=float)
    col_descriptors = np.asarray([descriptor_vector(item["mol"]) for item in col_molecules], dtype=float)
    scaler = StandardScaler()
    all_descriptors = scaler.fit_transform(np.vstack([row_descriptors, col_descriptors]))
    row_scaled = all_descriptors[: len(row_molecules)]
    col_scaled = all_descriptors[len(row_molecules) :]
    return pairwise_distances(row_scaled, col_scaled, metric="euclidean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lht2_file", default=str(DEFAULT_LHT2_FILE), help="LHT2 CSV file")
    parser.add_argument(
        "--reference_files",
        default=",".join(str(path) for path in DEFAULT_REFERENCE_FILES),
        help="comma-separated reference pair CSV files",
    )
    parser.add_argument("--fp_radius", type=int, default=4, help="Morgan fingerprint radius")
    parser.add_argument("--fp_bits", type=int, default=2048, help="Morgan fingerprint bit length")
    parser.add_argument("--out_dir", default="scripts_outputs/lht2_similarity", help="output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row_molecules = read_lht2_molecules(args.lht2_file)
    reference_files = parse_csv_paths(args.reference_files)
    col_molecules = read_reference_molecules(reference_files)
    reference_occurrences = read_reference_occurrences(reference_files)
    if not row_molecules:
        raise ValueError("no valid molecules found in LHT2 file")
    if not col_molecules:
        raise ValueError("no valid reference molecules found")

    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row_labels = make_unique_labels(row_molecules, "lht2")
    col_labels = make_unique_labels(col_molecules, "reference")

    tanimoto = compute_tanimoto_matrix(row_molecules, col_molecules, args.fp_radius, args.fp_bits)
    euclidean = compute_descriptor_euclidean_matrix(row_molecules, col_molecules)

    pd.DataFrame(tanimoto, index=row_labels, columns=col_labels).to_csv(out_dir / "lht2_vs_training_tanimoto_similarity.csv")
    pd.DataFrame(euclidean, index=row_labels, columns=col_labels).to_csv(
        out_dir / "lht2_vs_training_descriptor_euclidean_distance.csv"
    )
    topk_df = save_tanimoto_topk_summary(
        tanimoto,
        row_molecules,
        col_molecules,
        row_labels,
        col_labels,
        out_dir / "lht2_tanimoto_topk_summary.csv",
    )
    save_molecule_table(row_molecules, row_labels, out_dir / "lht2_molecules.csv")
    save_molecule_table(col_molecules, col_labels, out_dir / "training_reference_molecules.csv")
    occurrence_df = save_occurrence_table(
        row_molecules,
        row_labels,
        reference_occurrences,
        reference_files,
        out_dir / "lht2_reference_occurrence.csv",
    )
    pd.DataFrame(
        [
            {
                "lht2_molecules": len(row_molecules),
                "reference_molecules": len(col_molecules),
                "lht2_molecules_in_reference": int(occurrence_df["appears_in_reference"].sum()),
                "lht2_molecules_not_in_reference": int((~occurrence_df["appears_in_reference"]).sum()),
                "fp_radius": args.fp_radius,
                "fp_bits": args.fp_bits,
                "descriptor_count": len(Descriptors._descList),
                "tanimoto_file": "lht2_vs_training_tanimoto_similarity.csv",
                "euclidean_file": "lht2_vs_training_descriptor_euclidean_distance.csv",
                "tanimoto_topk_file": "lht2_tanimoto_topk_summary.csv",
                "mean_max_similarity": float(topk_df["max_similarity"].mean()),
                "mean_top5_similarity": float(topk_df["top5_mean_similarity"].mean()),
                "mean_top10_similarity": float(topk_df["top10_mean_similarity"].mean()),
            }
        ]
    ).to_csv(out_dir / "matrix_summary.csv", index=False)

    print(f"LHT2 molecules: {len(row_molecules)}")
    print(f"Reference molecules: {len(col_molecules)}")
    print(f"LHT2 molecules in reference files: {int(occurrence_df['appears_in_reference'].sum())}")
    print(f"Saved matrices to {out_dir}")


if __name__ == "__main__":
    main()
