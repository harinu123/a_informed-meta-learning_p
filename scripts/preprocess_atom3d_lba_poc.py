import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans


AMINO_ACIDS = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

WATER_IONS = {
    "HOH",
    "WAT",
    "NA",
    "CL",
    "K",
    "ZN",
    "MG",
    "CA",
    "MN",
    "CO",
    "FE",
    "CU",
    "NI",
    "BR",
    "IOD",
    "IODIDE",
    "SO4",
    "PO4",
}

ELEMENT_TO_TYPE = {
    "C": 0,
    "N": 1,
    "O": 2,
    "S": 3,
    "F": 4,
    "CL": 4,
    "BR": 4,
    "I": 4,
}

DIST_EDGES = np.linspace(2.0, 8.0, 13)


def get_atoms_df(example: Dict) -> pd.DataFrame:
    for key in ["atoms", "atoms_df"]:
        if key in example:
            return example[key]
    raise KeyError("Example does not contain atoms dataframe")


def get_coord_columns(atoms_df: pd.DataFrame) -> List[str]:
    for cols in [("x", "y", "z"), ("X", "Y", "Z")]:
        if all(c in atoms_df.columns for c in cols):
            return list(cols)
    raise KeyError("Atoms dataframe missing coordinate columns")


def element_to_type(element: str) -> int:
    if not isinstance(element, str):
        return 0
    element = element.upper()
    return ELEMENT_TO_TYPE.get(element, 0)


def bin_distances(distances: np.ndarray) -> np.ndarray:
    bins = np.digitize(distances, DIST_EDGES, right=False) - 1
    bins = np.clip(bins, 0, 11)
    return bins


def select_ligand_atoms(atoms_df: pd.DataFrame) -> pd.DataFrame:
    element_series = atoms_df["element"].fillna("").astype(str).str.upper()
    heavy_atoms = atoms_df[element_series != "H"]
    non_protein = ~heavy_atoms["resname"].isin(AMINO_ACIDS)
    not_water = ~heavy_atoms["resname"].isin(WATER_IONS)
    ligand_candidates = heavy_atoms[non_protein & not_water]

    if ligand_candidates.empty:
        return pd.DataFrame(columns=atoms_df.columns)

    grouped = ligand_candidates.groupby(["chain", "resid", "resname"], sort=False)
    ligand_group = max(grouped, key=lambda x: len(x[1]))
    return ligand_group[1]


def select_protein_atoms(atoms_df: pd.DataFrame) -> pd.DataFrame:
    element_series = atoms_df["element"].fillna("").astype(str).str.upper()
    heavy_atoms = atoms_df[element_series != "H"]
    return heavy_atoms[heavy_atoms["resname"].isin(AMINO_ACIDS)]


def select_pocket_atoms(
    protein_atoms: pd.DataFrame, ligand_atoms: pd.DataFrame, cutoff: float = 6.0
) -> pd.DataFrame:
    if protein_atoms.empty or ligand_atoms.empty:
        return pd.DataFrame(columns=protein_atoms.columns)
    coord_cols = get_coord_columns(protein_atoms)
    protein_coords = protein_atoms[coord_cols].to_numpy(dtype=np.float32)
    ligand_coords = ligand_atoms[coord_cols].to_numpy(dtype=np.float32)
    distances = np.linalg.norm(
        protein_coords[:, None, :] - ligand_coords[None, :, :], axis=2
    )
    mask = distances.min(axis=1) <= cutoff
    return protein_atoms.iloc[np.where(mask)[0]]


def compute_ligand_distogram(ligand_atoms: pd.DataFrame) -> np.ndarray:
    coord_cols = get_coord_columns(ligand_atoms)
    coords = ligand_atoms[coord_cols].to_numpy(dtype=np.float32)
    types = ligand_atoms["element"].apply(element_to_type).to_numpy(dtype=np.int64)
    n = coords.shape[0]
    hist = np.zeros((5, 5, 12), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            bin_idx = int(bin_distances(np.array([dist]))[0])
            ti, tj = types[i], types[j]
            hist[ti, tj, bin_idx] += 1
            hist[tj, ti, bin_idx] += 1
    denom = hist.sum() + 1e-8
    hist = hist / denom
    if hist.sum() > 0:
        assert np.isclose(hist.sum(), 1.0, atol=1e-3)
    return hist.flatten()


def compute_pocket_ligand_distogram(
    pocket_atoms: pd.DataFrame, ligand_atoms: pd.DataFrame
) -> np.ndarray:
    coord_cols = get_coord_columns(ligand_atoms)
    pocket_coords = pocket_atoms[coord_cols].to_numpy(dtype=np.float32)
    ligand_coords = ligand_atoms[coord_cols].to_numpy(dtype=np.float32)
    pocket_types = pocket_atoms["element"].apply(element_to_type).to_numpy(dtype=np.int64)
    ligand_types = ligand_atoms["element"].apply(element_to_type).to_numpy(dtype=np.int64)
    hist = np.zeros((5, 5, 12), dtype=np.float32)
    for i in range(pocket_coords.shape[0]):
        for j in range(ligand_coords.shape[0]):
            dist = np.linalg.norm(pocket_coords[i] - ligand_coords[j])
            bin_idx = int(bin_distances(np.array([dist]))[0])
            ti, tj = pocket_types[i], ligand_types[j]
            hist[ti, tj, bin_idx] += 1
    denom = hist.sum() + 1e-8
    hist = hist / denom
    if hist.sum() > 0:
        assert np.isclose(hist.sum(), 1.0, atol=1e-3)
    return hist.flatten()


def select_label(example: Dict) -> Tuple[float, str]:
    preferred = [
        "label",
        "affinity",
        "binding_affinity",
        "pkd",
        "pki",
        "pic50",
        "ic50",
        "ki",
        "kd",
        "k_i",
        "k_d",
    ]
    numeric_fields = {}
    for key, value in example.items():
        if key in {"atoms", "atoms_df"}:
            continue
        if isinstance(value, (int, float, np.number)):
            numeric_fields[key] = float(value)
        elif isinstance(value, np.ndarray) and value.size == 1:
            numeric_fields[key] = float(value.reshape(-1)[0])
        elif isinstance(value, list) and len(value) == 1 and isinstance(
            value[0], (int, float, np.number)
        ):
            numeric_fields[key] = float(value[0])

    if not numeric_fields:
        raise ValueError("No numeric labels found in example")

    for pref in preferred:
        for key in numeric_fields:
            if key.lower() == pref:
                if len(numeric_fields) > 1:
                    print(
                        f"Multiple numeric labels found. Selected '{key}' from {list(numeric_fields.keys())}"
                    )
                return numeric_fields[key], key
    selected_key = sorted(numeric_fields.keys())[0]
    if len(numeric_fields) > 1:
        print(
            f"Multiple numeric labels found. Selected '{selected_key}' from {list(numeric_fields.keys())}"
        )
    return numeric_fields[selected_key], selected_key


def get_task_id(example: Dict, atoms_df: pd.DataFrame) -> str:
    identity_keys = [
        "uniprot_id",
        "uniprot",
        "protein_id",
        "target_id",
        "pdb_id",
        "pdb",
        "pdb_code",
        "pdbid",
        "pdb_chain",
        "pdb_chain_id",
        "chain",
        "protein",
        "target",
    ]
    for key in identity_keys:
        if key in example and example[key] is not None:
            value = str(example[key])
            if key == "pdb" and "chain" in example and example["chain"] is not None:
                value = f"{value}_{example['chain']}"
            return value

    protein_atoms = select_protein_atoms(atoms_df)
    if protein_atoms.empty:
        return hashlib.sha1("UNKNOWN".encode()).hexdigest()

    ca_atoms = protein_atoms[protein_atoms["name"] == "CA"]
    residue_df = ca_atoms
    if residue_df.empty:
        residue_df = protein_atoms.drop_duplicates(subset=["chain", "resid", "resname"])
    residue_df = residue_df.sort_values(["chain", "resid"])
    sequence = "".join(
        AA3_TO_1.get(res.upper(), "X") for res in residue_df["resname"].tolist()
    )
    return hashlib.sha1(sequence.encode()).hexdigest()


def load_lba_dataset(data_dir: str):
    try:
        from atom3d.datasets import LMDBDataset, download_dataset
    except ImportError as exc:
        raise ImportError(
            "atom3d is required. Please install dependencies via environment.yaml"
        ) from exc

    if not os.path.exists(data_dir):
        base_dir = os.path.dirname(data_dir)
        os.makedirs(base_dir, exist_ok=True)
        download_dataset("lba", base_dir)
        if os.path.exists(os.path.join(base_dir, "lba")):
            data_dir = os.path.join(base_dir, "lba")

    splits = {}
    for split in ["train", "val", "test"]:
        split_path = os.path.join(data_dir, split)
        if os.path.exists(split_path):
            splits[split] = LMDBDataset(split_path)
    if not splits:
        if os.path.exists(os.path.join(data_dir, "lba")):
            return load_lba_dataset(os.path.join(data_dir, "lba"))
        lmdb_path = os.path.join(data_dir, "data.mdb")
        if os.path.exists(lmdb_path):
            print(
                f"No split directories found in {data_dir}. Loading single LMDB as 'all' split."
            )
            splits["all"] = LMDBDataset(data_dir)
            return splits
        if any(name.endswith(".mdb") for name in os.listdir(data_dir)):
            print(
                f"Detected LMDB files in {data_dir}. Loading as 'all' split."
            )
            splits["all"] = LMDBDataset(data_dir)
            return splits
        raw_dir = os.path.join(data_dir, "raw", "pdbbind_2019-refined-set", "data")
        if os.path.exists(os.path.join(raw_dir, "data.mdb")):
            print(
                "Found raw PDBBind LMDB. Loading as 'all' split."
            )
            splits["all"] = LMDBDataset(raw_dir)
            return splits
        raise FileNotFoundError(
            f"No splits found in {data_dir}. Expected train/val/test LMDB dirs or data.mdb."
        )
    return splits


def maybe_cluster_tasks(
    entries: List[Dict],
    min_points_per_task: int,
    min_tasks_threshold: int,
    seed: int,
    max_tasks: Optional[int],
) -> Tuple[Dict[str, List[Dict]], bool]:
    tasks = defaultdict(list)
    for entry in entries:
        tasks[entry["task_id"]].append(entry)

    filtered_tasks = {k: v for k, v in tasks.items() if len(v) >= min_points_per_task}
    if len(filtered_tasks) >= min_tasks_threshold:
        return filtered_tasks, False

    if len(entries) < min_points_per_task:
        return {}, False

    k_vectors = np.stack([e["k"] for e in entries], axis=0)
    n_examples = k_vectors.shape[0]
    target_k = max(20, n_examples // 30)
    if max_tasks is not None:
        target_k = min(target_k, max_tasks)
    n_clusters = min(200, target_k, n_examples)
    n_clusters = max(1, n_clusters)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = kmeans.fit_predict(k_vectors)

    clustered = defaultdict(list)
    for entry, label in zip(entries, labels):
        cluster_id = f"cluster_{label}"
        clustered[cluster_id].append(entry)

    clustered = {
        k: v for k, v in clustered.items() if len(v) >= min_points_per_task
    }
    return clustered, True


def prepare_tasks_for_split(
    split_name: str,
    dataset: Iterable,
    min_points_per_task: int,
    min_tasks_threshold: int,
    seed: int,
    max_tasks: Optional[int],
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, int], str, Counter, Counter]:
    entries = []
    label_field = None
    ligand_counts = Counter()
    pocket_counts = Counter()

    for example in dataset:
        atoms_df = get_atoms_df(example)
        ligand_atoms = select_ligand_atoms(atoms_df)
        if ligand_atoms.empty:
            continue
        protein_atoms = select_protein_atoms(atoms_df)
        if protein_atoms.empty:
            continue
        pocket_atoms = select_pocket_atoms(protein_atoms, ligand_atoms)
        if pocket_atoms.empty:
            continue

        x_vec = compute_ligand_distogram(ligand_atoms)
        k_vec = compute_pocket_ligand_distogram(pocket_atoms, ligand_atoms)

        if not np.isfinite(x_vec).all() or not np.isfinite(k_vec).all():
            continue
        if x_vec.sum() <= 0 or k_vec.sum() <= 0:
            continue

        y_val, chosen_label = select_label(example)
        if label_field is None:
            label_field = chosen_label

        task_id = get_task_id(example, atoms_df)

        ligand_counts[len(ligand_atoms)] += 1
        pocket_counts[len(pocket_atoms)] += 1

        entries.append(
            {
                "task_id": task_id,
                "x": x_vec,
                "y": y_val,
                "k": k_vec,
            }
        )

    tasks, clustered = maybe_cluster_tasks(
        entries, min_points_per_task, min_tasks_threshold, seed, max_tasks
    )
    if clustered:
        print(
            f"[{split_name}] Triggered clustering fallback, created {len(tasks)} tasks"
        )

    if max_tasks is not None and len(tasks) > max_tasks:
        rng = np.random.default_rng(seed)
        selected = rng.choice(list(tasks.keys()), size=max_tasks, replace=False)
        tasks = {k: tasks[k] for k in selected}

    tensor_tasks = {}
    task_sizes = {}
    for task_id, items in tasks.items():
        x = torch.tensor(np.stack([e["x"] for e in items]), dtype=torch.float32)
        y = torch.tensor(np.array([e["y"] for e in items]), dtype=torch.float32).unsqueeze(
            -1
        )
        k = torch.tensor(np.stack([e["k"] for e in items]), dtype=torch.float32)
        tensor_tasks[str(task_id)] = {"x": x, "y": y, "k": k}
        task_sizes[str(task_id)] = len(items)

    return tensor_tasks, task_sizes, label_field, ligand_counts, pocket_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="data/atom3d/lba", help="Path to ATOM3D LBA dataset"
    )
    parser.add_argument(
        "--out_dir", default="data/atom3d_lba_poc", help="Output directory"
    )
    parser.add_argument("--min_points_per_task", type=int, default=10)
    parser.add_argument("--min_tasks_threshold", type=int, default=20)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use_all",
        action="store_true",
        help="Process all splits as a single split",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    splits = load_lba_dataset(args.data_dir)
    if args.use_all:
        combined = []
        for split_ds in splits.values():
            combined.extend(list(split_ds))
        splits = {"all": combined}

    all_tasks = {}
    all_task_sizes = {}
    split_rows = []
    label_field = None
    total_examples = 0
    ligand_counts = Counter()
    pocket_counts = Counter()

    for split_name, dataset in splits.items():
        tasks, task_sizes, split_label, split_ligand_counts, split_pocket_counts = (
            prepare_tasks_for_split(
                split_name,
                dataset,
                args.min_points_per_task,
                args.min_tasks_threshold,
                args.seed,
                args.max_tasks,
            )
        )
        total_examples += sum(task_sizes.values())
        all_tasks.update(tasks)
        all_task_sizes.update(task_sizes)
        for task_id in tasks:
            split_rows.append({"task_id": task_id, "split": split_name})
        ligand_counts.update(split_ligand_counts)
        pocket_counts.update(split_pocket_counts)
        if split_label is not None:
            if label_field is None:
                label_field = split_label
            elif label_field != split_label:
                print(
                    f"[{split_name}] Label field mismatch: previously '{label_field}', now '{split_label}'"
                )

    unique_splits = set(row["split"] for row in split_rows)
    if unique_splits == {"all"}:
        task_ids = sorted(list(all_tasks.keys()))
        rng = np.random.default_rng(args.seed)
        rng.shuffle(task_ids)

        n = len(task_ids)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        train_ids = task_ids[:n_train]
        val_ids = task_ids[n_train : n_train + n_val]
        test_ids = task_ids[n_train + n_val :]

        split_rows = (
            [{"task_id": t, "split": "train"} for t in train_ids]
            + [{"task_id": t, "split": "val"} for t in val_ids]
            + [{"task_id": t, "split": "test"} for t in test_ids]
        )

        print(
            f"[task-split] all -> train/val/test = {len(train_ids)}/{len(val_ids)}/{len(test_ids)}"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(all_tasks, os.path.join(args.out_dir, "tasks.pt"))
    pd.DataFrame(split_rows).to_csv(os.path.join(args.out_dir, "splits.csv"), index=False)

    task_sizes = list(all_task_sizes.values())
    size_hist = dict(Counter(task_sizes))
    meta = {
        "total_examples": total_examples,
        "num_tasks": len(all_tasks),
        "label_field": label_field,
        "task_size_histogram": size_hist,
        "min_task_size": min(task_sizes) if task_sizes else 0,
        "median_task_size": float(np.median(task_sizes)) if task_sizes else 0,
        "max_task_size": max(task_sizes) if task_sizes else 0,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Total examples processed: {total_examples}")
    print(f"Number of tasks kept: {len(all_tasks)}")
    if task_sizes:
        print(
            f"Task sizes (min/median/max): {min(task_sizes)}/{np.median(task_sizes):.1f}/{max(task_sizes)}"
        )
    print(f"Label field chosen: {label_field}")

    print("Ligand atom count distribution:")
    for count, freq in ligand_counts.most_common(10):
        print(f"  {count}: {freq}")

    print("Pocket atom count distribution:")
    for count, freq in pocket_counts.most_common(10):
        print(f"  {count}: {freq}")

    print("Task size histogram:")
    for size, freq in sorted(size_hist.items()):
        print(f"  {size}: {freq}")


if __name__ == "__main__":
    main()
