import os
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Atom3DLBAPOC(Dataset):
    def __init__(
        self,
        split: str = "train",
        root: str = "data/atom3d_lba_poc",
        num_points: int = 32,
        template_mode: str = "within",
        shuffle_knowledge: bool = False,
        seed: Optional[int] = None,
    ):
        self.root = root
        self.split = split
        self.num_points = num_points
        self.template_mode = template_mode
        self.shuffle_knowledge = shuffle_knowledge
        self.seed = seed

        tasks_path = os.path.join(root, "tasks.pt")
        splits_path = os.path.join(root, "splits.csv")
        if not os.path.exists(tasks_path):
            raise FileNotFoundError(
                f"Missing tasks file at {tasks_path}. Run scripts/preprocess_atom3d_lba_poc.py"
            )
        if not os.path.exists(splits_path):
            raise FileNotFoundError(
                f"Missing splits file at {splits_path}. Run scripts/preprocess_atom3d_lba_poc.py"
            )

        self.tasks: Dict[str, Dict[str, torch.Tensor]] = torch.load(tasks_path)
        splits_df = pd.read_csv(splits_path)
        self.task_ids = (
            splits_df[splits_df["split"] == split]["task_id"].astype(str).tolist()
        )
        self.task_ids = [t for t in self.task_ids if t in self.tasks]

        if len(self.task_ids) == 0:
            raise ValueError(f"No tasks found for split={split} in {root}")

        self.dim_x = 300
        self.dim_y = 1
        self.knowledge_input_dim = 300

        self._rng = np.random.default_rng(seed)
        self._shuffle_map: Optional[Dict[str, str]] = None
        if self.shuffle_knowledge:
            permuted = self.task_ids.copy()
            self._rng.shuffle(permuted)
            self._shuffle_map = dict(zip(self.task_ids, permuted))

    def __len__(self) -> int:
        return len(self.task_ids)

    def _sample_indices(self, n: int) -> np.ndarray:
        replace = n < self.num_points
        return self._rng.choice(n, self.num_points, replace=replace)

    def __getitem__(self, idx: int):
        task_id = self.task_ids[idx]
        task = self.tasks[task_id]
        x_t = task["x"]
        y_t = task["y"]
        k_t = task["k"]

        sample_idx = self._sample_indices(x_t.shape[0])
        knowledge_task_id = task_id
        if self.shuffle_knowledge and self._shuffle_map is not None:
            knowledge_task_id = self._shuffle_map[task_id]
        knowledge_task = self.tasks[knowledge_task_id]
        if self.template_mode == "within":
            if knowledge_task_id == task_id:
                template_idx = self._rng.choice(sample_idx, 1)[0]
            else:
                template_idx = self._rng.integers(0, knowledge_task["k"].shape[0])
        else:
            template_idx = self._rng.integers(0, knowledge_task["k"].shape[0])
        knowledge = knowledge_task["k"][template_idx]

        x = x_t[sample_idx]
        y = y_t[sample_idx]

        return x, y, knowledge
