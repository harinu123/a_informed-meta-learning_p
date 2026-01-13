import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from dataset.atom3d_lba_poc import Atom3DLBAPOC
from models.inp import INP


def evaluate(
    model: INP,
    dataset: Atom3DLBAPOC,
    context_sizes: List[int],
    device: torch.device,
    use_knowledge: bool,
) -> Dict[int, float]:
    rng = np.random.default_rng(0)
    rmse_by_context = {m: [] for m in context_sizes}

    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            x, y, knowledge = dataset[idx]
            x = x.to(device)
            y = y.to(device)
            if use_knowledge:
                knowledge = knowledge.to(device)

            x_target = x.unsqueeze(0)
            y_target = y.unsqueeze(0)
            knowledge_batch = knowledge.unsqueeze(0) if use_knowledge else None

            n_points = x.shape[0]
            for m in context_sizes:
                if m == 0:
                    x_context = torch.zeros((1, 0, x.shape[-1]), device=device)
                    y_context = torch.zeros((1, 0, y.shape[-1]), device=device)
                else:
                    replace = n_points < m
                    idxs = rng.choice(n_points, m, replace=replace)
                    x_context = x[idxs].unsqueeze(0)
                    y_context = y[idxs].unsqueeze(0)

                pred_dist, _, _, _ = model(
                    x_context,
                    y_context,
                    x_target,
                    y_target=None,
                    knowledge=knowledge_batch,
                )
                pred = pred_dist.mean
                rmse = torch.sqrt(torch.mean((pred - y_target) ** 2)).item()
                rmse_by_context[m].append(rmse)

    return {m: float(np.mean(v)) for m, v in rmse_by_context.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to model_best.pt")
    parser.add_argument("--config", default="config.toml", help="Path to config file")
    parser.add_argument(
        "--results_path",
        default=None,
        help="Path to write results.json (defaults near checkpoint)",
    )
    parser.add_argument("--plot", action="store_true", help="Save plot to PNG")
    args = parser.parse_args()

    config = Config.from_toml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    dataset = Atom3DLBAPOC(
        split="test",
        num_points=config.num_targets,
        shuffle_knowledge=getattr(config, "shuffle_knowledge", False),
        seed=config.seed,
    )
    config.knowledge_input_dim = dataset.knowledge_input_dim

    model = INP(config).to(device)
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    context_sizes = [0, 1, 3, 5, 10, 15]
    results = evaluate(model, dataset, context_sizes, device, config.use_knowledge)

    if args.results_path is None:
        ckpt_dir = os.path.dirname(args.ckpt)
        args.results_path = os.path.join(ckpt_dir, "results.json")
    with open(args.results_path, "w") as f:
        json.dump({"rmse": results}, f, indent=2)

    print("RMSE by context size:")
    for m in context_sizes:
        print(f"  {m}: {results[m]:.4f}")

    if args.plot:
        import matplotlib.pyplot as plt

        x_vals = list(results.keys())
        y_vals = [results[m] for m in x_vals]
        plt.figure()
        plt.plot(x_vals, y_vals, marker="o")
        plt.xlabel("# context points")
        plt.ylabel("RMSE")
        plt.title("ATOM3D-LBA POC context curve")
        plot_path = os.path.splitext(args.results_path)[0] + ".png"
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
