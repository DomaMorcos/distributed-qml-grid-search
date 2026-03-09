#!/usr/bin/env python3
"""Distributed Variational Quantum Classifier (VQC) Training.

Trains a VQC on the Iris dataset with configurable hyperparameters.
Designed to be invoked as a single grid-search task inside a Slurm job array,
where each task explores a unique hyperparameter combination.

Usage:
    python train_vqc.py --reps 3 --optimizer COBYLA --max-iter 100
    python train_vqc.py --task-id 2 --param-file params.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.algorithms import VQC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported optimizers (Qiskit-compatible)
# ---------------------------------------------------------------------------
OPTIMIZERS = {
    "COBYLA": "qiskit.algorithms.optimizers.COBYLA",
    "SPSA": "qiskit.algorithms.optimizers.SPSA",
    "L_BFGS_B": "qiskit.algorithms.optimizers.L_BFGS_B",
    "ADAM": "qiskit.algorithms.optimizers.ADAM",
}


def _load_optimizer(name: str, max_iter: int):
    """Dynamically instantiate a Qiskit optimizer by name."""
    from qiskit_machine_learning import optimizers as qopt

    cls = getattr(qopt, name, None)
    if cls is None:
        raise ValueError(
            f"Unknown optimizer '{name}'. Supported: {list(OPTIMIZERS)}"
        )
    return cls(maxiter=max_iter)


def prepare_data(
    n_features: int = 4, test_size: float = 0.3, seed: int = 42
) -> tuple:
    """Load Iris (2-class subset) and scale features to [0, pi]."""
    iris = load_iris()
    X = iris.data[:, :n_features]
    y = iris.target

    # Binary classification: class 0 vs class 1
    mask = y < 2
    X, y = X[mask], y[mask]

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=test_size, random_state=seed)


def build_and_train(
    reps: int,
    optimizer_name: str,
    max_iter: int,
    feature_dim: int = 4,
    seed: int = 42,
) -> dict:
    """Build a VQC, train it, and return metrics."""
    logger.info(
        "Hyperparameters: reps=%d  optimizer=%s  max_iter=%d",
        reps, optimizer_name, max_iter,
    )

    X_train, X_test, y_train, y_test = prepare_data(n_features=feature_dim)
    logger.info(
        "Dataset: %d train / %d test samples, %d features",
        len(X_train), len(X_test), feature_dim,
    )

    feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2)
    ansatz = RealAmplitudes(num_qubits=feature_dim, reps=reps)
    optimizer = _load_optimizer(optimizer_name, max_iter)

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=None,  # uses default statevector Sampler
    )

    logger.info("Starting training ...")
    t0 = time.perf_counter()
    vqc.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info("Training completed in %.2f s", elapsed)

    train_score = vqc.score(X_train, y_train)
    test_score = vqc.score(X_test, y_test)
    logger.info("Train accuracy: %.4f", train_score)
    logger.info("Test  accuracy: %.4f", test_score)

    return {
        "reps": reps,
        "optimizer": optimizer_name,
        "max_iter": max_iter,
        "train_accuracy": round(train_score, 4),
        "test_accuracy": round(test_score, 4),
        "training_time_s": round(elapsed, 2),
        "n_qubits": feature_dim,
        "seed": seed,
    }


def load_params_from_csv(path: str, task_id: int) -> dict:
    """Read hyperparameters for a specific task ID from a CSV file.

    The CSV must have columns: task_id,reps,optimizer,max_iter
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Parameter file not found: {path}")

    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["task_id"]) == task_id:
                return {
                    "reps": int(row["reps"]),
                    "optimizer": row["optimizer"],
                    "max_iter": int(row["max_iter"]),
                }
    raise ValueError(f"task_id {task_id} not found in {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a VQC with configurable hyperparameters."
    )
    parser.add_argument(
        "--reps", type=int, default=2,
        help="Number of ansatz repetition layers (default: 2).",
    )
    parser.add_argument(
        "--optimizer", type=str, default="COBYLA",
        choices=list(OPTIMIZERS),
        help="Classical optimizer (default: COBYLA).",
    )
    parser.add_argument(
        "--max-iter", type=int, default=100,
        help="Maximum optimizer iterations (default: 100).",
    )
    parser.add_argument(
        "--task-id", type=int, default=None,
        help="Slurm array task ID. Overrides --reps/--optimizer if --param-file is set.",
    )
    parser.add_argument(
        "--param-file", type=str, default=None,
        help="CSV file mapping task IDs to hyperparameters.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for JSON result files (default: results/).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Resolve hyperparameters ------------------------------------------------
    if args.param_file and args.task_id is not None:
        params = load_params_from_csv(args.param_file, args.task_id)
        reps = params["reps"]
        optimizer_name = params["optimizer"]
        max_iter = params["max_iter"]
    else:
        reps = args.reps
        optimizer_name = args.optimizer
        max_iter = args.max_iter

    # Train ------------------------------------------------------------------
    metrics = build_and_train(
        reps=reps,
        optimizer_name=optimizer_name,
        max_iter=max_iter,
        seed=args.seed,
    )

    # Persist results --------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"reps{reps}_{optimizer_name}_iter{max_iter}"
    if args.task_id is not None:
        tag = f"task{args.task_id}_{tag}"
        metrics["task_id"] = args.task_id

    out_path = out_dir / f"result_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Results saved to %s", out_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
