# Distributed Quantum Machine Learning Grid Search

Orchestrating parameterized Variational Quantum Classifier (VQC) hyperparameter
grid search across an HPC cluster using containerized Qiskit simulations, Slurm
job arrays, and automated CI/CD.

## Project Status

🚧 Under active development.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   GitHub Actions CI/CD                   │
│            (Auto-build Docker image on push)             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│               Docker Container (qml-grid)                │
│   Python 3.11 · Qiskit · Qiskit-ML · Scikit-learn       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Slurm HPC Cluster (Vagrant)                 │
│   ┌─────────┐  ┌──────────┐  ┌──────────┐              │
│   │ Master  │  │ Compute1 │  │ Compute2 │              │
│   └─────────┘  └──────────┘  └──────────┘              │
│         Job Array: --array=1-N                           │
│   Each task trains VQC with unique hyperparameters       │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Build the container
docker build -t qml-grid .

# Run a single training locally
docker run --rm qml-grid python train_vqc.py --reps 2 --optimizer COBYLA

# Submit the full grid search to Slurm
sbatch run_grid_search.sh
```

## License

MIT
