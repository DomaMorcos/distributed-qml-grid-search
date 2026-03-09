# Distributed Quantum Machine Learning Grid Search

Orchestrating parameterized **Variational Quantum Classifier (VQC)** hyperparameter
grid search across an HPC cluster using containerized Qiskit simulations, Slurm
job arrays, and automated CI/CD.

> Built as a micro-project demonstrating HPC + quantum simulation integration skills
> relevant to semiconductor qubit characterization workflows.

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
│          Slurm HPC Node (Vagrant + Ansible)              │
│   ┌───────────────────────────────────────────┐         │
│   │  slurmctld + slurmd + Docker              │         │
│   │  Job Array: --array=1-N                   │         │
│   │  Each task trains VQC with unique params  │         │
│   └───────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
distributed-qml-grid-search/
├── train_vqc.py             # VQC training script (Qiskit)
├── params.csv               # Hyperparameter grid (task_id → config)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Multi-stage container build
├── run_grid_search.sh       # Slurm job array launcher
├── infrastructure/          # Vagrant + Ansible provisioning
│   ├── Vagrantfile          # Single-node VM definition
│   ├── playbooks/           # Ansible roles (Slurm, Docker, deploy)
│   ├── templates/           # slurm.conf, cgroup.conf, hosts
│   └── inventory/           # Ansible inventory
├── .github/workflows/
│   └── docker.yml           # CI/CD: lint + build + push
└── results/                 # JSON output from each training run
```

## Quick Start

### 1. Local (single run)

```bash
pip install -r requirements.txt
python train_vqc.py --reps 2 --optimizer COBYLA --max-iter 100
```

### 2. Docker (single run)

```bash
docker build -t qml-grid .
docker run --rm -v "$(pwd)/results:/app/results" qml-grid \
    --reps 2 --optimizer COBYLA --max-iter 100
```

### 3. Slurm cluster (full grid search)

```bash
# Build or pull the image on all compute nodes first
docker build -t qml-grid .

# Submit the 4-task array
sbatch run_grid_search.sh

# Monitor
squeue -u $USER
watch -n5 'ls -la results/'
```

## Hyperparameter Grid

Defined in `params.csv`:

| task_id | reps | optimizer | max_iter |
|---------|------|-----------|----------|
| 1       | 1    | COBYLA    | 80       |
| 2       | 2    | COBYLA    | 100      |
| 3       | 3    | SPSA     | 100      |
| 4       | 2    | L_BFGS_B | 120      |

To expand the grid, add rows to `params.csv` and update `--array=1-N` in
`run_grid_search.sh`.

## Infrastructure

The local lab uses a **single lightweight VM** running both `slurmctld` and `slurmd` plus Docker.
This keeps host resource usage minimal while still demonstrating real Slurm job scheduling with queuing.

```bash
cd infrastructure/
vagrant up          # Boot + provision via Ansible
vagrant ssh qml-node -c "sinfo -N -l"   # Verify cluster
```

Slurm schedules the 4-task job array as 2 concurrent + 2 queued (1 CPU per task, 2 CPUs on node).
This is standard HPC behavior and demonstrates real job queuing.

## CI/CD

The GitHub Actions pipeline (`.github/workflows/docker.yml`) runs on every push to `main`:

1. **Lint** — Checks `train_vqc.py` with `ruff`.
2. **Build** — Builds the Docker image with BuildKit caching.
3. **Push** — Pushes to `ghcr.io/<owner>/qml-grid` tagged with the git SHA and `latest`.


