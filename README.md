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
│              Slurm HPC Cluster (Vagrant)                 │
│   ┌─────────┐  ┌──────────┐  ┌──────────┐              │
│   │ Master  │  │ Compute1 │  │ Compute2 │              │
│   └─────────┘  └──────────┘  └──────────┘              │
│         Job Array: --array=1-N                           │
│   Each task trains VQC with unique hyperparameters       │
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

## Vagrant VM Resource Mapping (i7-13620H, 10 cores / 16 GB)

For a 1-master + 2-compute Slurm cluster on a 10-core i7-13620H (6P + 4E cores),
allocate resources so your host stays responsive:

| Node     | vCPUs | RAM   | Notes                              |
|----------|-------|-------|------------------------------------|
| **Host** | 2     | 4 GB  | Reserved — never allocate to VMs   |
| master   | 2     | 2 GB  | Runs slurmctld, no compute tasks   |
| compute1 | 3     | 4 GB  | Runs slurmd + Docker containers    |
| compute2 | 3     | 4 GB  | Runs slurmd + Docker containers    |
| **Total**| **10**| **14 GB** | Leaves headroom for Fedora + IDE |

**Vagrantfile snippet:**

```ruby
Vagrant.configure("2") do |config|
  config.vm.define "master" do |m|
    m.vm.provider "virtualbox" do |v|
      v.cpus   = 2
      v.memory = 2048
    end
  end

  (1..2).each do |i|
    config.vm.define "compute#{i}" do |c|
      c.vm.provider "virtualbox" do |v|
        v.cpus   = 3
        v.memory = 4096
      end
    end
  end
end
```

**Key tuning tips:**

- The Slurm script requests `--cpus-per-task=2` and `--mem=2G` per array task.
  With 3 vCPUs per compute node, one task runs per node with 1 spare core for OS overhead.
- To run 2 tasks per node, reduce to `--cpus-per-task=1` and `--mem=1500M`.
- Keep Docker's `--cpus` and `--memory` flags aligned with Slurm allocations (already handled in `run_grid_search.sh`).
- Disable GUI/compositor on Fedora before running: `sudo systemctl isolate multi-user.target`
  (switch back with `sudo systemctl isolate graphical.target`).

## CI/CD

The GitHub Actions pipeline (`.github/workflows/docker.yml`) runs on every push to `main`:

1. **Lint** — Checks `train_vqc.py` with `ruff`.
2. **Build** — Builds the Docker image with BuildKit caching.
3. **Push** — Pushes to `ghcr.io/<owner>/qml-grid` tagged with the git SHA and `latest`.

## License

MIT
