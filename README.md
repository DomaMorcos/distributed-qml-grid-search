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
│   ├── playbooks/           # Ansible roles (Slurm, Docker, deploy, monitoring)
│   ├── templates/           # slurm.conf, cgroup.conf, hosts
│   ├── inventory/           # Ansible inventory
│   └── monitoring/          # Prometheus + Grafana configs
│       ├── prometheus.yml   # Prometheus scrape targets
│       ├── slurm_exporter.py # Lightweight Slurm metrics exporter
│       └── grafana/         # Dashboard + datasource provisioning
├── results/                 # JSON output from each training run
│   ├── result_task1_reps1_COBYLA_iter80.json
│   ├── result_task2_reps2_COBYLA_iter100.json
│   ├── result_task3_reps3_SPSA_iter100.json
│   └── result_task4_reps2_L_BFGS_B_iter120.json
├── .github/workflows/
│   └── docker.yml           # CI/CD: lint + build + push
└── README.md
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

## Results

Each task produces a JSON file in `results/` containing the hyperparameters, accuracies, and
training time. Below is a summary of the completed grid search:

| Task | Reps | Optimizer | Max Iter | Train Acc | Test Acc | Time (s) |
|------|------|-----------|----------|-----------|----------|----------|
| 1    | 1    | COBYLA    | 80       | 0.6571    | 0.5667   | 9.08     |
| 2    | 2    | COBYLA    | 100      | 0.7286    | 0.6333   | 12.49    |
| 3    | 3    | SPSA      | 100      | 0.6857    | 0.5667   | 32.13    |
| 4    | 2    | L_BFGS_B  | 120      | 0.7143    | 0.5000   | 121.27   |

**Best configuration**: Task 2 (reps=2, COBYLA, 100 iter) achieved the highest test accuracy
of **63.3%** with a fast training time of 12.49s.

<details>
<summary>Example result JSON (Task 2)</summary>

```json
{
  "reps": 2,
  "optimizer": "COBYLA",
  "max_iter": 100,
  "train_accuracy": 0.7286,
  "test_accuracy": 0.6333,
  "training_time_s": 12.49,
  "n_qubits": 4,
  "seed": 42,
  "task_id": 2
}
```

</details>

## Monitoring (Prometheus + Grafana)

The cluster is monitored using a full observability stack deployed via Ansible:

| Component          | Port  | Purpose                                   |
|--------------------|-------|-------------------------------------------|
| **Prometheus**     | 9090  | Time-series metrics collection & queries  |
| **Grafana**        | 3000  | Dashboards & visualization (admin/admin)  |
| **node_exporter**  | 9100  | System metrics (CPU, memory, disk, net)   |
| **slurm_exporter** | 9092  | Slurm job queue & node state metrics      |

### Monitoring Architecture

```
┌────────────────────────────────────────────────────┐
│                  Grafana (:3000)                    │
│   Dashboard: QML Grid Search — Cluster Monitor     │
│   Panels: CPU, Memory, Disk I/O, Network,          │
│           Slurm Jobs, Node States, CPU Allocation   │
└─────────────────────┬──────────────────────────────┘
                      │ queries
                      ▼
┌────────────────────────────────────────────────────┐
│               Prometheus (:9090)                    │
│   Scrape interval: 15s (node) / 30s (slurm)       │
│   Retention: 7 days                                │
└───────┬──────────────────────┬─────────────────────┘
        │ scrape               │ scrape
        ▼                      ▼
┌──────────────────┐  ┌───────────────────────┐
│ node_exporter    │  │ slurm_exporter.py     │
│  :9100           │  │  :9092                │
│ CPU, mem, disk,  │  │ squeue/sinfo parsing  │
│ network, load    │  │ job counts, node      │
│                  │  │ states, CPU alloc     │
└──────────────────┘  └───────────────────────┘
```

### Deploy Monitoring

Monitoring is included in the full provisioning (`vagrant up`), or can be deployed separately:

```bash
cd infrastructure/
ansible-playbook -i inventory/hosts.ini playbooks/deploy-monitoring.yml
```

After deployment:
- **Prometheus UI**: http://192.168.56.10:9090
- **Grafana UI**: http://192.168.56.10:3000 (default login: admin / admin)
- **Grafana Dashboard**: Pre-provisioned "QML Grid Search — Cluster Monitor"

### Monitoring Proof — Live Capture During Job Execution

The following output was captured from the running cluster while Slurm jobs were actively
executing the VQC grid search. This proves end-to-end monitoring integration.

**Prometheus targets (all healthy):**

```
Prometheus Targets:
          node  health=up  url=http://qml-node:9100/metrics
    prometheus  health=up  url=http://localhost:9090/metrics
         slurm  health=up  url=http://qml-node:9092/metrics
```

**Live metrics during job execution (queried from Prometheus API):**

```
CPU Usage:           70.6%
Memory Usage:        32.9%
Load 1m:             1.24
Slurm Running Jobs:  1
Slurm Pending Jobs:  0
Slurm CPUs:          1 / 2
```

**Slurm exporter metrics (raw Prometheus format):**

```
# HELP slurm_queue_jobs Number of Slurm jobs by state
# TYPE slurm_queue_jobs gauge
slurm_queue_jobs{state="running"} 2
slurm_queue_jobs{state="pending"} 0
slurm_nodes_state{state="idle"} 0
slurm_nodes_state{state="alloc"} 0
slurm_cpus_alloc 2
slurm_cpus_total 2
```

**Grafana provisioned dashboard and datasource:**

```
Grafana Dashboards:
  QML Grid Search — Cluster Monitor  uid=qml-cluster-monitor

Grafana Datasources:
  Prometheus  type=prometheus  url=http://localhost:9090

Grafana Health:
  version=10.4.1  database=ok
```

**Slurm queue during execution (2 running, 2 pending on single node):**

```
$ squeue -l
             JOBID PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
          14_[3-4]   compute qml-grid  vagrant  PENDING       0:00     30:00      1 (Resources)
              14_1   compute qml-grid  vagrant  RUNNING       0:05     30:00      1 qml-node
              14_2   compute qml-grid  vagrant  RUNNING       0:05     30:00      1 qml-node
```

## Scaling to Multiple Nodes

The current setup uses a single all-in-one VM for portability. To scale to a multi-node
cluster when more resources are available:

### 1. Add Compute Nodes to Vagrantfile

```ruby
# infrastructure/Vagrantfile — Multi-node example
NODES = {
  "qml-master"  => { ip: "192.168.56.10", cpus: 2, memory: 2048, role: "master" },
  "qml-node-01" => { ip: "192.168.56.11", cpus: 4, memory: 4096, role: "compute" },
  "qml-node-02" => { ip: "192.168.56.12", cpus: 4, memory: 4096, role: "compute" },
}

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/jammy64"

  NODES.each do |name, spec|
    config.vm.define name do |node|
      node.vm.hostname = name
      node.vm.network "private_network", ip: spec[:ip]
      node.vm.provider "virtualbox" do |vb|
        vb.cpus   = spec[:cpus]
        vb.memory = spec[:memory]
      end
    end
  end
end
```

### 2. Update Ansible Inventory

```ini
[master]
qml-master ansible_host=192.168.56.10

[compute]
qml-node-01 ansible_host=192.168.56.11
qml-node-02 ansible_host=192.168.56.12

[cluster:children]
master
compute
```

### 3. Update slurm.conf Template

```
# Add all nodes
NodeName=qml-node-01 NodeAddr=192.168.56.11 CPUs=4 RealMemory=3800 State=UNKNOWN
NodeName=qml-node-02 NodeAddr=192.168.56.12 CPUs=4 RealMemory=3800 State=UNKNOWN

# Partition includes all compute nodes
PartitionName=compute Nodes=qml-node-[01-02] Default=YES MaxTime=01:00:00 State=UP
```

### 4. Update Prometheus Targets

```yaml
# infrastructure/monitoring/prometheus.yml
scrape_configs:
  - job_name: "node"
    static_configs:
      - targets:
          - "qml-master:9100"
          - "qml-node-01:9100"
          - "qml-node-02:9100"
```

### 5. Scale the Grid Search

```bash
# Expand params.csv with more hyperparameter combinations
# Update the job array range
#SBATCH --array=1-16

# With 2 compute nodes × 4 CPUs = 8 concurrent tasks
# 16 tasks complete in 2 waves instead of 8 on a single node
```

### Key Scaling Considerations

- **Munge key**: Must be identical on all nodes (the Ansible playbook handles this)
- **Docker image**: Must be available on all compute nodes (use a registry or `docker save/load`)
- **Shared storage**: For production, use NFS or a parallel filesystem for `results/`
- **Monitoring**: Deploy `node_exporter` on every node; Prometheus scrapes all targets automatically
- **Network**: All nodes must be on the same private network and able to resolve hostnames

