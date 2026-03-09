#!/usr/bin/env python3
"""Lightweight Prometheus exporter for Slurm metrics.

Exposes job queue and node state metrics by parsing squeue/sinfo output.
Listens on :9092/metrics by default.
"""

import subprocess
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import argparse
import re


def run_cmd(cmd):
    """Run a shell command and return stdout lines."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []
        return result.stdout.strip().split("\n") if result.stdout.strip() else []
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def collect_job_metrics():
    """Parse squeue to collect job state counts."""
    states = {"RUNNING": 0, "PENDING": 0, "COMPLETED": 0, "FAILED": 0, "CANCELLED": 0}
    lines = run_cmd("squeue -h -o '%T' --all")
    for line in lines:
        state = line.strip().upper()
        if state in states:
            states[state] += 1
        elif state:
            states.setdefault(state, 0)
            states[state] += 1
    return states


def collect_node_metrics():
    """Parse sinfo to collect node state counts and CPU info."""
    node_states = {"idle": 0, "alloc": 0, "mix": 0, "drain": 0, "down": 0}
    cpus_alloc = 0
    cpus_total = 0

    lines = run_cmd("sinfo -h -N -o '%N %T %C'")
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            state = parts[1].lower().rstrip("*")
            # State might be compound like "idle~"
            base_state = re.split(r"[~+#!%$@^]", state)[0]
            if base_state in node_states:
                node_states[base_state] += 1
            # CPU format: A/I/O/T (allocated/idle/other/total)
            cpu_parts = parts[2].split("/")
            if len(cpu_parts) == 4:
                try:
                    cpus_alloc += int(cpu_parts[0])
                    cpus_total += int(cpu_parts[3])
                except ValueError:
                    pass

    return node_states, cpus_alloc, cpus_total


def generate_metrics():
    """Generate Prometheus-format metrics text."""
    lines = []

    # Job metrics
    job_states = collect_job_metrics()
    lines.append("# HELP slurm_queue_jobs Number of Slurm jobs by state")
    lines.append("# TYPE slurm_queue_jobs gauge")
    for state, count in job_states.items():
        lines.append(f'slurm_queue_jobs{{state="{state.lower()}"}} {count}')

    # Convenience aliases (for dashboard compatibility)
    lines.append("# HELP slurm_queue_running Running jobs in queue")
    lines.append("# TYPE slurm_queue_running gauge")
    lines.append(f"slurm_queue_running {job_states.get('RUNNING', 0)}")
    lines.append("# HELP slurm_queue_pending Pending jobs in queue")
    lines.append("# TYPE slurm_queue_pending gauge")
    lines.append(f"slurm_queue_pending {job_states.get('PENDING', 0)}")
    lines.append("# HELP slurm_queue_completed Completed jobs")
    lines.append("# TYPE slurm_queue_completed gauge")
    lines.append(f"slurm_queue_completed {job_states.get('COMPLETED', 0)}")

    # Node metrics
    node_states, cpus_alloc, cpus_total = collect_node_metrics()
    lines.append("# HELP slurm_nodes_state Number of Slurm nodes by state")
    lines.append("# TYPE slurm_nodes_state gauge")
    for state, count in node_states.items():
        lines.append(f'slurm_nodes_state{{state="{state}"}} {count}')

    lines.append("# HELP slurm_nodes_idle Idle nodes")
    lines.append("# TYPE slurm_nodes_idle gauge")
    lines.append(f"slurm_nodes_idle {node_states.get('idle', 0)}")
    lines.append("# HELP slurm_nodes_alloc Allocated nodes")
    lines.append("# TYPE slurm_nodes_alloc gauge")
    lines.append(f"slurm_nodes_alloc {node_states.get('alloc', 0)}")
    lines.append("# HELP slurm_nodes_drain Drained nodes")
    lines.append("# TYPE slurm_nodes_drain gauge")
    lines.append(f"slurm_nodes_drain {node_states.get('drain', 0)}")

    lines.append("# HELP slurm_cpus_alloc Allocated CPUs")
    lines.append("# TYPE slurm_cpus_alloc gauge")
    lines.append(f"slurm_cpus_alloc {cpus_alloc}")
    lines.append("# HELP slurm_cpus_total Total CPUs in cluster")
    lines.append("# TYPE slurm_cpus_total gauge")
    lines.append(f"slurm_cpus_total {cpus_total}")

    return "\n".join(lines) + "\n"


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            body = generate_metrics().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            body = b"<html><body><h1>Slurm Exporter</h1><p><a href='/metrics'>Metrics</a></p></body></html>"
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # Suppress request logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prometheus Slurm Exporter")
    parser.add_argument("--port", type=int, default=9092, help="Port to listen on")
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), MetricsHandler)
    print(f"Slurm exporter listening on :{args.port}/metrics")
    server.serve_forever()
