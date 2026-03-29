import asyncio
import csv
import logging
import os
import tempfile
from pathlib import Path

import docker
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Benchmark Service")

try:
    docker_client = docker.from_env()
except Exception:
    docker_client = None
    logger.warning("Docker socket unavailable — resource stats will be skipped")


EMBED_ENDPOINT = os.getenv("EMBED_ENDPOINT", "/embed")


class BenchmarkRequest(BaseModel):
    users: int = 10
    spawn_rate: int = 1
    duration: int = 30
    target_service: str = "app"
    endpoint: str = EMBED_ENDPOINT


class ResourceCollector:
    def __init__(self):
        self.cpu_samples: list[float] = []
        self.mem_samples: list[float] = []

    def sample(self, container) -> None:
        stats = container.stats(stream=False)

        cpu_delta = (
            stats["cpu_stats"]["cpu_usage"]["total_usage"]
            - stats["precpu_stats"]["cpu_usage"]["total_usage"]
        )
        system_delta = (
            stats["cpu_stats"]["system_cpu_usage"]
            - stats["precpu_stats"]["system_cpu_usage"]
        )
        num_cpus = stats["cpu_stats"].get("online_cpus", 1)
        if system_delta > 0:
            self.cpu_samples.append((cpu_delta / system_delta) * num_cpus * 100.0)

        mem_usage = stats["memory_stats"].get("usage", 0)
        self.mem_samples.append(mem_usage / 1024 / 1024)

    def summary(self) -> dict:
        if not self.cpu_samples:
            return {}
        return {
            "cpu_avg_pct": round(sum(self.cpu_samples) / len(self.cpu_samples), 2),
            "cpu_max_pct": round(max(self.cpu_samples), 2),
            "mem_avg_mb": round(sum(self.mem_samples) / len(self.mem_samples), 2),
            "mem_max_mb": round(max(self.mem_samples), 2),
        }


def find_container(service: str):
    if docker_client is None:
        return None
    containers = docker_client.containers.list(
        filters={"label": f"com.docker.compose.service={service}"}
    )
    return containers[0] if containers else None


async def collect_resources(collector: ResourceCollector, container, stop: asyncio.Event):
    while not stop.is_set():
        try:
            collector.sample(container)
        except Exception as e:
            logger.debug("Stats sample failed: %s", e)
        await asyncio.sleep(1)


def parse_locust_csv(csv_prefix: str, endpoint: str) -> dict:
    stats_file = Path(f"{csv_prefix}_stats.csv")
    if not stats_file.exists():
        return {}

    with open(stats_file) as f:
        for row in csv.DictReader(f):
            if row["Name"] == endpoint:
                return {
                    "throughput_rps": float(row["Requests/s"]),
                    "latency_p50_ms": float(row["50%"]),
                    "latency_p95_ms": float(row["95%"]),
                    "latency_p99_ms": float(row["99%"]),
                    "total_requests": int(row["Request Count"]),
                    "failures": int(row["Failure Count"]),
                }
    return {}


@app.post("/benchmark")
async def run_benchmark(req: BenchmarkRequest):
    logger.info(
        "Starting benchmark: users=%d, spawn_rate=%d, duration=%ds",
        req.users, req.spawn_rate, req.duration,
    )

    container = find_container(req.target_service)
    if container is None:
        logger.warning("Container for service '%s' not found — skipping resource stats", req.target_service)

    collector = ResourceCollector()
    stop_event = asyncio.Event()

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_prefix = f"{tmpdir}/results"

        locust_proc = await asyncio.create_subprocess_exec(
            "locust", "-f", "locustfile.py",
            "--headless",
            "-u", str(req.users),
            "-r", str(req.spawn_rate),
            "--run-time", f"{req.duration}s",
            "--csv", csv_prefix,
            "--host", "http://app:8000",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "EMBED_ENDPOINT": req.endpoint},
        )

        resource_task = None
        if container:
            resource_task = asyncio.create_task(
                collect_resources(collector, container, stop_event)
            )

        await locust_proc.wait()

        stop_event.set()
        if resource_task:
            await resource_task

        perf = parse_locust_csv(csv_prefix, req.endpoint)

    if not perf:
        raise HTTPException(status_code=500, detail="Failed to parse locust results")

    result = {
        "config": {
            "users": req.users,
            "spawn_rate": req.spawn_rate,
            "duration_s": req.duration,
        },
        "performance": perf,
        "resources": collector.summary(),
    }

    logger.info("Benchmark complete: %s", result)
    return result
