from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass

import psutil


@dataclass
class HardwareSnapshot:
    ram_total_bytes: int
    ram_available_bytes: int
    ram_used_bytes: int
    cpu_percent: float
    gpu_power_mw: float | None
    gpu_utilization_pct: float | None


def _parse_powermetrics_gpu(output: str) -> tuple[float | None, float | None]:
    power_match = re.search(r"GPU Power:\s*(\d+(?:\.\d+)?)\s*mW", output)
    util_match = re.search(r"GPU HW active residency:\s*(\d+(?:\.\d+)?)%", output)
    power = float(power_match.group(1)) if power_match else None
    util = float(util_match.group(1)) if util_match else None
    return power, util


def _capture_gpu_powermetrics() -> tuple[float | None, float | None]:
    try:
        proc = subprocess.run(
            ["sudo", "powermetrics", "--samplers", "gpu_power", "-i", "1000", "-n", "1"],
            capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None, None
    if proc.returncode != 0:
        return None, None
    return _parse_powermetrics_gpu(proc.stdout)


def capture_hardware_snapshot(skip_gpu: bool = False) -> HardwareSnapshot:
    vm = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.5)
    gpu_power, gpu_util = (None, None)
    if not skip_gpu and sys.platform == "darwin":
        gpu_power, gpu_util = _capture_gpu_powermetrics()
    return HardwareSnapshot(
        ram_total_bytes=vm.total,
        ram_available_bytes=vm.available,
        ram_used_bytes=vm.used,
        cpu_percent=cpu_percent,
        gpu_power_mw=gpu_power,
        gpu_utilization_pct=gpu_util,
    )


def estimate_load_bandwidth_gbps(model_size_bytes: int, time_to_first_token_s: float) -> float | None:
    """Estimated load bandwidth — a derived figure, NOT a verified
    measurement. True memory bandwidth needs Instruments-level tooling this
    project doesn't build. Label this as an estimate everywhere it's shown."""
    if time_to_first_token_s <= 0:
        return None
    return (model_size_bytes / 1024**3) / time_to_first_token_s
