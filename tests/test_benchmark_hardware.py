from llm_ladder.benchmark_hardware import (
    capture_hardware_snapshot, _parse_powermetrics_gpu, estimate_load_bandwidth_gbps,
)


class _FakeVM:
    total = 1000
    available = 400
    used = 600


def test_capture_hardware_snapshot_skip_gpu(monkeypatch):
    monkeypatch.setattr("llm_ladder.benchmark_hardware.psutil.virtual_memory", lambda: _FakeVM())
    monkeypatch.setattr("llm_ladder.benchmark_hardware.psutil.cpu_percent", lambda interval=0.5: 42.0)
    snap = capture_hardware_snapshot(skip_gpu=True)
    assert snap.ram_total_bytes == 1000
    assert snap.ram_available_bytes == 400
    assert snap.cpu_percent == 42.0
    assert snap.gpu_power_mw is None
    assert snap.gpu_utilization_pct is None

def test_parse_powermetrics_gpu_extracts_values():
    sample = (
        "**** GPU usage ****\n\n"
        "GPU HW active frequency: 444 MHz\n"
        "GPU HW active residency: 20.50%\n"
        "GPU idle residency: 79.50%\n"
        "GPU Power: 1234 mW\n"
    )
    power, util = _parse_powermetrics_gpu(sample)
    assert power == 1234.0
    assert util == 20.50

def test_parse_powermetrics_gpu_missing_fields_returns_none():
    power, util = _parse_powermetrics_gpu("no gpu data here")
    assert power is None
    assert util is None

def test_estimate_load_bandwidth():
    result = estimate_load_bandwidth_gbps(model_size_bytes=10 * 1024**3, time_to_first_token_s=2.0)
    assert abs(result - 5.0) < 1e-6

def test_estimate_load_bandwidth_zero_time_returns_none():
    assert estimate_load_bandwidth_gbps(1000, 0) is None

def test_estimate_load_bandwidth_negative_time_returns_none():
    assert estimate_load_bandwidth_gbps(1000, -1) is None
