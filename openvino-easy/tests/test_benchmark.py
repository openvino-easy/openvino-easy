"""Unit tests for oe.benchmark."""

import numpy as np
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path
from oe.benchmark import (
    benchmark_model,
    _generate_dummy_input,
    _calculate_percentiles,
    save_benchmark_results,
    load_benchmark_results,
    compare_benchmarks,
)
import importlib
benchmark_module = importlib.import_module('oe.benchmark')


def test_generate_dummy_input():
    """Test dummy input generation."""
    # Mock compiled model with input nodes
    mock_model = MagicMock()
    mock_input1 = MagicMock()
    mock_input1.shape = [1, 3, 224, 224]
    mock_input1.get_any_name.return_value = "input1"

    mock_input2 = MagicMock()
    mock_input2.shape = [1, 10]
    mock_input2.get_any_name.return_value = "input2"

    mock_model.inputs = [mock_input1, mock_input2]

    input_data = _generate_dummy_input(mock_model, batch_size=2)

    assert "input1" in input_data
    assert "input2" in input_data
    assert input_data["input1"].shape == (2, 3, 224, 224)
    assert input_data["input2"].shape == (2, 10)
    assert input_data["input1"].dtype == np.float32
    assert input_data["input2"].dtype == np.float32


def test_generate_dummy_input_dynamic_shapes():
    """Test dummy input generation with dynamic shapes."""
    mock_model = MagicMock()
    mock_input = MagicMock()
    mock_input.shape = [-1, 3, 224, 224]  # Dynamic batch size
    mock_input.get_any_name.return_value = "input"
    mock_model.inputs = [mock_input]

    input_data = _generate_dummy_input(mock_model, batch_size=4)

    assert input_data["input"].shape == (4, 3, 224, 224)


def test_calculate_percentiles():
    """Test percentile calculation."""
    # Create test data
    times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    percentiles = _calculate_percentiles(times)

    assert "p50_ms" in percentiles
    assert "p90_ms" in percentiles
    assert "p95_ms" in percentiles
    assert "p99_ms" in percentiles

    # Check that p50 is median
    assert percentiles["p50_ms"] == 5.5
    # Check that p90 is approximately 9.1 (numpy percentile interpolation)
    assert abs(percentiles["p90_ms"] - 9.1) < 0.1


@patch.object(benchmark_module, '_generate_dummy_input')
@patch("time.perf_counter_ns")
def test_benchmark_model(mock_time, mock_generate_input):
    """Test model benchmarking."""
    # Mock compiled model
    mock_model = MagicMock()
    mock_model.inputs = [MagicMock()]

    # Mock input data
    mock_input_data = {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}
    mock_generate_input.return_value = mock_input_data

    # Mock timing (need start and end times for each run: 2 calls per run * 5 runs = 10 calls)
    mock_time.side_effect = [
        0,
        1000000,
        0,
        2000000,
        0,
        1000000,
        0,
        3000000,
        0,
        1000000,
    ]  # pairs: (start, end)

    results = benchmark_model(
        mock_model, warmup_runs=0, benchmark_runs=5, device_name="CPU"
    )

    # Check that infer request was created and called
    assert mock_model.create_infer_request.call_count == 1

    # Check results structure
    assert "device" in results
    assert "mean_ms" in results
    assert "fps" in results
    assert "p50_ms" in results
    assert "p90_ms" in results

    assert results["device"] == "CPU"
    assert results["warmup_runs"] == 0
    assert results["benchmark_runs"] == 5
    assert results["batch_size"] == 1


# Note: benchmark_pipeline function removed in favor of stateful oe.benchmark()


def test_save_and_load_benchmark_results():
    """Test saving and loading benchmark results."""
    # Test data
    test_results = {
        "device": "CPU",
        "mean_ms": 5.0,
        "fps": 200.0,
        "p50_ms": 4.8,
        "p90_ms": 5.2,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        # Save results
        save_benchmark_results(test_results, temp_path)

        # Load results
        loaded_results = load_benchmark_results(temp_path)

        # Check that data is preserved
        assert loaded_results == test_results

    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)


def test_compare_benchmarks():
    """Test benchmark comparison."""
    # Test data
    results1 = {"device": "CPU", "mean_ms": 10.0, "fps": 100.0, "p90_ms": 12.0}
    results2 = {"device": "NPU", "mean_ms": 5.0, "fps": 200.0, "p90_ms": 6.0}
    results3 = {"device": "GPU", "mean_ms": 8.0, "fps": 125.0, "p90_ms": 9.0}

    comparison = compare_benchmarks(
        [results1, results2, results3], ["model1", "model2", "model3"]
    )

    assert "models" in comparison
    assert "devices" in comparison
    assert "mean_latency_ms" in comparison
    assert "fps" in comparison
    assert "p90_latency_ms" in comparison
    assert "relative_fps" in comparison

    assert comparison["models"] == ["model1", "model2", "model3"]
    assert comparison["devices"] == ["CPU", "NPU", "GPU"]
    assert comparison["mean_latency_ms"] == [10.0, 5.0, 8.0]
    assert comparison["fps"] == [100.0, 200.0, 125.0]
    assert comparison["p90_latency_ms"] == [12.0, 6.0, 9.0]

    # Check relative FPS (NPU should be 1.0 as it's the fastest)
    assert comparison["relative_fps"] == [0.5, 1.0, 0.625]


def test_compare_benchmarks_empty():
    """Test benchmark comparison with empty list."""
    comparison = compare_benchmarks([])
    assert comparison == {}


def test_compare_benchmarks_single():
    """Test benchmark comparison with single result."""
    results = {"device": "CPU", "mean_ms": 10.0, "fps": 100.0, "p90_ms": 12.0}

    comparison = compare_benchmarks([results])

    assert "models" in comparison
    assert "fps" in comparison
    assert "relative_fps" not in comparison  # Should not be present for single result


@patch.object(benchmark_module, '_generate_dummy_input')
@patch("time.perf_counter_ns")
def test_benchmark_model_fps_calculation(mock_time, mock_generate_input):
    """Test FPS calculation in benchmarking."""
    mock_model = MagicMock()
    mock_model.inputs = [MagicMock()]
    mock_generate_input.return_value = {
        "input": np.random.randn(1, 3, 224, 224).astype(np.float32)
    }

    # Mock timing for 10ms average latency
    mock_time.side_effect = [0, 10000000, 0, 10000000, 0, 10000000]  # 10ms per run

    results = benchmark_model(mock_model, warmup_runs=0, benchmark_runs=3)

    # FPS should be 1000/10 = 100
    assert results["mean_ms"] == 10.0
    assert results["fps"] == 100.0


@patch.object(benchmark_module, '_generate_dummy_input')
@patch("time.perf_counter_ns")
def test_benchmark_model_zero_latency(mock_time, mock_generate_input):
    """Test FPS calculation with zero latency."""
    mock_model = MagicMock()
    mock_model.inputs = [MagicMock()]
    mock_generate_input.return_value = {
        "input": np.random.randn(1, 3, 224, 224).astype(np.float32)
    }

    # Mock timing for 0ms latency
    mock_time.side_effect = [0, 0, 0, 0, 0, 0]

    results = benchmark_model(mock_model, warmup_runs=0, benchmark_runs=3)

    # FPS should be 0 when latency is 0
    assert results["mean_ms"] == 0.0
    assert results["fps"] == 0
