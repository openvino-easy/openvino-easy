"""Benchmarking utilities for OpenVINO-Easy (latency, FPS, etc)."""

import time
import numpy as np
import json
from typing import Dict, Any, Optional
from pathlib import Path

def _generate_dummy_input(compiled_model, batch_size=1):
    """Generate dummy input data for benchmarking."""
    input_data = {}
    for input_node in compiled_model.inputs:
        shape = list(input_node.shape)
        # Replace dynamic dimensions with batch_size
        shape = [batch_size if dim == -1 else dim for dim in shape]
        # Generate random data
        input_data[input_node.get_any_name()] = np.random.randn(*shape).astype(np.float32)
    return input_data

def _calculate_percentiles(times, percentiles=[50, 90, 95, 99]):
    """Calculate percentiles from timing data."""
    results = {}
    for p in percentiles:
        results[f"p{p}_ms"] = round(np.percentile(times, p), 2)
    return results

def benchmark_model(compiled_model, 
                   warmup_runs=5, 
                   benchmark_runs=20, 
                   batch_size=1,
                   device_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Benchmark a compiled OpenVINO model.
    
    Args:
        compiled_model: Compiled OpenVINO model
        warmup_runs: Number of warmup runs to discard
        benchmark_runs: Number of benchmark runs to measure
        batch_size: Batch size for inference
        device_name: Device name for reporting (optional)
    
    Returns:
        Dictionary with benchmark results
    """
    # Generate dummy input data
    input_data = _generate_dummy_input(compiled_model, batch_size)
    
    # Create infer request (OpenVINO 2024 API)
    infer_request = compiled_model.create_infer_request()
    
    # Warmup runs
    for _ in range(warmup_runs):
        infer_request.infer(input_data)
    
    # Benchmark runs
    times = []
    for _ in range(benchmark_runs):
        start_time = time.perf_counter_ns()
        infer_request.infer(input_data)
        end_time = time.perf_counter_ns()
        times.append((end_time - start_time) / 1_000_000)  # Convert to milliseconds
    
    # Calculate statistics
    mean_ms = round(np.mean(times), 2)
    std_ms = round(np.std(times), 2)
    min_ms = round(np.min(times), 2)
    max_ms = round(np.max(times), 2)
    
    # Calculate percentiles
    percentiles = _calculate_percentiles(times)
    
    # Calculate FPS
    fps = round(1000 / mean_ms, 1) if mean_ms > 0 else 0
    
    # Prepare results
    results = {
        "device": device_name or "unknown",
        "batch_size": batch_size,
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "fps": fps,
        **percentiles
    }
    
    return results

def benchmark_pipeline(pipeline, 
                      warmup_runs=5, 
                      benchmark_runs=20,
                      batch_size=1) -> Dict[str, Any]:
    """
    Benchmark an OpenVINO-Easy pipeline.
    
    Args:
        pipeline: OpenVINO-Easy Pipeline object
        warmup_runs: Number of warmup runs to discard
        benchmark_runs: Number of benchmark runs to measure
        batch_size: Batch size for inference
    
    Returns:
        Dictionary with benchmark results
    """
    return benchmark_model(
        pipeline.compiled_model,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        batch_size=batch_size,
        device_name=pipeline.device
    )

def save_benchmark_results(results: Dict[str, Any], output_path: str | Path):
    """
    Save benchmark results to a JSON file.
    
    Args:
        results: Benchmark results dictionary
        output_path: Path to save the JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def load_benchmark_results(input_path: str | Path) -> Dict[str, Any]:
    """
    Load benchmark results from a JSON file.
    
    Args:
        input_path: Path to the JSON file
    
    Returns:
        Benchmark results dictionary
    """
    with open(input_path, 'r') as f:
        return json.load(f)

def compare_benchmarks(results_list: list[Dict[str, Any]], 
                      model_names: Optional[list[str]] = None) -> Dict[str, Any]:
    """
    Compare multiple benchmark results.
    
    Args:
        results_list: List of benchmark result dictionaries
        model_names: Optional list of model names for comparison
    
    Returns:
        Comparison results dictionary
    """
    if not results_list:
        return {}
    
    comparison = {
        "models": model_names or [f"model_{i}" for i in range(len(results_list))],
        "devices": [r.get("device", "unknown") for r in results_list],
        "mean_latency_ms": [r.get("mean_ms", 0) for r in results_list],
        "fps": [r.get("fps", 0) for r in results_list],
        "p90_latency_ms": [r.get("p90_ms", 0) for r in results_list]
    }
    
    # Calculate relative performance
    if len(results_list) > 1:
        best_fps = max(comparison["fps"])
        comparison["relative_fps"] = [fps / best_fps if best_fps > 0 else 0 for fps in comparison["fps"]]
    
    return comparison 