"""Benchmarking utilities for OpenVINO-Easy (latency, FPS, etc)."""

import time
import numpy as np
import json
import csv
import concurrent.futures
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking runs."""

    warmup_runs: int = 5
    benchmark_runs: int = 20
    batch_size: int = 1
    max_threads: int = 4
    use_real_inputs: bool = False
    input_data: Optional[List[Any]] = None
    sequence_lengths: Optional[List[int]] = None  # For text models
    image_sizes: Optional[List[tuple]] = None  # For vision models
    measure_memory: bool = False
    measure_cpu_usage: bool = False


@dataclass
class BenchmarkResult:
    """Structured benchmark results."""

    device: str
    model_name: str = "unknown"
    batch_size: int = 1
    warmup_runs: int = 5
    benchmark_runs: int = 20
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    fps: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    throughput_fps: float = 0.0  # For batch processing
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    input_type: str = "dummy"
    config: Optional[BenchmarkConfig] = None


def _generate_dummy_input(compiled_model, batch_size=1):
    """Generate dummy input data for benchmarking."""
    input_data = {}
    for input_node in compiled_model.inputs:
        shape = list(input_node.shape)
        # Replace dynamic dimensions with batch_size, also replace first dimension if it's 1
        shape = [
            batch_size if (dim == -1 or (i == 0 and dim == 1)) else dim
            for i, dim in enumerate(shape)
        ]
        # Generate random data
        input_data[input_node.get_any_name()] = np.random.randn(*shape).astype(
            np.float32
        )
    return input_data


def _generate_realistic_inputs(
    pipeline, config: BenchmarkConfig
) -> List[Dict[str, np.ndarray]]:
    """Generate realistic input data based on model type."""
    inputs = []

    # Check if it's a text model
    if (
        hasattr(pipeline.runtime, "tokenizer")
        and pipeline.runtime.tokenizer is not None
    ):
        # Generate diverse text inputs
        text_samples = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning enables computers to process natural language efficiently.",
            "OpenVINO optimizes deep learning models for Intel hardware acceleration.",
            "Artificial intelligence is transforming industries across the globe.",
            "Edge computing brings inference closer to data sources for reduced latency.",
        ]

        # Add variable length sequences if specified
        if config.sequence_lengths:
            extended_samples = []
            for length in config.sequence_lengths:
                # Create text of approximately the desired length
                words_needed = length // 5  # Rough approximation
                text = " ".join(text_samples[0].split()[:words_needed])
                extended_samples.append(text)
            text_samples.extend(extended_samples)

        for text in text_samples[: config.benchmark_runs]:
            try:
                processed = pipeline.runtime._preprocess_input(text)
                inputs.append(processed)
            except:
                # Fallback to dummy input
                inputs.append(
                    _generate_dummy_input(pipeline.compiled_model, config.batch_size)
                )

    # Check if it's a vision model
    elif _is_vision_model(pipeline):
        # Generate diverse image inputs
        for i in range(config.benchmark_runs):
            if config.image_sizes:
                # Use specified image sizes
                size_idx = i % len(config.image_sizes)
                h, w = config.image_sizes[size_idx]
                img_data = np.random.rand(config.batch_size, 3, h, w).astype(np.float32)
            else:
                # Standard ImageNet size
                img_data = np.random.rand(config.batch_size, 3, 224, 224).astype(
                    np.float32
                )

            # Convert to proper input format
            input_name = list(pipeline.runtime.input_info.keys())[0]
            inputs.append({input_name: img_data})

    else:
        # Generic model - use dummy inputs
        for _ in range(config.benchmark_runs):
            inputs.append(
                _generate_dummy_input(pipeline.compiled_model, config.batch_size)
            )

    return inputs


def _is_vision_model(pipeline) -> bool:
    """Check if pipeline is a vision model."""
    for input_info in pipeline.runtime.input_info.values():
        shape = input_info["shape"]
        # Vision models typically have 4D inputs [B, C, H, W]
        if len(shape) == 4 and len([d for d in shape if d > 0 and d > 100]) >= 2:
            return True
    return False


def _calculate_percentiles(times, percentiles=[50, 90, 95, 99]):
    """Calculate percentiles from timing data."""
    results = {}
    for p in percentiles:
        results[f"p{p}_ms"] = float(np.percentile(times, p))
    return results


def _measure_system_resources(func, *args, **kwargs):
    """Measure memory and CPU usage during function execution."""
    try:
        import psutil

        process = psutil.Process()

        # Measure before
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        cpu_before = process.cpu_percent()

        # Execute function
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time

        # Measure after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        cpu_after = process.cpu_percent()

        return result, {
            "memory_usage_mb": memory_after - memory_before,
            "cpu_usage_percent": (cpu_after - cpu_before) / duration
            if duration > 0
            else 0,
        }
    except ImportError:
        # psutil not available
        return func(*args, **kwargs), {
            "memory_usage_mb": None,
            "cpu_usage_percent": None,
        }


def benchmark_model_enhanced(
    compiled_model,
    config: BenchmarkConfig,
    device_name: Optional[str] = None,
    model_name: str = "unknown",
) -> BenchmarkResult:
    """
    Enhanced benchmark with configurable options and real inputs.

    Args:
        compiled_model: Compiled OpenVINO model
        config: Benchmark configuration
        device_name: Device name for reporting
        model_name: Model name for reporting

    Returns:
        Structured benchmark results
    """
    # Prepare input data
    if config.input_data:
        input_data_list = config.input_data[: config.benchmark_runs]
    else:
        # Generate dummy inputs
        input_data_list = [
            _generate_dummy_input(compiled_model, config.batch_size)
            for _ in range(config.benchmark_runs)
        ]

    # Create infer request (OpenVINO 2025 API)
    infer_request = compiled_model.create_infer_request()

    # Warmup runs
    for i in range(config.warmup_runs):
        idx = i % len(input_data_list)
        infer_request.infer(input_data_list[idx])

    # Benchmark function
    def run_inference():
        times = []
        for i in range(config.benchmark_runs):
            idx = i % len(input_data_list)
            start_time = time.perf_counter_ns()
            infer_request.infer(input_data_list[idx])
            end_time = time.perf_counter_ns()
            times.append((end_time - start_time) / 1_000_000)  # Convert to milliseconds
        return times

    # Run benchmark with optional resource monitoring
    if config.measure_memory or config.measure_cpu_usage:
        times, resources = _measure_system_resources(run_inference)
    else:
        times = run_inference()
        resources = {"memory_usage_mb": None, "cpu_usage_percent": None}

    # Calculate statistics
    mean_ms = round(np.mean(times), 2)
    std_ms = round(np.std(times), 2)
    min_ms = round(np.min(times), 2)
    max_ms = round(np.max(times), 2)

    # Calculate percentiles
    percentiles = _calculate_percentiles(times)

    # Calculate FPS and throughput
    fps = round(1000 / mean_ms, 1) if mean_ms > 0 else 0
    throughput_fps = (
        round((config.batch_size * 1000) / mean_ms, 1) if mean_ms > 0 else 0
    )

    # Create result
    result = BenchmarkResult(
        device=device_name or "unknown",
        model_name=model_name,
        batch_size=config.batch_size,
        warmup_runs=config.warmup_runs,
        benchmark_runs=config.benchmark_runs,
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        fps=fps,
        throughput_fps=throughput_fps,
        p50_ms=percentiles["p50_ms"],
        p90_ms=percentiles["p90_ms"],
        p95_ms=percentiles["p95_ms"],
        p99_ms=percentiles["p99_ms"],
        memory_usage_mb=resources.get("memory_usage_mb"),
        cpu_usage_percent=resources.get("cpu_usage_percent"),
        input_type="real" if config.use_real_inputs else "dummy",
        config=config,
    )

    # Add performance comparison with NPU generation expectations
    if device_name == "NPU":
        # Note: pipeline parameter would be needed for NPU analysis
        # _add_npu_performance_analysis(result, pipeline)
        pass

    return result


def benchmark_model_multithreaded(
    compiled_model,
    config: BenchmarkConfig,
    device_name: Optional[str] = None,
    model_name: str = "unknown",
) -> BenchmarkResult:
    """
    Multi-threaded benchmark for testing concurrent inference performance.

    Args:
        compiled_model: Compiled OpenVINO model
        config: Benchmark configuration
        device_name: Device name for reporting
        model_name: Model name for reporting

    Returns:
        Structured benchmark results
    """
    # Create multiple infer requests for threading
    infer_requests = [
        compiled_model.create_infer_request() for _ in range(config.max_threads)
    ]

    # Prepare input data
    if config.input_data:
        input_data_list = config.input_data
    else:
        input_data_list = [
            _generate_dummy_input(compiled_model, config.batch_size)
            for _ in range(config.benchmark_runs)
        ]

    # Warmup
    for i in range(config.warmup_runs):
        request = infer_requests[i % len(infer_requests)]
        data = input_data_list[i % len(input_data_list)]
        request.infer(data)

    # Benchmark with threading
    times = []
    start_total = time.perf_counter()

    def worker(request_idx, data_indices):
        """Worker function for each thread."""
        request = infer_requests[request_idx]
        thread_times = []

        for data_idx in data_indices:
            data = input_data_list[data_idx % len(input_data_list)]
            start_time = time.perf_counter_ns()
            request.infer(data)
            end_time = time.perf_counter_ns()
            thread_times.append((end_time - start_time) / 1_000_000)

        return thread_times

    # Distribute work across threads
    work_per_thread = config.benchmark_runs // config.max_threads
    remainder = config.benchmark_runs % config.max_threads

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.max_threads
    ) as executor:
        futures = []
        start_idx = 0

        for i in range(config.max_threads):
            # Calculate work for this thread
            thread_work = work_per_thread + (1 if i < remainder else 0)
            data_indices = list(range(start_idx, start_idx + thread_work))
            start_idx += thread_work

            # Submit work
            future = executor.submit(worker, i, data_indices)
            futures.append(future)

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            thread_times = future.result()
            times.extend(thread_times)

    end_total = time.perf_counter()
    total_time_s = end_total - start_total

    # Calculate statistics
    mean_ms = round(np.mean(times), 2)
    std_ms = round(np.std(times), 2)
    min_ms = round(np.min(times), 2)
    max_ms = round(np.max(times), 2)

    # Calculate percentiles
    percentiles = _calculate_percentiles(times)

    # Calculate FPS and total throughput
    fps = round(1000 / mean_ms, 1) if mean_ms > 0 else 0
    total_throughput = round(config.benchmark_runs / total_time_s, 1)

    # Create result
    result = BenchmarkResult(
        device=device_name or "unknown",
        model_name=model_name,
        batch_size=config.batch_size,
        warmup_runs=config.warmup_runs,
        benchmark_runs=config.benchmark_runs,
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        fps=fps,
        throughput_fps=total_throughput,
        p50_ms=percentiles["p50_ms"],
        p90_ms=percentiles["p90_ms"],
        p95_ms=percentiles["p95_ms"],
        p99_ms=percentiles["p99_ms"],
        input_type="real" if config.use_real_inputs else "dummy",
        config=config,
    )

    return result


# Legacy functions for backward compatibility
def benchmark_model(
    compiled_model,
    warmup_runs=5,
    benchmark_runs=20,
    batch_size=1,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Legacy benchmark function for backward compatibility.
    """
    # Prepare input data
    input_data = _generate_dummy_input(compiled_model, batch_size)

    # Create infer request
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
    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    min_ms = float(np.min(times))
    max_ms = float(np.max(times))

    # Calculate percentiles
    percentiles = _calculate_percentiles(times)

    # Calculate FPS
    fps = 1000 / mean_ms if mean_ms > 0 else 0

    return {
        "device": device_name or "unknown",
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
        "batch_size": batch_size,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "fps": fps,
        **percentiles,
    }


def benchmark_pipeline(
    pipeline,
    warmup_runs=5,
    benchmark_runs=20,
    batch_size=1,
    use_real_inputs=False,
    max_threads=1,
) -> Dict[str, Any]:
    """
    Enhanced pipeline benchmark with real inputs and threading options.

    Args:
        pipeline: OpenVINO-Easy Pipeline object
        warmup_runs: Number of warmup runs to discard
        benchmark_runs: Number of benchmark runs to measure
        batch_size: Batch size for inference
        use_real_inputs: Whether to use realistic inputs instead of dummy data
        max_threads: Number of threads for concurrent inference (1 = single-threaded)

    Returns:
        Dictionary with benchmark results
    """
    return benchmark_model(
        pipeline.compiled_model,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        batch_size=batch_size,
        device_name=pipeline.device,
    )


def save_benchmark_results(
    results: Union[Dict[str, Any], BenchmarkResult],
    output_path: Union[str, Path],
    format: str = "json",
):
    """
    Save benchmark results to file in various formats.

    Args:
        results: Benchmark results dictionary or BenchmarkResult object
        output_path: Path to save the file
        format: Output format ("json", "csv")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict if needed
    if isinstance(results, BenchmarkResult):
        results_dict = asdict(results)
    else:
        results_dict = results

    if format.lower() == "json":
        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)

    elif format.lower() == "csv":
        # Flatten nested config if present
        flattened = {}
        for key, value in results_dict.items():
            if key == "config" and isinstance(value, dict):
                for config_key, config_value in value.items():
                    flattened[f"config_{config_key}"] = config_value
            else:
                flattened[key] = value

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flattened.keys())
            writer.writeheader()
            writer.writerow(flattened)

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'")


def _add_npu_performance_analysis(result: BenchmarkResult, pipeline) -> None:
    """
    Add NPU generation-specific performance analysis to benchmark results.

    Args:
        result: Benchmark result to enhance
        pipeline: Pipeline object with NPU device info
    """
    try:
        from ._core import check_npu_driver

        npu_status = check_npu_driver()
        if not npu_status.get("npu_functional"):
            return

        npu_generation = npu_status.get("npu_generation", "unknown")
        capabilities = npu_status.get("capabilities", {})

        # Add NPU-specific metadata to result
        if not hasattr(result, "npu_analysis"):
            result.npu_analysis = {}

        result.npu_analysis = {
            "npu_generation": npu_generation,
            "generation_number": capabilities.get("generation", "unknown"),
            "expected_performance": {},
            "performance_comparison": {},
            "recommendations": [],
        }

        # Model-specific performance expectations
        model_type = _detect_benchmark_model_type(pipeline)

        if model_type == "stable_diffusion":
            expected_fps = capabilities.get("expected_stable_diffusion_fps", 1.0)
            actual_fps = result.fps

            result.npu_analysis["expected_performance"]["stable_diffusion_fps"] = (
                expected_fps
            )
            result.npu_analysis["performance_comparison"]["vs_expected_fps"] = {
                "expected": expected_fps,
                "actual": actual_fps,
                "ratio": round(actual_fps / expected_fps, 2) if expected_fps > 0 else 0,
                "performance_level": _categorize_performance_vs_expected(
                    actual_fps, expected_fps
                ),
            }

            # Generation-specific recommendations
            if npu_generation in ["lunar_lake", "arrow_lake"]:
                if actual_fps < expected_fps * 0.8:  # Less than 80% of expected
                    result.npu_analysis["recommendations"].extend(
                        [
                            "Consider using FP16-NF4 precision for better performance",
                            "Ensure latest Intel NPU drivers are installed",
                            "Check if other applications are using NPU resources",
                        ]
                    )
                elif actual_fps >= expected_fps:
                    result.npu_analysis["recommendations"].append(
                        f"Excellent performance! {npu_generation} NPU is performing at or above expectations"
                    )

        elif model_type == "text_generation":
            expected_tps = capabilities.get("expected_dialog_gpt_tps", 25)
            # Convert FPS to tokens per second (rough approximation)
            actual_tps = result.fps * 10  # Assume ~10 tokens per inference

            result.npu_analysis["expected_performance"]["text_generation_tps"] = (
                expected_tps
            )
            result.npu_analysis["performance_comparison"]["vs_expected_tps"] = {
                "expected": expected_tps,
                "actual": actual_tps,
                "ratio": round(actual_tps / expected_tps, 2) if expected_tps > 0 else 0,
                "performance_level": _categorize_performance_vs_expected(
                    actual_tps, expected_tps
                ),
            }

        # General NPU generation recommendations
        if npu_generation == "lunar_lake":
            result.npu_analysis["recommendations"].append(
                "Latest generation NPU with FP16-NF4 support - optimal for AI workloads"
            )
        elif npu_generation == "arrow_lake":
            result.npu_analysis["recommendations"].append(
                "High-performance desktop NPU with advanced quantization support"
            )
        elif npu_generation == "meteor_lake":
            result.npu_analysis["recommendations"].append(
                "First-generation NPU - consider upgrading to Arrow/Lunar Lake for better performance"
            )

    except Exception:
        # Don't fail benchmarking if NPU analysis fails
        pass


def _detect_benchmark_model_type(pipeline) -> str:
    """Detect model type for benchmark expectations."""
    try:
        model_info = pipeline.get_info()
        source_path = model_info.get("source_path", "").lower()

        if "stable-diffusion" in source_path or "diffusion" in source_path:
            return "stable_diffusion"
        elif any(
            model in source_path for model in ["gpt", "dialog", "t5", "bert", "text"]
        ):
            return "text_generation"
        elif any(
            model in source_path for model in ["resnet", "efficientnet", "vit", "image"]
        ):
            return "vision_classification"
        else:
            return "unknown"
    except:
        return "unknown"


def _categorize_performance_vs_expected(actual: float, expected: float) -> str:
    """Categorize performance relative to expectations."""
    if expected <= 0:
        return "unknown"

    ratio = actual / expected

    if ratio >= 1.2:
        return "excellent"
    elif ratio >= 1.0:
        return "good"
    elif ratio >= 0.8:
        return "acceptable"
    elif ratio >= 0.6:
        return "below_expected"
    else:
        return "poor"


def load_benchmark_results(input_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load benchmark results from a JSON file.

    Args:
        input_path: Path to the JSON file

    Returns:
        Benchmark results dictionary
    """
    with open(input_path, "r") as f:
        return json.load(f)


def compare_benchmarks(
    results_list: List[Union[Dict[str, Any], BenchmarkResult]],
    model_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare multiple benchmark results with enhanced analysis.

    Args:
        results_list: List of benchmark result dictionaries or BenchmarkResult objects
        model_names: Optional list of model names for comparison

    Returns:
        Comprehensive comparison results dictionary
    """
    if not results_list:
        return {}

    # Convert to dicts if needed
    results_dicts = []
    for result in results_list:
        if isinstance(result, BenchmarkResult):
            results_dicts.append(asdict(result))
        else:
            results_dicts.append(result)

    comparison = {
        "models": model_names
        or [r.get("model_name", f"model_{i}") for i, r in enumerate(results_dicts)],
        "devices": [r.get("device", "unknown") for r in results_dicts],
        "mean_latency_ms": [r.get("mean_ms", 0) for r in results_dicts],
        "fps": [r.get("fps", 0) for r in results_dicts],
        "throughput_fps": [r.get("throughput_fps", 0) for r in results_dicts],
        "p90_latency_ms": [r.get("p90_ms", 0) for r in results_dicts],
        "p99_latency_ms": [r.get("p99_ms", 0) for r in results_dicts],
        "batch_sizes": [r.get("batch_size", 1) for r in results_dicts],
        "input_types": [r.get("input_type", "dummy") for r in results_dicts],
    }

    # Calculate relative performance
    if len(results_dicts) > 1:
        best_fps = max(comparison["fps"])
        best_throughput = max(comparison["throughput_fps"])

        comparison["relative_fps"] = [
            fps / best_fps if best_fps > 0 else 0 for fps in comparison["fps"]
        ]
        comparison["relative_throughput"] = [
            tp / best_throughput if best_throughput > 0 else 0
            for tp in comparison["throughput_fps"]
        ]

        # Performance analysis
        comparison["analysis"] = {
            "fastest_model": comparison["models"][comparison["fps"].index(best_fps)],
            "highest_throughput": comparison["models"][
                comparison["throughput_fps"].index(best_throughput)
            ],
            "most_consistent": comparison["models"][
                comparison["p99_latency_ms"].index(min(comparison["p99_latency_ms"]))
            ],
            "performance_spread": {
                "fps_range": f"{min(comparison['fps']):.1f} - {max(comparison['fps']):.1f}",
                "latency_range": f"{min(comparison['mean_latency_ms']):.1f} - {max(comparison['mean_latency_ms']):.1f} ms",
            },
        }

    return comparison


async def benchmark_pipeline_async(
    pipeline, config: BenchmarkConfig
) -> BenchmarkResult:
    """
    Asynchronous benchmark using pipeline's async inference capabilities.

    Args:
        pipeline: OpenVINO-Easy Pipeline object with async support
        config: Benchmark configuration

    Returns:
        Benchmark results
    """
    # Generate real inputs if requested
    if config.use_real_inputs:
        input_data_list = _generate_realistic_inputs(pipeline, config)
    else:
        input_data_list = [
            _generate_dummy_input(pipeline.compiled_model, config.batch_size)
            for _ in range(config.benchmark_runs)
        ]

    # Convert to format expected by async inference
    text_inputs = []
    for input_dict in input_data_list:
        if len(input_dict) == 1:
            # Single input - extract the data
            text_inputs.append(list(input_dict.values())[0])
        else:
            # Multiple inputs - keep as dict
            text_inputs.append(input_dict)

    # Warmup
    for i in range(config.warmup_runs):
        idx = i % len(text_inputs)
        await pipeline.infer_async(text_inputs[idx])

    # Benchmark runs
    times = []
    for i in range(config.benchmark_runs):
        idx = i % len(text_inputs)
        start_time = time.perf_counter_ns()
        await pipeline.infer_async(text_inputs[idx])
        end_time = time.perf_counter_ns()
        times.append((end_time - start_time) / 1_000_000)  # Convert to milliseconds

    # Calculate statistics (same as sync version)
    mean_ms = round(np.mean(times), 2)
    std_ms = round(np.std(times), 2)
    min_ms = round(np.min(times), 2)
    max_ms = round(np.max(times), 2)

    percentiles = _calculate_percentiles(times)
    fps = round(1000 / mean_ms, 1) if mean_ms > 0 else 0
    throughput_fps = (
        round((config.batch_size * 1000) / mean_ms, 1) if mean_ms > 0 else 0
    )

    return BenchmarkResult(
        device=pipeline.device,
        model_name=getattr(pipeline, "model_path", "unknown"),
        batch_size=config.batch_size,
        warmup_runs=config.warmup_runs,
        benchmark_runs=config.benchmark_runs,
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        fps=fps,
        throughput_fps=throughput_fps,
        p50_ms=percentiles["p50_ms"],
        p90_ms=percentiles["p90_ms"],
        p95_ms=percentiles["p95_ms"],
        p99_ms=percentiles["p99_ms"],
        input_type="real" if config.use_real_inputs else "dummy",
        config=config,
    )
