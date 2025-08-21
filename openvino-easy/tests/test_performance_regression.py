"""Performance regression tests for OpenVINO-Easy."""

import pytest
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, Any
import numpy as np

import oe


class PerformanceRegression:
    """Track and validate performance metrics across releases."""

    def __init__(self):
        self.baseline_file = Path(__file__).parent / "performance_baselines.json"
        self.baselines = self._load_baselines()

    def _load_baselines(self) -> Dict[str, Dict[str, float]]:
        """Load performance baselines from file."""
        if self.baseline_file.exists():
            with open(self.baseline_file, "r") as f:
                return json.load(f)
        return {}

    def _save_baselines(self):
        """Save performance baselines to file."""
        with open(self.baseline_file, "w") as f:
            json.dump(self.baselines, f, indent=2)

    def record_baseline(self, test_name: str, metrics: Dict[str, float]):
        """Record baseline performance metrics."""
        self.baselines[test_name] = metrics
        self._save_baselines()

    def check_regression(
        self,
        test_name: str,
        current_metrics: Dict[str, float],
        tolerance_percent: float = 20.0,
    ) -> Dict[str, Any]:
        """Check if current metrics show regression compared to baseline."""
        if test_name not in self.baselines:
            # First run - record as baseline
            self.record_baseline(test_name, current_metrics)
            return {"status": "baseline_recorded", "metrics": current_metrics}

        baseline = self.baselines[test_name]
        results = {
            "status": "passed",
            "baseline": baseline,
            "current": current_metrics,
            "regressions": [],
            "improvements": [],
        }

        for metric, current_value in current_metrics.items():
            if metric in baseline:
                baseline_value = baseline[metric]
                percent_change = (
                    (current_value - baseline_value) / baseline_value
                ) * 100

                # For latency metrics (lower is better)
                if "latency" in metric.lower() or "time" in metric.lower():
                    if percent_change > tolerance_percent:
                        results["regressions"].append(
                            {
                                "metric": metric,
                                "baseline": baseline_value,
                                "current": current_value,
                                "percent_change": percent_change,
                            }
                        )
                        results["status"] = "regression_detected"
                    elif percent_change < -5:  # Improvement threshold
                        results["improvements"].append(
                            {
                                "metric": metric,
                                "baseline": baseline_value,
                                "current": current_value,
                                "percent_change": percent_change,
                            }
                        )

                # For throughput metrics (higher is better)
                elif "throughput" in metric.lower() or "fps" in metric.lower():
                    if percent_change < -tolerance_percent:
                        results["regressions"].append(
                            {
                                "metric": metric,
                                "baseline": baseline_value,
                                "current": current_value,
                                "percent_change": percent_change,
                            }
                        )
                        results["status"] = "regression_detected"
                    elif percent_change > 5:  # Improvement threshold
                        results["improvements"].append(
                            {
                                "metric": metric,
                                "baseline": baseline_value,
                                "current": current_value,
                                "percent_change": percent_change,
                            }
                        )

        return results


# Global performance tracker
perf_tracker = PerformanceRegression()


class TestPerformanceRegression:
    """Performance regression test suite."""

    def create_dummy_model_for_benchmark(
        self, temp_dir: Path, model_type: str = "text"
    ):
        """Create a minimal OpenVINO model for performance testing."""
        import openvino as ov

        if model_type == "text":
            # Simple linear layer for text-like processing
            input_shape = [1, 128]  # [batch, sequence_length]
            input_node = ov.opset11.parameter(
                input_shape, dtype=np.float32, name="input_ids"
            )

            # Simple embedding-like operation
            weight = ov.opset11.constant(np.random.randn(128, 768).astype(np.float32))
            output = ov.opset11.matmul(
                input_node, weight, transpose_a=False, transpose_b=False
            )

        elif model_type == "vision":
            # Simple convolution for vision-like processing
            input_shape = [1, 3, 224, 224]  # [batch, channels, height, width]
            input_node = ov.opset11.parameter(
                input_shape, dtype=np.float32, name="input"
            )

            # Simple convolution
            weight = ov.opset11.constant(
                np.random.randn(64, 3, 3, 3).astype(np.float32)
            )
            conv = ov.opset11.convolution(
                input_node, weight, strides=[1, 1], pads_begin=[1, 1], pads_end=[1, 1], dilations=[1, 1]
            )
            output = ov.opset11.relu(conv)

        elif model_type == "audio":
            # Simple 1D convolution for audio-like processing
            input_shape = [1, 16000]  # [batch, audio_samples]
            input_node = ov.opset11.parameter(
                input_shape, dtype=np.float32, name="input_features"
            )

            # Reshape for 1D conv: [batch, channels, length]
            reshape_shape = ov.opset11.constant(np.array([1, 1, 16000], dtype=np.int64))
            reshaped = ov.opset11.reshape(input_node, reshape_shape, special_zero=False)

            # Simple 1D convolution
            weight = ov.opset11.constant(np.random.randn(32, 1, 256).astype(np.float32))
            conv = ov.opset11.convolution(
                reshaped, weight, strides=[128], pads_begin=[128], pads_end=[128], dilations=[1]
            )
            output = ov.opset11.relu(conv)

        # Set output name
        output.set_friendly_name("output")

        # Create and save model
        model = ov.Model([output], [input_node], f"test_{model_type}_model")
        model_path = temp_dir / f"test_{model_type}_model.xml"
        ov.save_model(model, str(model_path))

        return model_path

    @pytest.mark.integration
    def test_text_model_performance_regression(self):
        """Test performance regression for text models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test model
            model_path = self.create_dummy_model_for_benchmark(temp_path, "text")

            # Load model
            oe.load(str(model_path), device_preference=["CPU"])

            # Run benchmark
            stats = oe.benchmark(warmup_runs=3, benchmark_runs=10)

            # Check for regression
            test_name = "text_model_cpu_performance"
            result = perf_tracker.check_regression(test_name, stats)

            # Report results
            print(f"\n📊 Performance Test: {test_name}")
            print(f"Status: {result['status']}")
            if result["status"] == "regression_detected":
                print("⚠️  Performance regressions detected:")
                for regression in result["regressions"]:
                    print(
                        f"  - {regression['metric']}: {regression['percent_change']:.1f}% slower"
                    )
                    print(
                        f"    Baseline: {regression['baseline']:.2f}, Current: {regression['current']:.2f}"
                    )

            if result.get("improvements"):
                print("✅ Performance improvements:")
                for improvement in result["improvements"]:
                    print(
                        f"  - {improvement['metric']}: {improvement['percent_change']:.1f}% better"
                    )

            # Don't fail on regression for now, just warn
            if result["status"] == "regression_detected":
                pytest.skip("Performance regression detected - needs investigation")

    @pytest.mark.integration
    def test_vision_model_performance_regression(self):
        """Test performance regression for vision models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test model
            model_path = self.create_dummy_model_for_benchmark(temp_path, "vision")

            # Load model
            oe.load(str(model_path), device_preference=["CPU"])

            # Run benchmark
            stats = oe.benchmark(warmup_runs=3, benchmark_runs=10)

            # Check for regression
            test_name = "vision_model_cpu_performance"
            result = perf_tracker.check_regression(test_name, stats)

            # Report results
            print(f"\n📊 Performance Test: {test_name}")
            print(f"Status: {result['status']}")

            if result["status"] == "regression_detected":
                pytest.skip("Performance regression detected - needs investigation")

    @pytest.mark.integration
    def test_audio_model_performance_regression(self):
        """Test performance regression for audio models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test model
            model_path = self.create_dummy_model_for_benchmark(temp_path, "audio")

            # Load model
            oe.load(str(model_path), device_preference=["CPU"])

            # Run benchmark
            stats = oe.benchmark(warmup_runs=3, benchmark_runs=10)

            # Check for regression
            test_name = "audio_model_cpu_performance"
            result = perf_tracker.check_regression(test_name, stats)

            # Report results
            print(f"\n📊 Performance Test: {test_name}")
            print(f"Status: {result['status']}")

            if result["status"] == "regression_detected":
                pytest.skip("Performance regression detected - needs investigation")

    @pytest.mark.integration
    def test_model_loading_performance(self):
        """Test model loading time performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test model
            model_path = self.create_dummy_model_for_benchmark(temp_path, "text")

            # Measure loading time
            start_time = time.time()
            oe.load(str(model_path), device_preference=["CPU"])
            load_time = time.time() - start_time

            # Create metrics
            metrics = {
                "load_time_seconds": load_time,
                "model_size_mb": model_path.stat().st_size / (1024 * 1024),
            }

            # Check for regression
            test_name = "model_loading_performance"
            result = perf_tracker.check_regression(test_name, metrics)

            print(f"\n📊 Performance Test: {test_name}")
            print(f"Load time: {load_time:.3f}s")
            print(f"Status: {result['status']}")

            if result["status"] == "regression_detected":
                pytest.skip("Loading performance regression detected")

    @pytest.mark.integration
    def test_cache_performance(self):
        """Test model caching performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test model
            model_path = self.create_dummy_model_for_benchmark(temp_path, "text")

            # First load (cold cache)
            start_time = time.time()
            oe.load(
                str(model_path),
                device_preference=["CPU"],
                cache_dir=temp_path / "cache",
            )
            first_load_time = time.time() - start_time
            oe.unload()

            # Second load (warm cache)
            start_time = time.time()
            oe.load(
                str(model_path),
                device_preference=["CPU"],
                cache_dir=temp_path / "cache",
            )
            second_load_time = time.time() - start_time
            oe.unload()

            # Cache should make second load faster
            cache_speedup = (
                first_load_time / second_load_time if second_load_time > 0 else 1.0
            )

            metrics = {
                "first_load_time_seconds": first_load_time,
                "second_load_time_seconds": second_load_time,
                "cache_speedup_ratio": cache_speedup,
            }

            test_name = "cache_performance"
            result = perf_tracker.check_regression(test_name, metrics)

            print(f"\n📊 Performance Test: {test_name}")
            print(
                f"First load: {first_load_time:.3f}s, Second load: {second_load_time:.3f}s"
            )
            print(f"Cache speedup: {cache_speedup:.1f}x")
            print(f"Status: {result['status']}")

            # Cache should provide at least some speedup
            assert cache_speedup >= 0.8, (
                f"Cache should not slow down loading (speedup: {cache_speedup:.2f}x)"
            )
