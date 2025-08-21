#!/usr/bin/env python3
"""
Production Deployment Example with OpenVINO-Easy

This example demonstrates:
- Production-ready model loading and caching
- Error handling and graceful degradation
- Performance monitoring and logging
- Resource management and optimization
- Batch processing for high throughput
- Health checks and monitoring endpoints
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import oe
import time
import logging
import json
import threading
import queue
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import psutil


# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/openvino_easy_production.log"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for production model deployment."""

    model_id: str
    device_preference: List[str]
    dtype: str = "int8"
    cache_dir: Optional[str] = None
    max_batch_size: int = 1
    timeout_seconds: float = 30.0
    fallback_models: List[str] = None
    health_check_interval: float = 300.0  # 5 minutes


@dataclass
class InferenceResult:
    """Structured inference result with metadata."""

    result: Any
    inference_time: float
    model_id: str
    device: str
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class ModelRegistry:
    """Model registry for production deployment and lifecycle management."""

    def __init__(self):
        self.loaded_models: Dict[str, str] = {}  # Model name to identifier mapping
        self.configs: Dict[str, ModelConfig] = {}  # Configuration store
        self.stats: Dict[str, Dict[str, Any]] = {}  # Performance metrics
        self._lock = threading.Lock()  # Thread safety for concurrent access
        self._current_model = None  # Active model tracking

    def register_model(self, name: str, config: ModelConfig) -> bool:
        """Register model configuration in the enterprise registry."""
        try:
            with self._lock:
                logger.info(f"Registering model configuration: {name}")
                self.configs[name] = config
                self.stats[name] = {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "total_inference_time": 0.0,
                    "last_request_time": None,
                    "last_error": None,
                }
                return True
        except Exception as e:
            logger.error(f"Failed to register model {name}: {e}")
            return False

    def load_model(self, name: str) -> bool:
        """Load and deploy registered model with enterprise configurations."""
        if name not in self.configs:
            logger.error(f"Model configuration not found: {name}")
            return False

        config = self.configs[name]

        try:
            with self._lock:
                logger.info(f"Loading model: {name}")
                start_time = time.time()

                # Try main model first (NEW API)
                try:
                    # Unload current model if different
                    if self._current_model and self._current_model != name:
                        oe.unload()
                        self._current_model = None

                    oe.load(
                        config.model_id,
                        device_preference=config.device_preference,
                        dtype=config.dtype,
                        cache_dir=config.cache_dir,
                    )

                    # Get model info (NEW API)
                    info = oe.get_info()
                    self.loaded_models[name] = config.model_id
                    self._current_model = name
                    load_time = time.time() - start_time

                    logger.info(
                        f"Model {name} loaded successfully in {load_time:.2f}s on {info['device']}"
                    )
                    return True

                except Exception as main_error:
                    logger.warning(
                        f"Failed to load main model {config.model_id}: {main_error}"
                    )

                    # Try fallback models
                    if config.fallback_models:
                        for fallback_id in config.fallback_models:
                            try:
                                logger.info(f"Trying fallback model: {fallback_id}")
                                # Unload failed model first
                                if self._current_model:
                                    oe.unload()
                                    self._current_model = None

                                oe.load(
                                    fallback_id,
                                    device_preference=config.device_preference,
                                    dtype=config.dtype,
                                    cache_dir=config.cache_dir,
                                )

                                self.loaded_models[name] = fallback_id
                                self._current_model = name
                                load_time = time.time() - start_time

                                logger.info(
                                    f"Fallback model {fallback_id} loaded in {load_time:.2f}s"
                                )
                                return True

                            except Exception as fallback_error:
                                logger.warning(
                                    f"Fallback model {fallback_id} failed: {fallback_error}"
                                )

                    # All models failed
                    logger.error(f"All models failed for {name}")
                    return False

        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")
            return False

    def infer(
        self, name: str, input_data: Any, timeout: Optional[float] = None
    ) -> InferenceResult:
        """Run inference with error handling and monitoring."""
        start_time = time.time()

        # Update stats
        with self._lock:
            self.stats[name]["total_requests"] += 1
            self.stats[name]["last_request_time"] = start_time

        if name not in self.loaded_models:
            error_msg = f"Model {name} not loaded"
            logger.error(error_msg)
            with self._lock:
                self.stats[name]["failed_requests"] += 1
                self.stats[name]["last_error"] = error_msg

            return InferenceResult(
                result=None,
                inference_time=0.0,
                model_id=name,
                device="unknown",
                success=False,
                error_message=error_msg,
            )

        # Ensure correct model is loaded
        if self._current_model != name:
            # Load the requested model
            if not self.load_model(name):
                error_msg = f"Failed to switch to model {name}"
                return InferenceResult(
                    result=None,
                    inference_time=0.0,
                    model_id=name,
                    device="unknown",
                    success=False,
                    error_message=error_msg,
                )

        config = self.configs[name]
        timeout = timeout or config.timeout_seconds

        try:
            # Run inference with timeout (NEW API)
            result = self._run_with_timeout(oe.infer, input_data, timeout)
            inference_time = time.time() - start_time

            # Update success stats
            with self._lock:
                self.stats[name]["successful_requests"] += 1
                self.stats[name]["total_inference_time"] += inference_time

            logger.info(f"Inference completed for {name} in {inference_time:.3f}s")

            # Get model info (NEW API)
            info = oe.get_info()

            return InferenceResult(
                result=result,
                inference_time=inference_time,
                model_id=self.loaded_models[name],
                device=info.get("device", "unknown"),
                success=True,
                metadata={
                    "quantized": info.get("quantized", False),
                    "cache_hit": info.get("cache_hit", False),
                },
            )

        except TimeoutError:
            error_msg = f"Inference timeout ({timeout}s) for {name}"
            logger.error(error_msg)
            with self._lock:
                self.stats[name]["failed_requests"] += 1
                self.stats[name]["last_error"] = error_msg

            # Get current device info if possible
            try:
                info = oe.get_info()
                device = info.get("device", "unknown")
            except:
                device = "unknown"

            return InferenceResult(
                result=None,
                inference_time=time.time() - start_time,
                model_id=self.loaded_models.get(name, "unknown"),
                device=device,
                success=False,
                error_message=error_msg,
            )

        except Exception as e:
            error_msg = f"Inference failed for {name}: {e}"
            logger.error(error_msg)
            inference_time = time.time() - start_time

            with self._lock:
                self.stats[name]["failed_requests"] += 1
                self.stats[name]["last_error"] = str(e)
                self.stats[name]["total_inference_time"] += inference_time

            # Get current device info if possible
            try:
                info = oe.get_info()
                device = info.get("device", "unknown")
            except:
                device = "unknown"

            return InferenceResult(
                result=None,
                inference_time=inference_time,
                model_id=self.loaded_models.get(name, "unknown"),
                device=device,
                success=False,
                error_message=error_msg,
            )

    def _run_with_timeout(self, func, *args, timeout: float):
        """Run function with timeout."""
        result_queue = queue.Queue()
        exception_queue = queue.Queue()

        def target():
            try:
                result = func(*args)
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            # Timeout occurred
            thread.join(0.1)  # Give a moment to clean up
            raise TimeoutError(f"Operation timed out after {timeout}s")

        if not exception_queue.empty():
            raise exception_queue.get()

        if not result_queue.empty():
            return result_queue.get()

        raise RuntimeError("No result or exception captured")

    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            if name:
                if name in self.stats:
                    stats = self.stats[name].copy()
                    # Calculate derived metrics
                    if stats["successful_requests"] > 0:
                        stats["avg_inference_time"] = (
                            stats["total_inference_time"] / stats["successful_requests"]
                        )
                        stats["success_rate"] = (
                            stats["successful_requests"] / stats["total_requests"]
                        )
                    else:
                        stats["avg_inference_time"] = 0.0
                        stats["success_rate"] = 0.0
                    return stats
                else:
                    return {}
            else:
                return {name: self.get_stats(name) for name in self.stats.keys()}

    def health_check(self, name: str) -> Dict[str, Any]:
        """Perform health check on a model."""
        health_status = {
            "model_name": name,
            "timestamp": time.time(),
            "status": "unknown",
            "details": {},
        }

        try:
            # Check if model is loaded
            if name not in self.loaded_models:
                health_status["status"] = "not_loaded"
                health_status["details"]["error"] = "Model not loaded"
                return health_status

            # Ensure correct model is loaded
            if self._current_model != name:
                if not self.load_model(name):
                    health_status["status"] = "load_failed"
                    health_status["details"]["error"] = (
                        "Failed to load model for health check"
                    )
                    return health_status

            # Get system resources and model info (NEW API)
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)

            try:
                info = oe.get_info()
            except:
                info = {}

            health_status["details"].update(
                {
                    "device": info.get("device", "unknown"),
                    "model_info": info,
                    "memory_usage_percent": memory_info.percent,
                    "cpu_usage_percent": cpu_percent,
                    "memory_available_gb": memory_info.available / (1024**3),
                }
            )

            # Try a simple inference
            config = self.configs[name]
            test_input = self._get_test_input(config.model_id)

            if test_input is not None:
                start_time = time.time()
                oe.infer(test_input)  # NEW API
                test_inference_time = time.time() - start_time

                health_status["details"].update(
                    {
                        "test_inference_time": test_inference_time,
                        "test_inference_success": True,
                    }
                )

                # Performance thresholds
                if test_inference_time < 1.0:
                    health_status["status"] = "healthy"
                elif test_inference_time < 5.0:
                    health_status["status"] = "warning"
                else:
                    health_status["status"] = "slow"
            else:
                health_status["status"] = "healthy"  # No test available, assume healthy

        except Exception as e:
            health_status["status"] = "error"
            health_status["details"]["error"] = str(e)
            logger.error(f"Health check failed for {name}: {e}")

        return health_status

    def _get_test_input(self, model_id: str):
        """Get appropriate test input for different model types."""
        if "whisper" in model_id.lower():
            # Create a small test audio file
            import tempfile
            import wave
            import numpy as np

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                # Generate 1 second of silence
                sample_rate = 16000
                duration = 1.0
                samples = int(sample_rate * duration)
                audio_data = np.zeros(samples, dtype=np.int16)

                with wave.open(f.name, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data.tobytes())

                return f.name

        elif any(term in model_id.lower() for term in ["resnet", "vit", "image"]):
            # Image models
            import numpy as np

            return np.random.randn(1, 3, 224, 224).astype(np.float32)

        else:
            # Text models
            return "This is a test input for health check."


class ProductionModelService:
    """High-level production service for model inference."""

    def __init__(self, config_file: Optional[str] = None):
        self.registry = ModelRegistry()
        self.config_file = config_file
        self._health_check_thread = None
        self._stop_health_checks = threading.Event()

        if config_file:
            self.load_config(config_file)

    def load_config(self, config_file: str):
        """Load model configurations from file."""
        try:
            with open(config_file, "r") as f:
                configs = json.load(f)

            for name, config_dict in configs.items():
                config = ModelConfig(**config_dict)
                self.registry.register_model(name, config)
                logger.info(f"Loaded config for model: {name}")

        except Exception as e:
            logger.error(f"Failed to load config from {config_file}: {e}")

    def start_models(self, model_names: Optional[List[str]] = None):
        """Start specified models or all registered models."""
        if model_names is None:
            model_names = list(self.registry.configs.keys())

        for name in model_names:
            success = self.registry.load_model(name)
            if success:
                logger.info(f"Successfully started model: {name}")
            else:
                logger.error(f"Failed to start model: {name}")

    def start_health_monitoring(self):
        """Start background health monitoring."""
        if self._health_check_thread is not None:
            return

        def health_monitor():
            while not self._stop_health_checks.wait(60):  # Check every minute
                for name in self.registry.loaded_models.keys():
                    try:
                        health = self.registry.health_check(name)
                        if health["status"] in ["error", "slow"]:
                            logger.warning(
                                f"Health issue detected for {name}: {health}"
                            )
                    except Exception as e:
                        logger.error(f"Health monitoring failed for {name}: {e}")

        self._health_check_thread = threading.Thread(target=health_monitor, daemon=True)
        self._health_check_thread.start()
        logger.info("Health monitoring started")

    def stop_health_monitoring(self):
        """Stop background health monitoring."""
        if self._health_check_thread is not None:
            self._stop_health_checks.set()
            self._health_check_thread.join(timeout=5)
            self._health_check_thread = None
            logger.info("Health monitoring stopped")

    def infer(self, model_name: str, input_data: Any, **kwargs) -> InferenceResult:
        """Run inference on specified model."""
        return self.registry.infer(model_name, input_data, **kwargs)

    def batch_infer(
        self, model_name: str, input_batch: List[Any], max_workers: int = 4
    ) -> List[InferenceResult]:
        """Process batch of inputs with parallel workers."""
        import concurrent.futures

        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all inference tasks
            future_to_input = {
                executor.submit(self.registry.infer, model_name, input_data): i
                for i, input_data in enumerate(input_batch)
            }

            # Collect results in order
            results = [None] * len(input_batch)

            for future in concurrent.futures.as_completed(future_to_input):
                index = future_to_input[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    logger.error(f"Batch inference failed for item {index}: {e}")
                    results[index] = InferenceResult(
                        result=None,
                        inference_time=0.0,
                        model_id=model_name,
                        device="unknown",
                        success=False,
                        error_message=str(e),
                    )

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics."""
        # Get model stats
        model_stats = self.registry.get_stats()

        # Get system metrics
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)

        # Get OpenVINO device info
        available_devices = oe.devices()

        return {
            "timestamp": time.time(),
            "models": model_stats,
            "system": {
                "memory_usage_percent": memory_info.percent,
                "memory_available_gb": memory_info.available / (1024**3),
                "cpu_usage_percent": cpu_percent,
                "available_devices": available_devices,
            },
            "service": {
                "loaded_models": list(self.registry.loaded_models.keys()),
                "registered_models": list(self.registry.configs.keys()),
                "current_model": self.registry._current_model,
                "health_monitoring_active": self._health_check_thread is not None,
            },
        }


def demo_production_setup():
    """Demonstrate production model deployment setup."""
    print("🏭 === Production Model Service Demo ===")

    # Create service configuration
    config = {
        "text_generator": {
            "model_id": "distilgpt2",
            "device_preference": ["NPU", "GPU", "CPU"],
            "dtype": "int8",
            "cache_dir": "/tmp/openvino_cache",
            "fallback_models": ["gpt2"],
            "timeout_seconds": 10.0,
        },
        "image_classifier": {
            "model_id": "microsoft/resnet-18",
            "device_preference": ["NPU", "GPU", "CPU"],
            "dtype": "int8",
            "cache_dir": "/tmp/openvino_cache",
            "timeout_seconds": 5.0,
        },
    }

    # Save config to temporary file
    config_file = "/tmp/model_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"📝 Created config file: {config_file}")

    # Initialize service
    service = ProductionModelService(config_file)

    # Start models
    print("🚀 Starting models...")
    service.start_models()

    # Start health monitoring
    service.start_health_monitoring()

    # Test individual inference
    print("\n🧪 Testing individual inference...")

    text_result = service.infer("text_generator", "The future of AI is")
    print(
        f"Text generation result: Success={text_result.success}, "
        f"Time={text_result.inference_time:.3f}s"
    )

    if text_result.success:
        print(f"Generated text: {str(text_result.result)[:100]}...")

    # Test batch inference
    print("\n📦 Testing batch inference...")

    text_batch = [
        "AI will revolutionize",
        "The benefits of automation include",
        "Future technology trends are",
    ]

    batch_results = service.batch_infer("text_generator", text_batch, max_workers=2)

    successful_results = sum(1 for r in batch_results if r.success)
    avg_time = sum(r.inference_time for r in batch_results if r.success) / max(
        successful_results, 1
    )

    print(f"Batch results: {successful_results}/{len(batch_results)} successful")
    print(f"Average inference time: {avg_time:.3f}s")

    # Get metrics
    print("\n📊 Service metrics:")
    metrics = service.get_metrics()

    print(f"System memory usage: {metrics['system']['memory_usage_percent']:.1f}%")
    print(f"Available devices: {metrics['system']['available_devices']}")

    for model_name, stats in metrics["models"].items():
        print(f"Model {model_name}:")
        print(
            f"  Requests: {stats['total_requests']} (success rate: {stats.get('success_rate', 0):.1%})"
        )
        print(f"  Avg inference time: {stats.get('avg_inference_time', 0):.3f}s")

    # Health checks
    print("\n🏥 Health checks:")
    for model_name in service.registry.loaded_models.keys():
        health = service.registry.health_check(model_name)
        print(f"Model {model_name}: {health['status']}")
        if health["status"] != "healthy":
            print(f"  Details: {health['details']}")

    # Stop monitoring and clean up (NEW API)
    service.stop_health_monitoring()

    # Clean up models
    if service.registry._current_model:
        oe.unload()

    print("✅ Production demo completed!")


def demo_error_handling():
    """Demonstrate robust error handling patterns."""
    print("\n🛡️ === Error Handling Demo ===")

    service = ProductionModelService()

    # Register a model that might fail
    config = ModelConfig(
        model_id="nonexistent/model",
        device_preference=["CPU"],
        fallback_models=["distilgpt2"],  # Working fallback
        timeout_seconds=5.0,
    )

    service.registry.register_model("fallback_test", config)

    # Try to load (should use fallback)
    print("🔄 Testing fallback mechanism...")
    success = service.registry.load_model("fallback_test")

    if success:
        print("✅ Fallback model loaded successfully")

        # Test inference with timeout
        print("⏱️ Testing timeout handling...")
        result = service.infer("fallback_test", "Test input with timeout")

        if result.success:
            print(f"✅ Inference successful: {result.inference_time:.3f}s")
        else:
            print(f"❌ Inference failed: {result.error_message}")

    else:
        print("❌ Model loading failed completely")

    # Test with non-existent model
    print("\n🚫 Testing non-existent model...")
    result = service.infer("nonexistent_model", "test")
    print(f"Expected failure: Success={result.success}, Error='{result.error_message}'")


def main():
    """Run production deployment demonstrations."""
    print("🏭 OpenVINO-Easy Production Deployment Demo")
    print("==========================================")

    try:
        demo_production_setup()
        demo_error_handling()

        print("\n✅ All production demos completed!")
        print("\n💡 Production Deployment Tips:")
        print("  - Use model registry for centralized management")
        print("  - Implement health checks and monitoring")
        print("  - Configure fallback models for reliability")
        print("  - Use batch processing for higher throughput")
        print("  - Set appropriate timeouts for your use case")
        print("  - Monitor memory and CPU usage")
        print("  - Cache models locally for faster startup")
        print("  - Log all inference requests for debugging")
        print("  - Implement graceful degradation strategies")

    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Production demo failed")


if __name__ == "__main__":
    main()
