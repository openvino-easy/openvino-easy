# Performance Tuning Guide

> **📝 Note**: This guide uses the new **3-function API** (`oe.load()`, `oe.infer()`, `oe.benchmark()`, `oe.unload()`). 
> For legacy Pipeline class documentation, see [Pipeline API Reference](api/pipeline.rst).

This guide covers techniques to optimize OpenVINO-Easy performance for production workloads.

## Table of Contents

1. [Performance Fundamentals](#performance-fundamentals)
2. [Device Selection & Optimization](#device-selection--optimization)
3. [Model Quantization](#model-quantization)
4. [Batching Strategies](#batching-strategies)
5. [Memory Optimization](#memory-optimization)
6. [Caching & Preprocessing](#caching--preprocessing)
7. [Benchmarking & Profiling](#benchmarking--profiling)
8. [Platform-Specific Tuning](#platform-specific-tuning)
9. [Advanced Optimization](#advanced-optimization)
10. [Monitoring & Debugging](#monitoring--debugging)

## Performance Fundamentals

### Key Performance Metrics

Key metrics to measure:

```python
import oe
import time

# Load model for benchmarking
oe.load("distilgpt2", device_preference=["CPU"], dtype="int8")

# Comprehensive benchmark
stats = oe.benchmark(
    warmup_runs=5,        # Warm up the model
    benchmark_runs=20     # Actual measurement runs
)

# Clean up
oe.unload()

print("📊 Performance Metrics:")
for metric, value in stats.items():
    print(f"  {metric}: {value:.3f}")
```

**Key Metrics Explained:**
- **Latency**: Time for single inference (ms)
- **Throughput**: Inferences per second (FPS/tokens/sec)
- **Memory Usage**: RAM consumption (MB)
- **Device Utilization**: Hardware efficiency (%)
- **Real-time Factor**: For audio/video (1.0x = real-time)

### Performance Baseline

Establish baseline performance before optimization:

```python
def establish_baseline(model_id: str, test_input: str = "test"):
    """Establish performance baseline for a model."""
    
    print(f"🎯 Establishing baseline for {model_id}")
    
    # Test different configurations
    configs = [
        {"device": "CPU", "dtype": "fp32", "desc": "CPU FP32 (baseline)"},
        {"device": "CPU", "dtype": "int8", "desc": "CPU INT8 (quantized)"},
    ]
    
    # Add GPU/NPU if available
    available_devices = oe.devices()
    if "GPU" in available_devices:
        configs.append({"device": "GPU", "dtype": "fp16", "desc": "GPU FP16"})
    if "NPU" in available_devices:
        configs.append({"device": "NPU", "dtype": "int8", "desc": "NPU INT8"})
    
    results = []
    
    for config in configs:
        try:
            print(f"\n⚡ Testing: {config['desc']}")
            
            # Load model with configuration
            oe.load(
                model_id,
                device_preference=[config["device"]],
                dtype=config["dtype"]
            )
            
            # Benchmark
            stats = oe.benchmark(
                warmup_runs=3,
                benchmark_runs=10
            )
            
            # Get model info
            info = oe.get_info()
            
            result = {
                "config": config["desc"],
                "device": info["device"],
                "latency_ms": stats.get("avg_latency_ms", 0),
                "throughput": stats.get("throughput_token_per_sec", 0),
                "memory_mb": stats.get("memory_usage_mb", 0)
            }
            
            # Clean up
            oe.unload()
            
            results.append(result)
            print(f"  ✅ Latency: {result['latency_ms']:.1f}ms, "
                  f"Throughput: {result['throughput']:.1f}/sec")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Display comparison
    print(f"\n📈 Baseline Results:")
    print(f"{'Configuration':<20} {'Latency (ms)':<12} {'Throughput':<12} {'Memory (MB)'}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['config']:<20} "
              f"{result['latency_ms']:<12.1f} "
              f"{result['throughput']:<12.1f} "
              f"{result['memory_mb']:<12.1f}")
    
    return results

# Example usage
baseline_results = establish_baseline("distilgpt2")
```

## Device Selection & Optimization

### Intelligent Device Selection

OpenVINO-Easy automatically selects the best device, but you can optimize this:

```python
def optimize_device_selection():
    """Demonstrate optimal device selection strategies."""
    
    available_devices = oe.devices()
    print(f"🖥️ Available devices: {available_devices}")
    
    # Device-specific optimization strategies
    optimization_strategies = {
        "NPU": {
            "preferred_models": ["whisper", "bert", "distilbert"],
            "optimal_dtype": "int8",
            "batch_size": 1,  # NPU works best with single inference
            "description": "Best for: NLP models, edge deployment, power efficiency"
        },
        "GPU": {
            "preferred_models": ["stable-diffusion", "resnet", "vit", "clip"],
            "optimal_dtype": "fp16",
            "batch_size": 4,  # GPU benefits from batching
            "description": "Best for: Vision models, large batch sizes, parallel processing"
        },
        "CPU": {
            "preferred_models": ["all"],
            "optimal_dtype": "int8",
            "batch_size": 1,
            "description": "Best for: Universal compatibility, reliable fallback"
        }
    }
    
    for device, strategy in optimization_strategies.items():
        if device in available_devices:
            print(f"\n🔧 {device} Optimization Strategy:")
            print(f"  Preferred models: {', '.join(strategy['preferred_models'])}")
            print(f"  Optimal dtype: {strategy['optimal_dtype']}")
            print(f"  Recommended batch size: {strategy['batch_size']}")
            print(f"  {strategy['description']}")

optimize_device_selection()
```

### Device-Specific Configuration

```python
def configure_for_device(model_id: str, target_device: str):
    """Configure model optimally for specific device."""
    
    device_configs = {
        "NPU": {
            "dtype": "int8",
            "batch_size": 1,
            "threads": 1,
            "optimization_level": "high"
        },
        "GPU": {
            "dtype": "fp16",
            "batch_size": 4,
            "optimization_level": "medium"
        },
        "CPU": {
            "dtype": "int8",
            "batch_size": 1,
            "threads": None,  # Auto-detect
            "optimization_level": "medium"
        }
    }
    
    if target_device not in device_configs:
        raise ValueError(f"Device {target_device} not supported")
    
    config = device_configs[target_device]
    
    print(f"⚙️ Configuring {model_id} for {target_device}")
    print(f"  Configuration: {config}")
    
    # Load with optimized configuration
    pipeline = oe.load(
        model_id,
        device_preference=[target_device],
        dtype=config["dtype"]
    )
    
    print(f"✅ Loaded on {pipeline.device}")
    return pipeline, config

# Example usage
if "NPU" in oe.devices():
    npu_pipeline, npu_config = configure_for_device("distilgpt2", "NPU")
```

## Model Quantization

### Quantization Strategies

```python
def compare_quantization_levels():
    """Compare different quantization strategies."""
    
    model_id = "distilgpt2"
    test_input = "Compare quantization performance"
    
    quantization_levels = [
        {"dtype": "fp32", "desc": "Full Precision (Baseline)", "quality": "Highest", "speed": "Slowest"},
        {"dtype": "fp16", "desc": "Half Precision", "quality": "High", "speed": "Fast"},
        {"dtype": "int8", "desc": "8-bit Quantized", "quality": "Good", "speed": "Fastest"},
    ]
    
    print("🔢 Quantization Level Comparison:")
    print(f"{'Level':<20} {'Quality':<10} {'Speed':<10} {'Latency (ms)':<12} {'Size Reduction'}")
    print("-" * 75)
    
    results = []
    baseline_latency = None
    
    for quant in quantization_levels:
        try:
            oe.load(
                model_id,
                device_preference=["CPU"],
                dtype=quant["dtype"]
            )
            
            # Benchmark
            stats = oe.benchmark(
                warmup_runs=3,
                benchmark_runs=10
            )
            
            # Clean up
            oe.unload()
            
            latency = stats.get("avg_latency_ms", 0)
            if baseline_latency is None:
                baseline_latency = latency
            
            speedup = f"{baseline_latency/latency:.1f}x" if latency > 0 else "N/A"
            
            print(f"{quant['desc']:<20} "
                  f"{quant['quality']:<10} "
                  f"{quant['speed']:<10} "
                  f"{latency:<12.1f} "
                  f"{speedup}")
            
            results.append({
                "dtype": quant["dtype"],
                "latency": latency,
                "speedup": baseline_latency/latency if latency > 0 else 1.0
            })
            
        except Exception as e:
            print(f"{quant['desc']:<20} ❌ Failed: {e}")
    
    return results

# Run quantization comparison
quant_results = compare_quantization_levels()
```

### Smart Quantization Selection

```python
def select_optimal_quantization(model_id: str, quality_threshold: float = 0.95):
    """Automatically select optimal quantization based on quality/speed trade-off."""
    
    print(f"🎯 Finding optimal quantization for {model_id}")
    
    # Test different quantization levels
    levels = ["fp32", "fp16", "int8"]
    results = []
    
    for dtype in levels:
        try:
            oe.load(model_id, dtype=dtype, device_preference=["CPU"])
            
            # Benchmark performance
            stats = oe.benchmark(
                warmup_runs=2,
                benchmark_runs=5
            )
            
            # Clean up
            oe.unload()
            
            # Simulate quality score (in real scenario, use validation dataset)
            quality_scores = {"fp32": 1.0, "fp16": 0.995, "int8": 0.98}
            quality = quality_scores.get(dtype, 0.9)
            
            results.append({
                "dtype": dtype,
                "latency": stats.get("avg_latency_ms", 0),
                "quality": quality,
                "meets_threshold": quality >= quality_threshold
            })
            
            print(f"  {dtype}: {stats.get('avg_latency_ms', 0):.1f}ms, "
                  f"quality: {quality:.3f}")
            
        except Exception as e:
            print(f"  {dtype}: Failed - {e}")
    
    # Select best option that meets quality threshold
    valid_options = [r for r in results if r["meets_threshold"]]
    
    if valid_options:
        # Choose fastest option that meets quality threshold
        optimal = min(valid_options, key=lambda x: x["latency"])
        print(f"\n✅ Optimal quantization: {optimal['dtype']}")
        print(f"   Latency: {optimal['latency']:.1f}ms")
        print(f"   Quality: {optimal['quality']:.3f}")
        return optimal["dtype"]
    else:
        print(f"\n⚠️ No quantization meets quality threshold {quality_threshold}")
        return "fp32"

# Example usage
optimal_dtype = select_optimal_quantization("distilgpt2", quality_threshold=0.97)
```

## Batching Strategies

### Dynamic Batching

```python
import asyncio
from typing import List, Any
import time

class DynamicBatcher:
    """Intelligent batching for improved throughput."""
    
    def __init__(self, model_pipeline, max_batch_size: int = 8, max_wait_time: float = 0.1):
        self.pipeline = model_pipeline
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.processing = False
    
    async def infer(self, input_data: Any) -> Any:
        """Add request to batch and wait for result."""
        
        # Create future for this request
        future = asyncio.Future()
        request = {
            "input": input_data,
            "future": future,
            "timestamp": time.time()
        }
        
        self.pending_requests.append(request)
        
        # Trigger batch processing if needed
        if not self.processing:
            asyncio.create_task(self._process_batches())
        
        # Wait for result
        return await future
    
    async def _process_batches(self):
        """Process requests in optimal batches."""
        if self.processing:
            return
        
        self.processing = True
        
        try:
            while self.pending_requests:
                # Wait for batch to fill or timeout
                start_time = time.time()
                
                while (len(self.pending_requests) < self.max_batch_size and 
                       time.time() - start_time < self.max_wait_time):
                    await asyncio.sleep(0.01)
                
                if not self.pending_requests:
                    break
                
                # Extract batch
                batch_size = min(len(self.pending_requests), self.max_batch_size)
                batch_requests = self.pending_requests[:batch_size]
                self.pending_requests = self.pending_requests[batch_size:]
                
                # Process batch
                await self._process_batch(batch_requests)
        
        finally:
            self.processing = False
    
    async def _process_batch(self, batch_requests: List[dict]):
        """Process a batch of requests."""
        try:
            # Extract inputs
            inputs = [req["input"] for req in batch_requests]
            
            # Process batch (simplified - actual batching depends on model type)
            if len(inputs) == 1:
                # Single inference
                result = self.pipeline.infer(inputs[0])
                results = [result]
            else:
                # Simulate batch processing
                results = []
                for inp in inputs:
                    result = self.pipeline.infer(inp)
                    results.append(result)
            
            # Return results to futures
            for request, result in zip(batch_requests, results):
                request["future"].set_result(result)
                
        except Exception as e:
            # Handle errors
            for request in batch_requests:
                request["future"].set_exception(e)

# Usage example
async def demo_dynamic_batching():
    """Demonstrate dynamic batching benefits."""
    
    # Load model
    pipeline = oe.load("distilgpt2", device_preference=["CPU"], dtype="int8")
    batcher = DynamicBatcher(pipeline, max_batch_size=4, max_wait_time=0.05)
    
    # Test individual vs batched processing
    test_inputs = [
        "First test input",
        "Second test input", 
        "Third test input",
        "Fourth test input"
    ]
    
    print("🔄 Testing dynamic batching...")
    
    # Individual processing (baseline)
    start_time = time.time()
    individual_results = []
    for inp in test_inputs:
        result = pipeline.infer(inp)
        individual_results.append(result)
    individual_time = time.time() - start_time
    
    # Batched processing
    start_time = time.time()
    batch_tasks = [batcher.infer(inp) for inp in test_inputs]
    batch_results = await asyncio.gather(*batch_tasks)
    batch_time = time.time() - start_time
    
    print(f"📊 Batching Results:")
    print(f"  Individual processing: {individual_time:.3f}s")
    print(f"  Batched processing: {batch_time:.3f}s")
    print(f"  Speedup: {individual_time/batch_time:.1f}x")
    
    return individual_time, batch_time

# Run demo (in async context)
# asyncio.run(demo_dynamic_batching())
```

### Batch Size Optimization

```python
def find_optimal_batch_size(model_id: str, test_inputs: List[str]):
    """Find optimal batch size for throughput."""
    
    pipeline = oe.load(model_id, device_preference=["CPU"], dtype="int8")
    
    batch_sizes = [1, 2, 4, 8, 16]
    results = []
    
    print(f"🔍 Finding optimal batch size for {model_id}")
    print(f"{'Batch Size':<12} {'Latency (ms)':<15} {'Throughput':<15} {'Efficiency'}")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        try:
            # Prepare batch
            batch = test_inputs[:batch_size]
            if len(batch) < batch_size:
                # Pad batch if needed
                batch.extend(test_inputs * ((batch_size // len(test_inputs)) + 1))
                batch = batch[:batch_size]
            
            # Benchmark batch processing
            start_time = time.time()
            
            # Process multiple batches for accurate measurement
            num_batches = 10
            for _ in range(num_batches):
                for inp in batch:  # Simulate batch processing
                    pipeline.infer(inp)
            
            total_time = time.time() - start_time
            
            # Calculate metrics
            total_inferences = batch_size * num_batches
            avg_latency = (total_time / num_batches) * 1000  # ms per batch
            throughput = total_inferences / total_time
            efficiency = throughput / batch_size  # throughput per item in batch
            
            results.append({
                "batch_size": batch_size,
                "latency": avg_latency,
                "throughput": throughput,
                "efficiency": efficiency
            })
            
            print(f"{batch_size:<12} "
                  f"{avg_latency:<15.1f} "
                  f"{throughput:<15.1f} "
                  f"{efficiency:<15.2f}")
            
        except Exception as e:
            print(f"{batch_size:<12} ❌ Failed: {e}")
    
    # Find optimal batch size (best throughput/efficiency trade-off)
    if results:
        optimal = max(results, key=lambda x: x["throughput"])
        print(f"\n✅ Optimal batch size: {optimal['batch_size']}")
        print(f"   Throughput: {optimal['throughput']:.1f} inferences/sec")
        return optimal["batch_size"]
    
    return 1

# Example usage
test_texts = ["test input " + str(i) for i in range(16)]
optimal_batch = find_optimal_batch_size("distilgpt2", test_texts)
```

## Memory Optimization

### Memory Usage Monitoring

```python
import psutil
import gc
import tracemalloc

def monitor_memory_usage():
    """Monitor and optimize memory usage."""
    
    def get_memory_info():
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent()
        }
    
    print("💾 Memory Usage Monitoring")
    
    # Baseline memory
    baseline = get_memory_info()
    print(f"Baseline memory: {baseline['rss_mb']:.1f} MB ({baseline['percent']:.1f}%)")
    
    # Start memory tracing
    tracemalloc.start()
    
    # Load model and monitor memory
    print("\n📈 Loading model...")
    oe.load("distilgpt2", device_preference=["CPU"], dtype="int8")
    
    after_load = get_memory_info()
    model_memory = after_load["rss_mb"] - baseline["rss_mb"]
    print(f"After model load: +{model_memory:.1f} MB")
    
    # Run inference and monitor
    print("\n🔄 Running inference...")
    for i in range(10):
        result = oe.infer(f"Test input {i}")
        if i == 0:
            first_inference = get_memory_info()
            inference_memory = first_inference["rss_mb"] - after_load["rss_mb"]
            print(f"After first inference: +{inference_memory:.1f} MB")
    
    final_memory = get_memory_info()
    total_increase = final_memory["rss_mb"] - baseline["rss_mb"]
    
    # Get top memory consumers
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\n📊 Memory Summary:")
    print(f"  Model loading: {model_memory:.1f} MB")
    print(f"  First inference: {inference_memory:.1f} MB")
    print(f"  Total increase: {total_increase:.1f} MB")
    print(f"  Peak traced memory: {peak / 1024 / 1024:.1f} MB")
    
    return {
        "model_memory": model_memory,
        "inference_memory": inference_memory,
        "total_memory": total_increase
    }

# Memory optimization tips
def optimize_memory_usage():
    """Demonstrate memory optimization techniques."""
    
    print("🚀 Memory Optimization Techniques:")
    
    # 1. Use quantization
    print("\n1. Quantization reduces memory usage:")
    fp32_pipeline = oe.load("distilgpt2", dtype="fp32", device_preference=["CPU"])
    int8_pipeline = oe.load("distilgpt2", dtype="int8", device_preference=["CPU"])
    
    print("   FP32 model loaded")
    print("   INT8 model loaded (typically 2-4x smaller)")
    
    # 2. Clear unused models
    print("\n2. Clear unused models from memory:")
    del fp32_pipeline
    gc.collect()
    print("   FP32 model cleared from memory")
    
    # 3. Use device-specific optimization
    print("\n3. Device-specific optimization:")
    print("   CPU: Use int8 quantization")
    print("   GPU: Use fp16 for balance of speed/memory")
    print("   NPU: Use int8 for maximum efficiency")
    
    return int8_pipeline

memory_stats = monitor_memory_usage()
optimized_pipeline = optimize_memory_usage()
```

### Memory-Efficient Loading

```python
def memory_efficient_model_loading():
    """Demonstrate memory-efficient model loading patterns."""
    
    print("🧠 Memory-Efficient Loading Strategies")
    
    # Strategy 1: Load models on-demand
    class OnDemandModelManager:
        def __init__(self):
            self.model_configs = {
                "text": {"id": "distilgpt2", "dtype": "int8"},
                "image": {"id": "microsoft/resnet-18", "dtype": "int8"},
            }
            self.loaded_models = {}
        
        def get_model(self, model_type: str):
            if model_type not in self.loaded_models:
                config = self.model_configs[model_type]
                print(f"  Loading {model_type} model on-demand...")
                self.loaded_models[model_type] = oe.load(
                    config["id"],
                    dtype=config["dtype"],
                    device_preference=["CPU"]
                )
            return self.loaded_models[model_type]
        
        def clear_unused(self, keep_model: str = None):
            """Clear all models except the one specified."""
            for model_type in list(self.loaded_models.keys()):
                if model_type != keep_model:
                    del self.loaded_models[model_type]
                    print(f"  Cleared {model_type} model from memory")
            gc.collect()
    
    # Strategy 2: Model sharing for similar tasks
    class SharedModelManager:
        def __init__(self):
            self.shared_models = {}
        
        def get_shared_model(self, model_family: str, task: str):
            """Share models across similar tasks."""
            if model_family not in self.shared_models:
                if model_family == "text":
                    model_id = "distilgpt2"  # Good for multiple text tasks
                elif model_family == "vision":
                    model_id = "microsoft/resnet-18"  # Good for classification
                else:
                    raise ValueError(f"Unknown model family: {model_family}")
                
                print(f"  Loading shared {model_family} model...")
                self.shared_models[model_family] = oe.load(
                    model_id,
                    dtype="int8",
                    device_preference=["CPU"]
                )
            
            return self.shared_models[model_family]
    
    # Demonstrate on-demand loading
    print("\n1. On-Demand Loading:")
    manager = OnDemandModelManager()
    
    # Only load when needed
    text_model = manager.get_model("text")
    print("   Text model loaded")
    
    # Clear when switching tasks
    manager.clear_unused(keep_model="text")
    
    # Demonstrate shared models
    print("\n2. Shared Model Strategy:")
    shared_manager = SharedModelManager()
    
    # Same model for multiple text tasks
    text_gen_model = shared_manager.get_shared_model("text", "generation")
    text_class_model = shared_manager.get_shared_model("text", "classification")
    # Both use same underlying model instance
    print(f"   Models are same instance: {text_gen_model is text_class_model}")

memory_efficient_model_loading()
```

## Caching & Preprocessing

### Intelligent Caching

```python
import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional

class IntelligentCache:
    """Advanced caching system for models and preprocessing."""
    
    def __init__(self, cache_dir: str = "/tmp/openvino_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessing_cache = {}
        
    def get_cache_key(self, model_id: str, config: dict) -> str:
        """Generate cache key from model ID and configuration."""
        config_str = str(sorted(config.items()))
        cache_input = f"{model_id}_{config_str}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def cache_preprocessing_result(self, input_data: Any, result: Any, model_type: str):
        """Cache preprocessing results."""
        # Create key from input data
        if isinstance(input_data, str):
            key = hashlib.md5(input_data.encode()).hexdigest()
        else:
            key = hashlib.md5(str(input_data).encode()).hexdigest()
        
        cache_key = f"{model_type}_{key}"
        self.preprocessing_cache[cache_key] = result
        
        # Persist to disk for large caches
        if len(self.preprocessing_cache) > 1000:
            cache_file = self.cache_dir / f"preprocessing_{model_type}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(self.preprocessing_cache, f)
    
    def get_preprocessing_result(self, input_data: Any, model_type: str) -> Optional[Any]:
        """Retrieve cached preprocessing result."""
        if isinstance(input_data, str):
            key = hashlib.md5(input_data.encode()).hexdigest()
        else:
            key = hashlib.md5(str(input_data).encode()).hexdigest()
        
        cache_key = f"{model_type}_{key}"
        return self.preprocessing_cache.get(cache_key)
    
    def cache_model_result(self, model_id: str, input_data: Any, result: Any):
        """Cache model inference results."""
        input_hash = hashlib.md5(str(input_data).encode()).hexdigest()
        cache_file = self.cache_dir / f"results_{model_id}_{input_hash}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    
    def get_model_result(self, model_id: str, input_data: Any) -> Optional[Any]:
        """Retrieve cached model result."""
        input_hash = hashlib.md5(str(input_data).encode()).hexdigest()
        cache_file = self.cache_dir / f"results_{model_id}_{input_hash}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

# Cached pipeline wrapper
class CachedPipeline:
    """Pipeline wrapper with intelligent caching."""
    
    def __init__(self, model_id: str, cache: IntelligentCache, **load_kwargs):
        self.model_id = model_id
        self.cache = cache
        self.pipeline = oe.load(model_id, **load_kwargs)
        self.cache_hits = 0
        self.cache_misses = 0
    
    def infer(self, input_data: Any, use_cache: bool = True) -> Any:
        """Run inference with caching."""
        
        if use_cache:
            # Check cache first
            cached_result = self.cache.get_model_result(self.model_id, input_data)
            if cached_result is not None:
                self.cache_hits += 1
                return cached_result
        
        # Cache miss - run inference
        self.cache_misses += 1
        result = self.pipeline.infer(input_data)
        
        if use_cache:
            # Cache the result
            self.cache.cache_model_result(self.model_id, input_data, result)
        
        return result
    
    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }

# Demonstration
def demo_intelligent_caching():
    """Demonstrate intelligent caching benefits."""
    
    print("🗄️ Intelligent Caching Demo")
    
    # Create cache and cached pipeline
    cache = IntelligentCache()
    cached_pipeline = CachedPipeline("distilgpt2", cache, dtype="int8")
    
    # Test inputs (some repeated)
    test_inputs = [
        "First test input",
        "Second test input", 
        "First test input",  # Repeat
        "Third test input",
        "Second test input",  # Repeat
        "First test input",   # Repeat
    ]
    
    print(f"\n🔄 Processing {len(test_inputs)} inputs (with repeats)...")
    
    start_time = time.time()
    for i, inp in enumerate(test_inputs):
        result = cached_pipeline.infer(inp)
        print(f"  Input {i+1}: Processed")
    
    total_time = time.time() - start_time
    
    # Get cache statistics
    stats = cached_pipeline.get_cache_stats()
    
    print(f"\n📊 Cache Performance:")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Total time: {total_time:.2f}s")
    
    # Estimate time savings
    avg_inference_time = total_time / stats['total_requests']
    time_without_cache = avg_inference_time * len(test_inputs)
    time_saved = time_without_cache - total_time
    
    print(f"  Estimated time saved: {time_saved:.2f}s ({time_saved/time_without_cache:.1%})")

demo_intelligent_caching()
```

## Benchmarking & Profiling

### Comprehensive Benchmarking

```python
def comprehensive_benchmark(model_id: str, test_data: Any = None):
    """Run comprehensive benchmark across all dimensions."""
    
    print(f"🏁 Comprehensive Benchmark: {model_id}")
    
    # Default test data
    if test_data is None:
        if "whisper" in model_id.lower():
            # Create dummy audio file
            import wave
            import numpy as np
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sample_rate = 16000
                duration = 2.0
                samples = int(sample_rate * duration)
                audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
                audio_int16 = (audio_data * 32767).astype(np.int16)
                
                with wave.open(f.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())
                
                test_data = f.name
        else:
            test_data = "Benchmark test input"
    
    # Benchmark configurations
    configs = []
    
    # Add available device configurations
    for device in oe.devices():
        if device == "CPU":
            configs.extend([
                {"device": device, "dtype": "fp32", "desc": f"{device} FP32"},
                {"device": device, "dtype": "int8", "desc": f"{device} INT8"},
            ])
        elif device == "GPU":
            configs.extend([
                {"device": device, "dtype": "fp16", "desc": f"{device} FP16"},
                {"device": device, "dtype": "int8", "desc": f"{device} INT8"},
            ])
        elif device == "NPU":
            configs.append({"device": device, "dtype": "int8", "desc": f"{device} INT8"})
    
    results = []
    
    print(f"\n📊 Running {len(configs)} benchmark configurations...")
    print(f"{'Configuration':<15} {'Load (s)':<10} {'Latency (ms)':<12} {'Throughput':<12} {'Memory (MB)'}")
    print("-" * 70)
    
    for config in configs:
        try:
            # Measure loading time
            load_start = time.time()
            pipeline = oe.load(
                model_id,
                device_preference=[config["device"]],
                dtype=config["dtype"]
            )
            load_time = time.time() - load_start
            
            # Measure memory usage
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Run benchmark
            stats = pipeline.benchmark(
                input_data=test_data,
                warmup_runs=3,
                benchmark_runs=10
            )
            
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_usage = memory_after - memory_before
            
            result = {
                "config": config["desc"],
                "device": pipeline.device,
                "dtype": config["dtype"],
                "load_time": load_time,
                "latency": stats.get("avg_latency_ms", 0),
                "throughput": stats.get("throughput_token_per_sec", stats.get("throughput_fps", 0)),
                "memory": memory_usage
            }
            
            results.append(result)
            
            print(f"{config['desc']:<15} "
                  f"{load_time:<10.2f} "
                  f"{result['latency']:<12.1f} "
                  f"{result['throughput']:<12.1f} "
                  f"{memory_usage:<12.1f}")
            
        except Exception as e:
            print(f"{config['desc']:<15} ❌ Failed: {e}")
    
    # Analysis
    if results:
        print(f"\n🏆 Performance Leaders:")
        
        fastest_load = min(results, key=lambda x: x["load_time"])
        print(f"  Fastest loading: {fastest_load['config']} ({fastest_load['load_time']:.2f}s)")
        
        lowest_latency = min(results, key=lambda x: x["latency"])
        print(f"  Lowest latency: {lowest_latency['config']} ({lowest_latency['latency']:.1f}ms)")
        
        highest_throughput = max(results, key=lambda x: x["throughput"])
        print(f"  Highest throughput: {highest_throughput['config']} ({highest_throughput['throughput']:.1f}/sec)")
        
        lowest_memory = min(results, key=lambda x: x["memory"])
        print(f"  Lowest memory: {lowest_memory['config']} ({lowest_memory['memory']:.1f}MB)")
    
    return results

# Run comprehensive benchmark
benchmark_results = comprehensive_benchmark("distilgpt2")
```

### Performance Profiling

```python
import cProfile
import pstats
from io import StringIO

def profile_inference(model_id: str, test_input: Any = "test"):
    """Profile inference performance to identify bottlenecks."""
    
    print(f"🔍 Profiling {model_id} inference")
    
    # Load model
    pipeline = oe.load(model_id, device_preference=["CPU"], dtype="int8")
    
    # Profile inference
    profiler = cProfile.Profile()
    
    print("🏃 Running profiled inference...")
    profiler.enable()
    
    # Run multiple inferences for better profiling
    for i in range(10):
        result = pipeline.infer(test_input)
    
    profiler.disable()
    
    # Analyze results
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    profile_output = s.getvalue()
    
    print("📈 Profiling Results (Top bottlenecks):")
    print(profile_output)
    
    # Save detailed profile
    profile_file = f"/tmp/profile_{model_id.replace('/', '_')}.prof"
    profiler.dump_stats(profile_file)
    print(f"💾 Detailed profile saved to: {profile_file}")
    
    return profile_file

# Example profiling
profile_file = profile_inference("distilgpt2")
```

## Platform-Specific Tuning

### Intel Hardware Optimization

```python
def optimize_for_intel_hardware():
    """Optimize specifically for Intel hardware (CPU, GPU, NPU)."""
    
    print("⚡ Intel Hardware Optimization")
    
    # Check Intel hardware availability
    available_devices = oe.devices()
    intel_devices = {
        "CPU": "Intel CPU" in str(available_devices),
        "GPU": "GPU" in available_devices,  # Assuming Intel GPU
        "NPU": "NPU" in available_devices
    }
    
    print(f"Intel hardware detected: {intel_devices}")
    
    # Device-specific optimizations
    optimizations = {
        "CPU": {
            "threads": "auto",  # Use all CPU cores
            "dtype": "int8",    # Quantization for speed
            "batch_size": 1,    # CPU works best with single samples
            "tips": [
                "Use int8 quantization for 2-4x speedup",
                "Ensure OpenVINO is built with optimized BLAS",
                "Use CPU pinning for consistent performance",
                "Consider model pruning for sparse models"
            ]
        },
        "GPU": {
            "dtype": "fp16",    # GPU optimized precision
            "batch_size": 4,    # GPUs benefit from batching
            "memory_pool": True,
            "tips": [
                "Use fp16 for balance of speed and accuracy",
                "Increase batch size for better GPU utilization",
                "Use dynamic shapes sparingly",
                "Monitor GPU memory usage"
            ]
        },
        "NPU": {
            "dtype": "int8",    # NPU requires quantization
            "batch_size": 1,    # NPU optimized for single inference
            "low_power": True,
            "tips": [
                "Always use int8 quantization for NPU",
                "NPU excels at NLP and small vision models",
                "Use for edge deployment and power efficiency",
                "Check NPU driver status regularly"
            ]
        }
    }
    
    for device, available in intel_devices.items():
        if available and device in optimizations:
            config = optimizations[device]
            print(f"\n🔧 {device} Optimization:")
            print(f"  Recommended dtype: {config['dtype']}")
            print(f"  Optimal batch size: {config['batch_size']}")
            print(f"  Tips:")
            for tip in config["tips"]:
                print(f"    - {tip}")

# Intel-specific model loading
def load_model_for_intel(model_id: str, target_device: str = "auto"):
    """Load model with Intel-specific optimizations."""
    
    if target_device == "auto":
        # Smart device selection for Intel hardware
        available = oe.devices()
        if "NPU" in available:
            target_device = "NPU"
        elif "GPU" in available:
            target_device = "GPU"
        else:
            target_device = "CPU"
    
    # Intel-optimized configuration
    intel_configs = {
        "NPU": {"dtype": "int8", "device_preference": ["NPU"]},
        "GPU": {"dtype": "fp16", "device_preference": ["GPU", "CPU"]},
        "CPU": {"dtype": "int8", "device_preference": ["CPU"]}
    }
    
    config = intel_configs.get(target_device, intel_configs["CPU"])
    
    print(f"🔧 Loading {model_id} for Intel {target_device}")
    oe.load(model_id, **config)
    
    info = oe.get_info()
    print(f"✅ Loaded on {info['device']} with {config['dtype']} precision")
    
    return info

optimize_for_intel_hardware()
intel_pipeline = load_model_for_intel("distilgpt2")
```

### Cloud Platform Optimization

```python
def optimize_for_cloud_platform(platform: str = "aws"):
    """Platform-specific optimizations for major cloud providers."""
    
    print(f"☁️ {platform.upper()} Optimization Guide")
    
    cloud_optimizations = {
        "aws": {
            "recommended_instances": {
                "cpu": ["c5.xlarge", "c5.2xlarge", "c5.4xlarge"],
                "gpu": ["p3.2xlarge", "g4dn.xlarge", "g4dn.2xlarge"],
                "inference": ["inf1.xlarge", "inf1.2xlarge"]
            },
            "storage": "Use EBS gp3 for model storage",
            "networking": "Enable enhanced networking",
            "tips": [
                "Use EC2 placement groups for multi-instance deployments",
                "Leverage ECS/EKS for container orchestration",
                "Use Application Load Balancer for traffic distribution",
                "Consider AWS Inferentia for cost-effective inference"
            ]
        },
        "azure": {
            "recommended_instances": {
                "cpu": ["Standard_F8s_v2", "Standard_F16s_v2"],
                "gpu": ["Standard_NC6s_v3", "Standard_NC12s_v3"],
                "inference": ["Standard_ND6s"]
            },
            "storage": "Use Premium SSD for model storage",
            "networking": "Enable accelerated networking",
            "tips": [
                "Use Azure Container Instances for simple deployments",
                "Leverage AKS for Kubernetes orchestration",
                "Use Azure Load Balancer for traffic distribution",
                "Consider Azure Machine Learning for MLOps"
            ]
        },
        "gcp": {
            "recommended_instances": {
                "cpu": ["n2-standard-4", "n2-standard-8"],
                "gpu": ["n1-standard-4-k80", "n1-standard-8-v100"],
                "inference": ["n2-standard-4-tpu"]
            },
            "storage": "Use SSD persistent disks",
            "networking": "Enable Premium Tier networking",
            "tips": [
                "Use Google Kubernetes Engine for orchestration",
                "Leverage Cloud Run for serverless deployment",
                "Use Cloud Load Balancing for traffic distribution",
                "Consider TPUs for specialized inference workloads"
            ]
        }
    }
    
    if platform not in cloud_optimizations:
        print(f"❌ Platform {platform} not supported")
        return
    
    config = cloud_optimizations[platform]
    
    print(f"\n🏗️ Recommended Instances:")
    for workload, instances in config["recommended_instances"].items():
        print(f"  {workload.title()}: {', '.join(instances)}")
    
    print(f"\n💾 Storage: {config['storage']}")
    print(f"🌐 Networking: {config['networking']}")
    
    print(f"\n💡 Platform Tips:")
    for tip in config["tips"]:
        print(f"  - {tip}")

# Cloud-specific environment variables
def setup_cloud_environment(platform: str):
    """Set up environment variables for cloud deployment."""
    
    cloud_env_vars = {
        "aws": {
            "OE_CACHE_DIR": "/opt/ml/model",
            "OE_DEVICES": "CPU,GPU",
            "OE_DTYPE": "int8",
            "OE_MAX_REQUESTS": "20",
            "OE_TIMEOUT": "30.0"
        },
        "azure": {
            "OE_CACHE_DIR": "/azureml-app/model",
            "OE_DEVICES": "CPU,GPU",
            "OE_DTYPE": "int8", 
            "OE_MAX_REQUESTS": "15",
            "OE_TIMEOUT": "25.0"
        },
        "gcp": {
            "OE_CACHE_DIR": "/gcs/model",
            "OE_DEVICES": "CPU,GPU",
            "OE_DTYPE": "int8",
            "OE_MAX_REQUESTS": "18",
            "OE_TIMEOUT": "28.0"
        }
    }
    
    if platform in cloud_env_vars:
        env_vars = cloud_env_vars[platform]
        print(f"\n🔧 {platform.upper()} Environment Variables:")
        for key, value in env_vars.items():
            print(f"  export {key}={value}")
        return env_vars
    
    return {}

# Example usage
optimize_for_cloud_platform("aws")
aws_env = setup_cloud_environment("aws")
```

## Advanced Optimization

### Model Fusion and Optimization

```python
def advanced_model_optimization():
    """Demonstrate advanced optimization techniques."""
    
    print("🚀 Advanced Model Optimization Techniques")
    
    # 1. Model-specific optimizations
    model_optimizations = {
        "text_generation": {
            "techniques": ["quantization", "pruning", "knowledge_distillation"],
            "optimal_dtype": "int8",
            "optimal_device": "NPU",
            "batch_strategy": "dynamic",
            "description": "Focus on token-level throughput"
        },
        "image_classification": {
            "techniques": ["quantization", "tensor_fusion", "layer_fusion"],
            "optimal_dtype": "int8",
            "optimal_device": "GPU",
            "batch_strategy": "fixed_large",
            "description": "Maximize batch throughput"
        },
        "speech_recognition": {
            "techniques": ["quantization", "streaming_optimization"],
            "optimal_dtype": "int8",
            "optimal_device": "NPU",
            "batch_strategy": "streaming",
            "description": "Optimize for real-time processing"
        }
    }
    
    for task, optimization in model_optimizations.items():
        print(f"\n🎯 {task.replace('_', ' ').title()} Optimization:")
        print(f"  Techniques: {', '.join(optimization['techniques'])}")
        print(f"  Optimal dtype: {optimization['optimal_dtype']}")
        print(f"  Optimal device: {optimization['optimal_device']}")
        print(f"  Batch strategy: {optimization['batch_strategy']}")
        print(f"  Focus: {optimization['description']}")
    
    # 2. Pipeline optimization
    print(f"\n🔧 Pipeline Optimization Techniques:")
    techniques = [
        "Pre-allocate memory buffers",
        "Use async preprocessing",
        "Implement model switching",
        "Cache frequent inputs",
        "Use connection pooling",
        "Optimize I/O operations"
    ]
    
    for i, technique in enumerate(techniques, 1):
        print(f"  {i}. {technique}")

# Custom optimization pipeline
class OptimizedPipeline:
    """Highly optimized pipeline with advanced features."""
    
    def __init__(self, model_id: str, optimization_level: str = "medium"):
        self.model_id = model_id
        self.optimization_level = optimization_level
        
        # Load model with optimization
        self.pipeline = self._load_optimized_model()
        
        # Initialize optimization features
        self.cache = {}
        self.memory_pool = self._init_memory_pool()
        self.preprocessor = self._init_preprocessor()
    
    def _load_optimized_model(self):
        """Load model with optimal configuration."""
        
        # Determine optimal configuration
        config = self._get_optimal_config()
        
        return oe.load(
            self.model_id,
            device_preference=config["devices"],
            dtype=config["dtype"]
        )
    
    def _get_optimal_config(self):
        """Get optimal configuration based on model and system."""
        
        available_devices = oe.devices()
        
        # Model-specific configuration
        if "whisper" in self.model_id.lower():
            config = {
                "devices": ["NPU", "CPU"],
                "dtype": "int8",
                "batch_size": 1
            }
        elif any(term in self.model_id.lower() for term in ["resnet", "vit"]):
            config = {
                "devices": ["GPU", "CPU"],
                "dtype": "int8" if "CPU" in available_devices else "fp16",
                "batch_size": 4
            }
        else:  # Text models
            config = {
                "devices": ["NPU", "GPU", "CPU"],
                "dtype": "int8",
                "batch_size": 1
            }
        
        return config
    
    def _init_memory_pool(self):
        """Initialize memory pool for faster inference."""
        # Simplified memory pool
        return {"allocated": False, "buffers": []}
    
    def _init_preprocessor(self):
        """Initialize optimized preprocessor."""
        return {"initialized": True}
    
    def infer_optimized(self, input_data: Any) -> Any:
        """Run optimized inference."""
        
        # Check cache first
        cache_key = str(hash(str(input_data)))
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Run inference
        result = self.pipeline.infer(input_data)
        
        # Cache result
        if len(self.cache) < 100:  # Limit cache size
            self.cache[cache_key] = result
        
        return result
    
    def benchmark_optimized(self) -> dict:
        """Benchmark the optimized pipeline."""
        
        test_input = "optimization test"
        
        # Standard inference
        start_time = time.time()
        for _ in range(10):
            self.pipeline.infer(test_input)
        standard_time = time.time() - start_time
        
        # Optimized inference
        start_time = time.time()
        for _ in range(10):
            self.infer_optimized(test_input)
        optimized_time = time.time() - start_time
        
        speedup = standard_time / optimized_time if optimized_time > 0 else 1.0
        
        return {
            "standard_time": standard_time,
            "optimized_time": optimized_time,
            "speedup": speedup,
            "cache_hits": len(self.cache)
        }

# Demonstrate advanced optimization
advanced_model_optimization()

print(f"\n🏎️ Testing Optimized Pipeline:")
optimized = OptimizedPipeline("distilgpt2", optimization_level="high")
benchmark_results = optimized.benchmark_optimized()

print(f"📊 Optimization Results:")
print(f"  Standard inference: {benchmark_results['standard_time']:.3f}s")
print(f"  Optimized inference: {benchmark_results['optimized_time']:.3f}s")
print(f"  Speedup: {benchmark_results['speedup']:.1f}x")
print(f"  Cache efficiency: {benchmark_results['cache_hits']} entries")
```

## Monitoring & Debugging

### Real-time Performance Monitoring

```python
import threading
import queue
from dataclasses import dataclass
from typing import List
import json

@dataclass
class PerformanceMetric:
    timestamp: float
    model_id: str
    latency_ms: float
    throughput: float
    memory_mb: float
    device: str
    success: bool

class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self, history_size: int = 1000):
        self.metrics_queue = queue.Queue()
        self.metrics_history: List[PerformanceMetric] = []
        self.history_size = history_size
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start background monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("📊 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        print("📊 Performance monitoring stopped")
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric."""
        if not self.metrics_queue.full():
            self.metrics_queue.put(metric)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # Get metric from queue
                metric = self.metrics_queue.get(timeout=1)
                
                # Add to history
                self.metrics_history.append(metric)
                
                # Maintain history size
                if len(self.metrics_history) > self.history_size:
                    self.metrics_history.pop(0)
                
                # Check for performance issues
                self._check_performance_alerts(metric)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
    
    def _check_performance_alerts(self, metric: PerformanceMetric):
        """Check for performance alerts."""
        
        # Define thresholds
        thresholds = {
            "high_latency": 5000,  # 5 seconds
            "low_throughput": 0.1,  # 0.1 per second
            "high_memory": 8000    # 8GB
        }
        
        alerts = []
        
        if metric.latency_ms > thresholds["high_latency"]:
            alerts.append(f"High latency: {metric.latency_ms:.1f}ms")
        
        if metric.throughput < thresholds["low_throughput"]:
            alerts.append(f"Low throughput: {metric.throughput:.2f}/sec")
        
        if metric.memory_mb > thresholds["high_memory"]:
            alerts.append(f"High memory usage: {metric.memory_mb:.1f}MB")
        
        if not metric.success:
            alerts.append("Inference failed")
        
        if alerts:
            print(f"⚠️ Performance Alert [{metric.model_id}]: {', '.join(alerts)}")
    
    def get_performance_summary(self, last_n: int = 100) -> dict:
        """Get performance summary for last N metrics."""
        
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        recent_metrics = self.metrics_history[-last_n:]
        successful_metrics = [m for m in recent_metrics if m.success]
        
        if not successful_metrics:
            return {"error": "No successful metrics"}
        
        latencies = [m.latency_ms for m in successful_metrics]
        throughputs = [m.throughput for m in successful_metrics]
        memory_usage = [m.memory_mb for m in successful_metrics]
        
        return {
            "total_requests": len(recent_metrics),
            "successful_requests": len(successful_metrics),
            "success_rate": len(successful_metrics) / len(recent_metrics),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "max_latency_ms": max(latencies),
            "min_latency_ms": min(latencies),
            "avg_throughput": sum(throughputs) / len(throughputs),
            "avg_memory_mb": sum(memory_usage) / len(memory_usage),
            "timestamp": time.time()
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to file for analysis."""
        
        metrics_data = []
        for metric in self.metrics_history:
            metrics_data.append({
                "timestamp": metric.timestamp,
                "model_id": metric.model_id,
                "latency_ms": metric.latency_ms,
                "throughput": metric.throughput,
                "memory_mb": metric.memory_mb,
                "device": metric.device,
                "success": metric.success
            })
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"📁 Metrics exported to {filepath}")

# Monitored pipeline wrapper
class MonitoredPipeline:
    """Pipeline wrapper with automatic performance monitoring."""
    
    def __init__(self, model_id: str, monitor: PerformanceMonitor, **load_kwargs):
        self.model_id = model_id
        self.monitor = monitor
        self.pipeline = oe.load(model_id, **load_kwargs)
    
    def infer(self, input_data: Any) -> Any:
        """Run inference with automatic monitoring."""
        
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            result = self.pipeline.infer(input_data)
            
            inference_time = time.time() - start_time
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Record metric
            metric = PerformanceMetric(
                timestamp=time.time(),
                model_id=self.model_id,
                latency_ms=inference_time * 1000,
                throughput=1.0 / inference_time if inference_time > 0 else 0,
                memory_mb=memory_after,
                device=self.pipeline.device,
                success=True
            )
            
            self.monitor.record_metric(metric)
            return result
            
        except Exception as e:
            inference_time = time.time() - start_time
            
            # Record failed metric
            metric = PerformanceMetric(
                timestamp=time.time(),
                model_id=self.model_id,
                latency_ms=inference_time * 1000,
                throughput=0,
                memory_mb=memory_before,
                device=getattr(self.pipeline, 'device', 'unknown'),
                success=False
            )
            
            self.monitor.record_metric(metric)
            raise

# Demonstration
def demo_performance_monitoring():
    """Demonstrate real-time performance monitoring."""
    
    print("📊 Real-time Performance Monitoring Demo")
    
    # Create monitor
    monitor = PerformanceMonitor(history_size=500)
    monitor.start_monitoring()
    
    # Create monitored pipeline
    monitored_pipeline = MonitoredPipeline("distilgpt2", monitor, dtype="int8")
    
    # Run test inferences
    test_inputs = [
        "First test for monitoring",
        "Second test with different length input",
        "Third test to see performance patterns",
    ]
    
    print(f"\n🔄 Running {len(test_inputs)} test inferences...")
    
    for i, test_input in enumerate(test_inputs):
        result = monitored_pipeline.infer(test_input)
        print(f"  Inference {i+1}: Completed")
        time.sleep(0.1)  # Small delay
    
    # Get performance summary
    time.sleep(0.5)  # Let monitoring catch up
    summary = monitor.get_performance_summary()
    
    print(f"\n📈 Performance Summary:")
    print(f"  Total requests: {summary['total_requests']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    print(f"  Average latency: {summary['avg_latency_ms']:.1f}ms")
    print(f"  Min/Max latency: {summary['min_latency_ms']:.1f}ms / {summary['max_latency_ms']:.1f}ms")
    print(f"  Average throughput: {summary['avg_throughput']:.2f}/sec")
    print(f"  Average memory: {summary['avg_memory_mb']:.1f}MB")
    
    # Export metrics
    monitor.export_metrics("/tmp/performance_metrics.json")
    
    # Stop monitoring
    monitor.stop_monitoring()

demo_performance_monitoring()
```

## Conclusion

This performance tuning guide provides comprehensive strategies for optimizing OpenVINO-Easy in production environments. Key takeaways:

### 🎯 **Essential Optimization Steps**

1. **Establish Baselines**: Always measure before optimizing
2. **Choose Optimal Devices**: NPU for NLP, GPU for vision, CPU for compatibility
3. **Use Quantization**: INT8 for speed, FP16 for GPU balance
4. **Implement Caching**: Cache models, preprocessing, and frequent results
5. **Monitor Continuously**: Track performance metrics in real-time

### 🚀 **Performance Multipliers**

- **Quantization**: 2-4x speedup with minimal quality loss
- **Device Optimization**: 3-10x improvement with NPU/GPU
- **Intelligent Caching**: 5-50x speedup for repeated inputs
- **Batching**: 2-8x throughput improvement for high-volume scenarios
- **Memory Optimization**: Reduced latency and resource usage

### 📊 **Monitoring Strategy**

- Set up real-time performance monitoring
- Define performance thresholds and alerts
- Export metrics for historical analysis
- Use profiling to identify bottlenecks
- Implement automated optimization adjustments

Remember: **measure, optimize, validate, repeat**. Performance tuning is an iterative process that requires continuous monitoring and adjustment based on your specific use case and requirements.

For implementation examples, see the [Production Deployment Guide](production_deployment.md) and [Examples](../examples/) directory.