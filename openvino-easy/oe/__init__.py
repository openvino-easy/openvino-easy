"""OpenVINO-Easy: Framework-agnostic Python wrapper for OpenVINO 2024."""

# Runtime guard - check OpenVINO availability immediately
try:
    import openvino as ov
except ImportError:
    # Dynamic installation hints based on available extras
    _INSTALL_VARIANTS = {
        "cpu": "CPU-only (40MB)",
        "runtime": "CPU runtime (40MB)", 
        "gpu": "Intel GPU support",
        "npu": "Intel NPU support",
        "quant": "With INT8 quantization",
        "full": "Full dev environment (~1GB)"
    }
    
    install_hints = []
    for variant, desc in _INSTALL_VARIANTS.items():
        install_hints.append(f"  • {desc}: pip install 'openvino-easy[{variant}]'")
    
    raise ImportError(
        "OpenVINO runtime not found. Install OpenVINO-Easy with hardware-specific extras:\n" +
        "\n".join(install_hints) +
        "\n\nFor more help: https://github.com/example/openvino-easy#installation"
    )

# Version compatibility check
import warnings
_RECOMMENDED_OV_VERSION = "2024"
try:
    _current_version = ov.__version__
    _major_version = _current_version.split('.')[0]
    if _major_version not in ["2024", "2025"]:
        warnings.warn(
            f"OpenVINO {_RECOMMENDED_OV_VERSION}.x recommended, found {_current_version}. "
            f"Some features may not work correctly. Consider upgrading:\n"
            f"  pip install --upgrade 'openvino>={_RECOMMENDED_OV_VERSION}.0,<2026.0'",
            UserWarning,
            stacklevel=2
        )
except Exception:
    # If version detection fails, continue silently
    pass

from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from .loader import load_model
from .quant import quantize_model
from .runtime import RuntimeWrapper
from .benchmark import benchmark_pipeline
from ._core import detect_device, get_available_devices

__version__ = "0.1.0"

class Pipeline:
    """OpenVINO-Easy pipeline for model inference and benchmarking."""
    
    def __init__(self, compiled_model, device: str, model_path: str, model_info: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline.
        
        Args:
            compiled_model: Compiled OpenVINO model
            device: Device name (NPU, GPU, CPU, etc.)
            model_path: Path to the model file
            model_info: Optional model metadata including source path
        """
        self.compiled_model = compiled_model
        self.device = device
        self.model_path = model_path
        self.model_info = model_info or {}
        
        # Initialize runtime wrapper with model info
        self.runtime = RuntimeWrapper(compiled_model, device, model_info)
    
    def infer(self, input_data, **kwargs):
        """
        Run inference on the model.
        
        Args:
            input_data: Input data (string, numpy array, or dict)
            **kwargs: Additional inference parameters
            
        Returns:
            Model output
        """
        return self.runtime.infer(input_data, **kwargs)
    
    def benchmark(self, warmup_runs=5, benchmark_runs=20, **kwargs):
        """
        Benchmark the model performance.
        
        Args:
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs
            **kwargs: Additional benchmark parameters
            
        Returns:
            Benchmark results dictionary
        """
        return benchmark_pipeline(self, warmup_runs, benchmark_runs, **kwargs)
    
    def get_info(self):
        """Get model and runtime information."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "runtime_info": self.runtime.get_model_info(),
            **self.model_info
        }

def load(model_id_or_path: str, 
         device_preference: Optional[List[str]] = None,
         dtype: str = "fp16",
         cache_dir: Optional[str] = None,
         **kwargs) -> Pipeline:
    """
    Load a model from Hugging Face Hub, ONNX, or OpenVINO IR.
    
    Args:
        model_id_or_path: Model identifier or path
        device_preference: List of preferred devices (e.g., ["NPU", "GPU", "CPU"])
        dtype: Model precision ("fp16", "int8")
        cache_dir: Directory for caching models
        **kwargs: Additional loading parameters
        
    Returns:
        Pipeline object for inference and benchmarking
    """
    # Detect best available device
    if device_preference is None:
        device_preference = ["NPU", "GPU", "CPU"]
    
    available_devices = get_available_devices()
    device = None
    for preferred in device_preference:
        if preferred in available_devices:
            device = preferred
            break
    
    if device is None:
        device = "CPU"  # Fallback
    
    # Load the model
    model = load_model(model_id_or_path, dtype, cache_dir or "~/.cache/oe")
    
    # Quantize if requested
    if dtype == "int8":
        model = quantize_model(model, dtype, cache_dir or "~/.cache/oe")
    
    # Compile the model
    core = ov.Core()
    compiled_model = core.compile_model(model, device)
    
    # Create model info with source path for tokenizer
    model_info = {
        "source_path": model_id_or_path,  # Pass original path for tokenizer
        "dtype": dtype,
        "quantized": dtype == "int8",
        "device": device
    }
    
    return Pipeline(compiled_model, device, model_id_or_path, model_info)

def devices() -> List[str]:
    """
    Get list of available devices.
    
    Returns:
        List of available device names
    """
    return get_available_devices()

# For backward compatibility
def detect_best_device() -> str:
    """Detect the best available device."""
    return detect_device() 