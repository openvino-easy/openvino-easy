"""OpenVINO-Easy: Framework-agnostic Python wrapper for OpenVINO 2025."""

# Lazy OpenVINO availability check so `import oe` works without OpenVINO
try:
    import openvino as ov  # type: ignore
    _OV_AVAILABLE = True
except Exception:
    ov = None  # type: ignore
    _OV_AVAILABLE = False

import warnings

_RECOMMENDED_OV_VERSION = "2025.2"
if _OV_AVAILABLE:
    try:
        _current_version = ov.__version__  # type: ignore[attr-defined]
        version_parts = _current_version.split(".")
        if len(version_parts) >= 2:
            version_key = f"{version_parts[0]}.{version_parts[1]}"
            if version_key not in ["2025.0", "2025.1", "2025.2"]:
                warnings.warn(
                    f"OpenVINO {_RECOMMENDED_OV_VERSION} recommended, found {_current_version}. "
                    f"Some features may not work correctly. Consider upgrading:\n"
                    f"  pip install --upgrade 'openvino>={_RECOMMENDED_OV_VERSION},<2026.0'",
                    UserWarning,
                    stacklevel=2,
                )
    except Exception:
        pass

from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import logging
import numpy as np
import os
from .benchmark import benchmark_pipeline

__version__ = "1.0.0"

# Global state for the 3-function API
_current_pipeline = None


def _validate_cache_path(original_path) -> None:
    """Validate that cache path is safe and doesn't escape expected boundaries."""
    try:
        # Convert to string if it's a Path object
        path_str = str(original_path)
        
        # Check original path string for dangerous patterns before resolution
        dangerous_patterns = ["..", "~/", "${", "`", "%", "$("]
        for pattern in dangerous_patterns:
            if pattern in path_str:
                raise ValueError(
                    f"Unsafe path pattern '{pattern}' detected in: {original_path}"
                )

        # Additional checks for common attack patterns
        if path_str.startswith("/"):
            # Unix absolute paths - check for system directories
            forbidden_starts = [
                "/bin",
                "/sbin",
                "/usr",
                "/etc",
                "/var",
                "/sys",
                "/proc",
                "/dev",
            ]
            for forbidden in forbidden_starts:
                if path_str.startswith(forbidden):
                    raise ValueError(
                        f"Cannot use system directory as cache: {original_path}"
                    )

        elif len(path_str) > 1 and path_str[1] == ":":
            # Windows absolute paths - check for system drives/directories
            if path_str.upper().startswith(
                ("C:\\WINDOWS", "C:\\PROGRAM", "C:\\SYSTEM")
            ):
                raise ValueError(
                    f"Cannot use system directory as cache: {original_path}"
                )

        # Resolve and check final path
        resolved_path = Path(original_path).expanduser().resolve()
        resolved_str = str(resolved_path).lower()

        # Final safety check on resolved path
        dangerous_resolved = [
            "windows",
            "system32",
            "program files",
            "/bin",
            "/sbin",
            "/usr",
        ]
        for dangerous in dangerous_resolved:
            if (
                dangerous in resolved_str and len(resolved_str.split(os.sep)) <= 4
            ):  # Not too deep
                raise ValueError(
                    f"Resolved path points to system directory: {resolved_path}"
                )

    except ValueError:
        raise  # Re-raise our validation errors
    except Exception as e:
        raise ValueError(f"Invalid cache path: {e}") from e


def _get_default_models_dir() -> Path:
    """Get platform-aware default models directory."""
    # Environment variable override
    if "OE_MODELS_DIR" in os.environ:
        env_path_str = os.environ["OE_MODELS_DIR"]
        _validate_cache_path(env_path_str)
        return Path(env_path_str).expanduser().resolve()

    # Platform-specific defaults
    if os.name == "nt":  # Windows
        base_dir = Path(
            os.environ.get("LOCALAPPDATA", os.path.expanduser("~\\AppData\\Local"))
        )
        return base_dir / "openvino-easy"
    else:  # Unix-like (Linux, macOS)
        return Path.home() / ".openvino-easy"


def _get_models_dir(cache_dir: Optional[str] = None) -> Path:
    """Get the models directory, with optional override."""
    if cache_dir:
        # Security: Validate the original path string before processing
        _validate_cache_path(cache_dir)
        base_dir = Path(cache_dir).expanduser().resolve()
    else:
        base_dir = _get_default_models_dir()

    return base_dir / "models"


def _get_cache_dir(cache_dir: Optional[str] = None) -> Path:
    """Get the temporary cache directory."""
    if cache_dir:
        # Security: Validate the original path string before processing
        _validate_cache_path(cache_dir)
        base_dir = Path(cache_dir).expanduser().resolve()
    else:
        base_dir = _get_default_models_dir()

    return base_dir / "cache"


class Pipeline:
    """OpenVINO-Easy pipeline for model inference and benchmarking."""

    def __init__(
        self,
        compiled_model,
        device: str,
        model_path: str,
        model_info: Optional[Dict[str, Any]] = None,
    ):
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

        # Initialize runtime wrapper with model info (imported lazily)
        from .runtime import RuntimeWrapper

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

    async def infer_async(self, input_data, **kwargs):
        """
        Run asynchronous inference for better throughput.

        Args:
            input_data: Input data (string, numpy array, or dict)
            **kwargs: Additional inference parameters

        Returns:
            Model output (awaitable)
        """
        return await self.runtime.infer_async(input_data, **kwargs)

    def infer_batch(self, input_batch, max_workers=None, **kwargs):
        """
        Run batch inference with thread pool for better throughput.

        Args:
            input_batch: List of input data
            max_workers: Maximum number of worker threads (auto-detected if None)
            **kwargs: Additional inference parameters

        Returns:
            List of model outputs in the same order as inputs
        """
        return self.runtime.infer_batch(input_batch, max_workers, **kwargs)

    async def infer_batch_async(self, input_batch, max_concurrent=None, **kwargs):
        """
        Run asynchronous batch inference with controlled concurrency.

        Args:
            input_batch: List of input data
            max_concurrent: Maximum number of concurrent inference requests (auto-detected if None)
            **kwargs: Additional inference parameters

        Returns:
            List of model outputs in the same order as inputs
        """
        return await self.runtime.infer_batch_async(
            input_batch, max_concurrent, **kwargs
        )

    def get_info(self):
        """Get comprehensive model and runtime information."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "runtime_info": self.runtime.get_model_info(),
            **self.model_info,
        }

    def get_performance_stats(self):
        """Get detailed performance statistics."""
        return self.runtime.get_pool_stats()

    def clear_cache(self):
        """Clear preprocessing cache to free memory."""
        return self.runtime.clear_cache()


def load(
    model_id_or_path: str,
    device_preference: Optional[List[str]] = None,
    dtype: str = "fp16",
    cache_dir: Optional[str] = None,
    retry_on_failure: bool = True,
    fallback_device: str = "CPU",
    **kwargs,
) -> "_ModuleContextManager":
    """
    Load a model from Hugging Face Hub, ONNX, or OpenVINO IR.

    This is the FIRST function in the 3-function experience:
    1. oe.load() - Load model
    2. oe.infer() - Run inference
    3. oe.benchmark() - Measure performance

    Args:
        model_id_or_path: Model identifier or path
        device_preference: List of preferred devices (e.g., ["NPU", "GPU", "CPU"])
        dtype: Model precision ("fp16", "int8", "fp16-nf4")
        cache_dir: Directory for caching models
        **kwargs: Additional loading parameters

    Note:
        FP16-NF4 precision is supported on Arrow Lake/Lunar Lake NPUs with OpenVINO 2025.2+
        After calling load(), use oe.infer() and oe.benchmark() directly.
    """
    global _current_pipeline

    # Detect best available device with enhanced fallback
    if device_preference is None:
        device_preference = ["NPU", "GPU", "CPU"]

    # Import heavy dependencies lazily
    if not _OV_AVAILABLE:
        _raise_openvino_missing()

    from .loader import load_model
    from .quant import quantize_model
    from ._core import get_available_devices, check_npu_driver

    available_devices = get_available_devices()
    device = None
    attempted_devices = []

    for preferred in device_preference:
        if preferred in available_devices:
            device = preferred
            break
        attempted_devices.append(preferred)

    if device is None:
        if fallback_device in available_devices:
            device = fallback_device
            logging.warning(
                f"Preferred devices {attempted_devices} not available, using fallback: {device}"
            )
        else:
            device = "CPU"  # Ultimate fallback
            logging.warning("No preferred devices available, using CPU")

    # Load the model (use models directory)
    models_dir = _get_models_dir(cache_dir)
    model = load_model(model_id_or_path, dtype, str(models_dir))

    # Create model info early for quantization
    model_info = {
        "source_path": model_id_or_path,  # Pass original path for tokenizer
        "dtype": dtype,
        "quantized": dtype in ["int8", "fp16-nf4"],
        "device": device,
    }

    # Check for FP16-NF4 precision support on NPU
    if dtype == "fp16-nf4":
        if device == "NPU":
            npu_status = check_npu_driver()
            if npu_status.get("npu_functional") and npu_status.get(
                "capabilities", {}
            ).get("supports_fp16_nf4", False):
                logging.info(
                    f"FP16-NF4 precision supported on {npu_status.get('npu_generation', 'unknown')} NPU"
                )
            else:
                warnings.warn(
                    f"FP16-NF4 precision requested but NPU doesn't support it. "
                    f"NPU generation: {npu_status.get('npu_generation', 'unknown')}. "
                    f"Falling back to FP16.",
                    UserWarning,
                )
                dtype = "fp16"
                model_info["dtype"] = dtype
                model_info["quantized"] = False
        else:
            warnings.warn(
                f"FP16-NF4 precision is only supported on NPU devices. "
                f"Current device: {device}. Falling back to FP16.",
                UserWarning,
            )
            dtype = "fp16"
            model_info["dtype"] = dtype
            model_info["quantized"] = False

    # Quantize if requested
    if dtype == "int8":
        model = quantize_model(
            model, dtype=dtype, cache_dir=str(models_dir), model_info=model_info
        )

    # Compile the model with enhanced error recovery
    import openvino as ov  # Local import to ensure availability
    core = ov.Core()
    compile_config = {}
    compiled_model = None
    compilation_error = None

    # Configure FP16-NF4 if supported
    if dtype == "fp16-nf4" and device == "NPU":
        compile_config["NPU_COMPILATION_MODE_PARAMS"] = "fp16-nf4"

    # Try to compile with retry and fallback logic
    devices_to_try = [device]
    if retry_on_failure and device != fallback_device:
        devices_to_try.append(fallback_device)

    for attempt_device in devices_to_try:
        try:
            logging.info(f"Attempting to compile model on {attempt_device}...")
            compiled_model = core.compile_model(model, attempt_device, compile_config)
            device = attempt_device  # Update device to successful one
            break
        except Exception as e:
            compilation_error = e
            logging.warning(f"Failed to compile on {attempt_device}: {e}")

            # Try with simpler config on retry
            if attempt_device == device and compile_config:
                try:
                    logging.info(f"Retrying {attempt_device} with default config...")
                    compiled_model = core.compile_model(model, attempt_device, {})
                    device = attempt_device
                    logging.warning(
                        "Succeeded with default config, some optimizations disabled"
                    )
                    break
                except Exception as e2:
                    logging.warning(f"Retry with default config also failed: {e2}")

    if compiled_model is None:
        raise RuntimeError(
            f"Failed to compile model on any device. "
            f"Tried: {devices_to_try}. "
            f"Last error: {compilation_error}"
        )

    # Create and store the pipeline globally
    _current_pipeline = Pipeline(compiled_model, device, model_id_or_path, model_info)

    logging.info(f"Model loaded on {device}. Use oe.infer() and oe.benchmark().")

    # Return the module itself to enable context manager usage
    return _ModuleContextManager()


def infer(input_data: Union[str, np.ndarray, Dict[str, Any]], **kwargs) -> Any:
    """
    Run inference on the currently loaded model.

    This is the SECOND function in the 3-function experience.

    Args:
        input_data: Input data (string, numpy array, or dict)
        **kwargs: Additional inference parameters

    Returns:
        Model output

    Raises:
        RuntimeError: If no model is loaded
    """
    if _current_pipeline is None:
        raise RuntimeError(
            "No model loaded. Call oe.load('model-name') first.\n"
            "Example: oe.load('microsoft/DialoGPT-medium')"
        )

    return _current_pipeline.infer(input_data, **kwargs)


def benchmark(
    warmup_runs: int = 5, benchmark_runs: int = 20, **kwargs
) -> Dict[str, Any]:
    """
    Benchmark the currently loaded model's performance.

    This is the THIRD function in the 3-function experience.

    Args:
        warmup_runs: Number of warmup runs
        benchmark_runs: Number of benchmark runs
        **kwargs: Additional benchmark parameters

    Returns:
        Benchmark results dictionary with latency, throughput, etc.

    Raises:
        RuntimeError: If no model is loaded
    """
    if _current_pipeline is None:
        raise RuntimeError(
            "No model loaded. Call oe.load('model-name') first.\n"
            "Example: oe.load('microsoft/DialoGPT-medium')"
        )

    return _current_pipeline.benchmark(warmup_runs, benchmark_runs, **kwargs)


def unload() -> None:
    """
    Unload the currently loaded model and free memory.

    This is the optional cleanup function.

    After calling unload(), you need to call oe.load() again before
    using oe.infer() or oe.benchmark().
    """
    global _current_pipeline

    if _current_pipeline is None:
        logging.info("No model currently loaded.")
        return

    _current_pipeline.runtime.unload()
    _current_pipeline = None
    logging.info("Model unloaded successfully.")


def is_loaded() -> bool:
    """
    Check if a model is currently loaded.

    Returns:
        True if model is loaded, False otherwise
    """
    return _current_pipeline is not None and _current_pipeline.runtime.is_loaded()


def get_info() -> Dict[str, Any]:
    """
    Get information about the currently loaded model.

    Returns:
        Model information dictionary

    Raises:
        RuntimeError: If no model is loaded
    """
    if _current_pipeline is None:
        raise RuntimeError("No model loaded. Call oe.load('model-name') first.")

    return _current_pipeline.get_info()


class _ModuleContextManager:
    """Context manager for automatic model cleanup."""

    def __enter__(self):
        # Return the module functions for use in context
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unload()
        return False

    def infer(self, input_data, **kwargs):
        """Context manager wrapper for infer."""
        return infer(input_data, **kwargs)

    def benchmark(self, warmup_runs=5, benchmark_runs=20, **kwargs):
        """Context manager wrapper for benchmark."""
        return benchmark(warmup_runs, benchmark_runs, **kwargs)

    def get_info(self):
        """Context manager wrapper for get_info."""
        return get_info()

    def unload(self):
        """Context manager wrapper for unload."""
        return unload()


def devices() -> List[str]:
    """
    Get list of available devices.

    Returns:
        List of available device names
    """
    try:
        from ._core import get_available_devices
        return get_available_devices()
    except Exception:
        # If OpenVINO is not installed, return CPU as a minimal safe default
        return ["CPU"]


# For backward compatibility
def detect_best_device() -> str:
    """Detect the best available device."""
    try:
        from ._core import detect_device
        return detect_device()
    except Exception:
        return "CPU"


# Cache management namespaces
class _ModelsNamespace:
    """Comprehensive models management namespace."""

    @staticmethod
    def dir(cache_dir: Optional[str] = None) -> str:
        """Get the models directory path."""
        return str(_get_models_dir(cache_dir))

    @staticmethod
    def search(
        query: str, limit: int = 10, model_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for models on Hugging Face Hub.

        Args:
            query: Search query string
            limit: Maximum number of results (default: 10)
            model_type: Filter by model type (text, image, audio, etc.)

        Returns:
            List of model information dictionaries
        """
        try:
            from huggingface_hub import HfApi

            api = HfApi()

            # Build filter parameters
            filter_params = {}
            if model_type:
                # Map our types to HF pipeline tags
                type_mapping = {
                    "text": [
                        "text-generation",
                        "text2text-generation",
                        "conversational",
                    ],
                    "image": [
                        "text-to-image",
                        "image-classification",
                        "object-detection",
                    ],
                    "audio": [
                        "automatic-speech-recognition",
                        "text-to-speech",
                        "audio-classification",
                    ],
                    "vision": [
                        "image-classification",
                        "object-detection",
                        "image-segmentation",
                    ],
                }
                if model_type in type_mapping:
                    filter_params["pipeline_tag"] = type_mapping[model_type]

            # Search models
            models = api.list_models(search=query, limit=limit, **filter_params)

            results = []
            for model in models:
                model_info = {
                    "id": model.id,
                    "author": model.id.split("/")[0] if "/" in model.id else "unknown",
                    "name": model.id.split("/")[-1],
                    "downloads": getattr(model, "downloads", 0),
                    "likes": getattr(model, "likes", 0),
                    "pipeline_tag": getattr(model, "pipeline_tag", "unknown"),
                    "tags": getattr(model, "tags", []),
                    "created_at": str(getattr(model, "created_at", "unknown")),
                    "last_modified": str(getattr(model, "last_modified", "unknown")),
                    "private": getattr(model, "private", False),
                    "gated": getattr(model, "gated", False),
                }
                results.append(model_info)

            return results

        except ImportError:
            raise RuntimeError(
                "Hugging Face Hub client required for model search. "
                "Install with: pip install 'openvino-easy[full]'"
            )
        except Exception as e:
            raise RuntimeError(f"Model search failed: {e}")

    def info(self, model_id: str, cache_dir: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information about a model (local or remote).

        Args:
            model_id: Model identifier or path
            cache_dir: Optional custom cache directory

        Returns:
            Comprehensive model information
        """
        info = {
            "id": model_id,
            "local": False,
            "remote": False,
            "local_info": None,
            "remote_info": None,
            "compatible": True,
            "requirements": {
                "min_memory_mb": "unknown",
                "recommended_devices": ["CPU", "GPU", "NPU"],
                "supported_precisions": ["fp16", "int8"],
            },
        }

        # Check if model exists locally
        local_models = self.list(cache_dir)
        for local_model in local_models:
            if model_id in local_model["name"]:
                info["local"] = True
                info["local_info"] = local_model
                break

        # Get remote information
        try:
            from huggingface_hub import HfApi, model_info as hf_model_info

            remote_info = hf_model_info(model_id)
            info["remote"] = True
            info["remote_info"] = {
                "id": remote_info.id,
                "downloads": getattr(remote_info, "downloads", 0),
                "likes": getattr(remote_info, "likes", 0),
                "pipeline_tag": getattr(remote_info, "pipeline_tag", "unknown"),
                "tags": getattr(remote_info, "tags", []),
                "library_name": getattr(remote_info, "library_name", "unknown"),
                "created_at": str(getattr(remote_info, "created_at", "unknown")),
                "last_modified": str(getattr(remote_info, "last_modified", "unknown")),
                "private": getattr(remote_info, "private", False),
                "gated": getattr(remote_info, "gated", False),
            }

            # Estimate requirements based on tags and type
            if "stable-diffusion" in info["remote_info"]["tags"]:
                info["requirements"]["min_memory_mb"] = 4000
                info["requirements"]["recommended_devices"] = ["NPU", "GPU", "CPU"]
            elif any(
                tag in info["remote_info"]["tags"] for tag in ["llm", "text-generation"]
            ):
                info["requirements"]["min_memory_mb"] = 2000
                info["requirements"]["recommended_devices"] = ["NPU", "GPU", "CPU"]
            else:
                info["requirements"]["min_memory_mb"] = 1000

        except Exception as e:
            info["remote_error"] = str(e)

        return info

    def install(
        self,
        model_id: str,
        dtype: str = "fp16",
        cache_dir: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Download and convert a model without loading it.

        Args:
            model_id: Model identifier to install
            dtype: Model precision (fp16, int8, etc.)
            cache_dir: Optional custom cache directory
            force: Force reinstall if model exists

        Returns:
            Installation result dictionary
        """
        # Check if already installed
        if not force:
            local_models = self.list(cache_dir)
            for model in local_models:
                # Convert model_id format (test/model -> test--model)
                normalized_model_id = model_id.replace("/", "--")
                if normalized_model_id in model["name"] and dtype in model["name"]:
                    return {
                        "installed": False,
                        "already_exists": True,
                        "model_name": model["name"],
                        "size_mb": model["size_mb"],
                        "message": f"Model '{model_id}' with {dtype} precision already installed. Use force=True to reinstall.",
                    }

        try:
            # Use the existing load_model function but don't compile
            models_dir = _get_models_dir(cache_dir)

            logging.info(f"Installing model: {model_id} ({dtype})")
            # Call module-level proxy so tests can patch `oe.load_model`
            model = load_model(model_id, dtype, str(models_dir))

            # Get size of installed model
            installed_models = self.list(cache_dir)
            for installed_model in installed_models:
                if (
                    model_id in installed_model["name"]
                    and dtype in installed_model["name"]
                ):
                    return {
                        "installed": True,
                        "model_name": installed_model["name"],
                        "model_id": model_id,
                        "dtype": dtype,
                        "size_mb": installed_model["size_mb"],
                        "files": len(installed_model.get("files", [])),
                        "message": f"Successfully installed '{model_id}' with {dtype} precision",
                    }

            return {
                "installed": True,
                "model_id": model_id,
                "dtype": dtype,
                "message": f"Successfully installed '{model_id}' with {dtype} precision",
            }

        except Exception as e:
            return {
                "installed": False,
                "error": str(e),
                "model_id": model_id,
                "dtype": dtype,
                "message": f"Failed to install '{model_id}': {e}",
            }

    def validate(
        self, model_name: str = None, cache_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate model integrity and compatibility.

        Args:
            model_name: Specific model to validate (if None, validates all)
            cache_dir: Optional custom cache directory

        Returns:
            Validation results
        """
        results = {"validated": 0, "passed": 0, "failed": 0, "models": []}

        models_to_check = self.list(cache_dir)
        if model_name:
            models_to_check = [m for m in models_to_check if model_name in m["name"]]

        for model in models_to_check:
            results["validated"] += 1
            model_result = {
                "name": model["name"],
                "valid": False,
                "errors": [],
                "warnings": [],
            }

            try:
                # Check if model files exist
                model_path = Path(model["path"])
                if not model_path.exists():
                    model_result["errors"].append("Model directory not found")
                else:
                    # Look for essential files
                    xml_files = list(model_path.glob("*.xml"))
                    bin_files = list(model_path.glob("*.bin"))

                    if not xml_files:
                        model_result["errors"].append("No .xml files found")
                    if not bin_files:
                        model_result["warnings"].append(
                            "No .bin files found (may be embedded)"
                        )

                    # Try to load the model to verify it's readable
                    if xml_files:
                        try:
                            import openvino as ov

                            core = ov.Core()
                            test_model = core.read_model(str(xml_files[0]))

                            # Basic validation
                            if len(test_model.inputs) == 0:
                                model_result["errors"].append("Model has no inputs")
                            if len(test_model.outputs) == 0:
                                model_result["errors"].append("Model has no outputs")

                        except Exception as e:
                            model_result["errors"].append(f"Failed to load model: {e}")

                # Determine if valid
                model_result["valid"] = len(model_result["errors"]) == 0
                if model_result["valid"]:
                    results["passed"] += 1
                else:
                    results["failed"] += 1

            except Exception as e:
                model_result["errors"].append(f"Validation error: {e}")
                results["failed"] += 1

            results["models"].append(model_result)

        return results

    def benchmark_all(
        self, cache_dir: Optional[str] = None, warmup: int = 3, runs: int = 10
    ) -> Dict[str, Any]:
        """Benchmark all installed models.

        Args:
            cache_dir: Optional custom cache directory
            warmup: Number of warmup runs per model
            runs: Number of benchmark runs per model

        Returns:
            Benchmark results for all models
        """
        models_list = self.list(cache_dir)

        results = {
            "total_models": len(models_list),
            "benchmarked": 0,
            "failed": 0,
            "results": [],
            "summary": {"fastest_model": None, "slowest_model": None, "average_fps": 0},
        }

        total_fps = 0
        fastest_fps = 0
        slowest_fps = float("inf")

        for model in models_list:
            try:
                # Extract model info from directory name
                name_parts = model["name"].split("--")
                if len(name_parts) >= 3:
                    model_id = "--".join(name_parts[:-2])  # Remove dtype and hash
                    dtype = name_parts[-2]
                else:
                    continue  # Skip models we can't parse

                logging.info(f"Benchmarking {model_id}...")

                # Load and benchmark
                load(model_id, dtype=dtype, cache_dir=cache_dir)
                bench_result = benchmark(warmup_runs=warmup, benchmark_runs=runs)
                unload()

                # Store results
                model_result = {
                    "model_id": model_id,
                    "model_name": model["name"],
                    "dtype": dtype,
                    "size_mb": model["size_mb"],
                    "benchmark": bench_result,
                }

                results["results"].append(model_result)
                results["benchmarked"] += 1

                # Update summary stats
                fps = bench_result.get("fps", 0)
                total_fps += fps

                if fps > fastest_fps:
                    fastest_fps = fps
                    results["summary"]["fastest_model"] = {
                        "id": model_id,
                        "fps": fps,
                        "device": bench_result.get("device", "unknown"),
                    }

                if fps < slowest_fps:
                    slowest_fps = fps
                    results["summary"]["slowest_model"] = {
                        "id": model_id,
                        "fps": fps,
                        "device": bench_result.get("device", "unknown"),
                    }

            except Exception as e:
                results["failed"] += 1
                results["results"].append(
                    {"model_name": model["name"], "error": str(e), "benchmarked": False}
                )
                logging.warning(f"Failed to benchmark {model['name']}: {e}")

        # Calculate averages
        if results["benchmarked"] > 0:
            results["summary"]["average_fps"] = round(
                total_fps / results["benchmarked"], 2
            )

        return results

    @staticmethod
    def list(cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all cached models with their metadata."""
        models_path = _get_models_dir(cache_dir)

        if not models_path.exists():
            return []

        models = []
        for model_dir in models_path.iterdir():
            if not model_dir.is_dir():
                continue

            model_info = {
                "name": model_dir.name,
                "path": str(model_dir),
                "size_mb": 0,
                "files": [],
            }

            # Calculate total size and list files
            try:
                for file_path in model_dir.rglob("*"):
                    if file_path.is_file():
                        size = file_path.stat().st_size
                        model_info["size_mb"] += size / (1024 * 1024)
                        model_info["files"].append(file_path.name)
            except (PermissionError, OSError):
                model_info["size_mb"] = 0
                model_info["files"] = ["<access denied>"]

            # Load metadata if available
            metadata_file = model_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    import json

                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    model_info.update(metadata)
                except (json.JSONDecodeError, IOError):
                    pass

            models.append(model_info)

        # Sort by size descending
        return sorted(models, key=lambda x: x["size_mb"], reverse=True)

    @staticmethod
    def remove(
        model_name: str, cache_dir: Optional[str] = None, confirm: bool = True
    ) -> Dict[str, Any]:
        """Remove a specific model from cache.

        Args:
            model_name: Model name to remove (exact match required for safety)
            cache_dir: Optional custom cache directory
            confirm: Whether to require confirmation for safety (default: True)

        Returns:
            Dictionary with removal results
        """
        import shutil

        models_path = _get_models_dir(cache_dir)
        model_path = None
        matches = []

        # Security: Find exact matches only, collect all matches for safety
        if models_path.exists():
            for model_dir in models_path.iterdir():
                if model_dir.is_dir():
                    # Exact match on directory name
                    if model_dir.name == model_name:
                        model_path = model_dir
                        break
                    # Also collect partial matches for safety warning
                    elif model_name in model_dir.name:
                        matches.append(model_dir.name)

        if not model_path:
            error_msg = f"Model '{model_name}' not found (exact match required)"
            if matches:
                error_msg += f". Did you mean: {', '.join(matches[:3])}?"
            return {"removed": False, "error": error_msg}

        try:
            # Calculate size before deletion
            total_size = 0
            file_count = 0
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1

            size_mb = round(total_size / (1024 * 1024), 2)

            # Safety check: Warn about large deletions
            if confirm and (size_mb > 100 or file_count > 1000):
                return {
                    "removed": False,
                    "error": f"Safety check: Model '{model_path.name}' is large ({size_mb} MB, {file_count} files). "
                    f"Use confirm=False to override: oe.models.remove('{model_name}', confirm=False)",
                }

            # Perform deletion
            shutil.rmtree(model_path)
            return {
                "removed": True,
                "model_name": model_path.name,
                "size_freed_mb": size_mb,
                "files_removed": file_count,
            }
        except (PermissionError, OSError) as e:
            return {"removed": False, "error": str(e)}

    @staticmethod
    def clear(cache_dir: Optional[str] = None, confirm: bool = True) -> Dict[str, Any]:
        """Remove all models from cache.

        WARNING: DANGEROUS OPERATION: This will delete ALL cached models.

        Args:
            cache_dir: Optional custom cache directory
            confirm: Whether to require confirmation for safety (default: True)

        Returns:
            Dictionary with clearing results

        Example:
            # Safe usage with confirmation
            result = oe.models.clear()

            # Override safety (dangerous)
            result = oe.models.clear(confirm=False)
        """
        import shutil

        models_path = _get_models_dir(cache_dir)

        if not models_path.exists():
            return {"cleared": False, "error": "Models directory does not exist"}

        try:
            # Calculate what would be deleted
            total_size = 0
            file_count = 0
            model_count = 0
            model_names = []

            for model_dir in models_path.iterdir():
                if model_dir.is_dir():
                    model_count += 1
                    model_names.append(model_dir.name)
                    for file_path in model_dir.rglob("*"):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
                            file_count += 1

            size_mb = round(total_size / (1024 * 1024), 2)

            # Safety check: Always require confirmation for clearing all models
            if confirm:
                return {
                    "cleared": False,
                    "error": f"SAFETY CHECK: This will delete {model_count} models ({size_mb} MB, {file_count} files).\n"
                    + f"Models to be deleted: {', '.join(model_names[:5])}{'...' if len(model_names) > 5 else ''}\n"
                    + "To proceed: oe.models.clear(confirm=False)\n"
                    + "Consider using oe.models.remove('specific-model') instead.",
                }

            # Perform deletion
            shutil.rmtree(models_path)
            return {
                "cleared": True,
                "models_removed": model_count,
                "size_freed_mb": size_mb,
                "files_removed": file_count,
                "model_names": model_names,
            }
        except (PermissionError, OSError) as e:
            return {"cleared": False, "error": str(e)}


class _CacheNamespace:
    """Cache management namespace."""

    @staticmethod
    def dir(cache_dir: Optional[str] = None) -> str:
        """Get the temporary cache directory path."""
        return str(_get_cache_dir(cache_dir))

    @staticmethod
    def size(cache_dir: Optional[str] = None) -> Dict[str, Any]:
        """Get total cache size information."""
        models_path = _get_models_dir(cache_dir)
        temp_cache_path = _get_cache_dir(cache_dir)

        def get_dir_size(path: Path) -> float:
            """Get directory size in MB."""
            if not path.exists():
                return 0.0

            total_size = 0
            try:
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
            except (PermissionError, OSError):
                pass
            return total_size / (1024 * 1024)

        models_size = get_dir_size(models_path)
        temp_size = get_dir_size(temp_cache_path)

        return {
            "models_size_mb": round(models_size, 2),
            "temp_cache_size_mb": round(temp_size, 2),
            "total_size_mb": round(models_size + temp_size, 2),
            "models_path": str(models_path),
            "temp_cache_path": str(temp_cache_path),
            "model_count": len(list(models_path.iterdir()))
            if models_path.exists()
            else 0,
        }

    @staticmethod
    def clear(
        cache_dir: Optional[str] = None, models: bool = False, confirm: bool = None
    ) -> Dict[str, Any]:
        """Clear cache directories.

        Args:
            cache_dir: Optional custom cache directory
            models: If True, also clear models (DANGEROUS). If False, only clear temporary cache.
            confirm: Whether to require confirmation. Auto-determined based on operation danger.

        Returns:
            Dictionary with cleanup results

        Example:
            # Safe: Clear only temporary files
            oe.cache.clear()

            # Dangerous: Clear models too (requires confirmation)
            oe.cache.clear(models=True)

            # Override safety (very dangerous)
            oe.cache.clear(models=True, confirm=False)
        """
        import shutil

        # Auto-determine confirmation requirement
        if confirm is None:
            confirm = models  # Require confirmation if clearing models

        results = {
            "temp_cache_cleared": False,
            "models_cleared": False,
            "temp_size_freed_mb": 0,
            "models_size_freed_mb": 0,
        }

        temp_cache_path = _get_cache_dir(cache_dir)

        # Clear temporary cache (relatively safe)
        if temp_cache_path.exists():
            try:
                # Calculate size before deletion
                temp_size = 0
                file_count = 0
                for file_path in temp_cache_path.rglob("*"):
                    if file_path.is_file():
                        temp_size += file_path.stat().st_size
                        file_count += 1

                shutil.rmtree(temp_cache_path)
                results["temp_cache_cleared"] = True
                results["temp_size_freed_mb"] = round(temp_size / (1024 * 1024), 2)
                results["temp_files_removed"] = file_count
            except (PermissionError, OSError) as e:
                logging.warning(f"Failed to clear temporary cache: {e}")
                results["temp_cache_error"] = str(e)

        # Clear models if requested (DANGEROUS)
        if models:
            models_result = _ModelsNamespace.clear(cache_dir, confirm)
            if models_result.get("cleared"):
                results["models_cleared"] = True
                results["models_size_freed_mb"] = models_result.get("size_freed_mb", 0)
                results["models_removed"] = models_result.get("models_removed", 0)
            elif "error" in models_result:
                results["models_error"] = models_result["error"]

        return results


# Create namespace instances
models = _ModelsNamespace()
cache = _CacheNamespace()


def _raise_openvino_missing() -> None:
    """Raise a helpful error when OpenVINO is missing at call time."""
    install_variants = {
        "cpu": "CPU-only (40MB)",
        "runtime": "CPU runtime (40MB)",
        "gpu": "Intel GPU support",
        "npu": "Intel NPU support",
        "quant": "With INT8 quantization",
        "full": "Full dev environment (~1GB)",
    }
    lines = [
        "OpenVINO runtime not found. Install OpenVINO-Easy with hardware-specific extras:",
        *[f"  • {desc}: pip install 'openvino-easy[{variant}]'" for variant, desc in install_variants.items()],
        "",
        "For more help: https://github.com/example/openvino-easy#installation",
    ]
    raise ImportError("\n".join(lines))


# Thin re-export to support tests that patch `oe.load_model`
def load_model(*args, **kwargs):
    """Lazy proxy to `oe.loader.load_model`.

    Kept at module level so tests can patch `oe.load_model` without importing heavy deps.
    """
    from .loader import load_model as _load_model

    return _load_model(*args, **kwargs)
