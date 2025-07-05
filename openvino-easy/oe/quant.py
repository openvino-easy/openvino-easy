"""Quantization utilities for OpenVINO-Easy (POT auto-INT8 pipeline)."""

from pathlib import Path
import openvino as ov
import numpy as np
import hashlib
import json
import tempfile

# Import proper POT 2024 APIs
try:
    import nncf
    from nncf import compress_weights
    NNCF_AVAILABLE = True
except ImportError:
    NNCF_AVAILABLE = False
    nncf = None
    compress_weights = None

def _generate_calibration_data(model, num_samples=100):
    """Generate random calibration data for quantization."""
    # Get input info from model
    input_info = {}
    for input_node in model.inputs:
        shape = input_node.shape
        # Replace dynamic dimensions with fixed values for calibration
        fixed_shape = [dim if dim > 0 else 1 for dim in shape]
        input_info[input_node.get_any_name()] = fixed_shape
    
    # Generate random calibration data
    calibration_data = []
    for _ in range(num_samples):
        sample = {}
        for input_name, shape in input_info.items():
            # Generate random data with appropriate range
            sample[input_name] = np.random.randn(*shape).astype(np.float32)
        calibration_data.append(sample)
    
    return calibration_data

def _get_model_checksum(model) -> str:
    """Generate a stable checksum from the model's IR representation."""
    # Save model to temporary buffer and hash the IR content
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "temp_model.xml"
        ov.save_model(model, str(temp_path))
        
        # Read and hash the IR content
        with open(temp_path, 'rb') as f:
            model_content = f.read()
        
        # Also hash the weights file if it exists
        weights_path = temp_path.with_suffix('.bin')
        if weights_path.exists():
            with open(weights_path, 'rb') as f:
                model_content += f.read()
        
        return hashlib.sha256(model_content).hexdigest()

def _get_quant_cache_key(model_checksum: str, quant_config: dict, ov_version: str) -> str:
    """Generate a cache key for quantized model."""
    config_str = json.dumps(quant_config, sort_keys=True)
    key_data = f"{model_checksum}:{config_str}:{ov_version}"
    return hashlib.sha256(key_data.encode()).hexdigest()

def quantize_model(model, dtype="int8", cache_dir="~/.cache/oe", **kwargs):
    """
    Quantize a model using NNCF (Neural Network Compression Framework).
    
    Args:
        model: OpenVINO model to quantize
        dtype: Target precision ("int8" or "fp16")
        cache_dir: Directory for caching quantized models
        **kwargs: Additional quantization parameters
    
    Returns:
        Quantized OpenVINO model
    """
    if dtype not in ["int8", "fp16"]:
        raise ValueError(f"Unsupported dtype: {dtype}. Use 'int8' or 'fp16'")
    
    # For fp16, just return the model (no quantization needed)
    if dtype == "fp16":
        return model
    
    # Check if NNCF is available
    if not NNCF_AVAILABLE:
        print("Warning: NNCF not available. Install with 'pip install openvino-easy[quant]' for INT8 quantization support.")
        return model
    
    # Generate stable model checksum
    model_checksum = _get_model_checksum(model)
    
    # Quantization configuration
    quant_config = {
        "algorithm": "DefaultQuantization",
        "preset": kwargs.get("preset", "mixed"),
        "stat_subset_size": kwargs.get("stat_subset_size", 300),
        "fast_bias_correction": kwargs.get("fast_bias_correction", True)
    }
    
    # Generate cache key
    ov_version = ov.__version__
    cache_key = _get_quant_cache_key(model_checksum, quant_config, ov_version)
    cache_dir = Path(cache_dir).expanduser()
    cache_path = cache_dir / "quantized" / cache_key
    
    # Check if quantized model is already cached
    if cache_path.exists():
        model_xml = cache_path / "model.xml"
        if model_xml.exists():
            return ov.Core().read_model(str(model_xml))
    
    try:
        # For INT8 quantization, use NNCF weight compression
        # This is the modern approach in OpenVINO 2024
        quantized_model = compress_weights(
            model,
            mode=nncf.CompressWeightsMode.INT8,
            ratio=kwargs.get("ratio", 1.0),
            group_size=kwargs.get("group_size", -1)
        )
        
        # Cache the quantized model
        cache_path.mkdir(parents=True, exist_ok=True)
        ov.save_model(quantized_model, str(cache_path / "model.xml"))
        
        # Save metadata
        metadata = {
            "original_model_checksum": model_checksum,
            "quant_config": quant_config,
            "ov_version": ov_version,
            "cache_key": cache_key,
            "dtype": dtype,
            "quantization_method": "NNCF compress_weights"
        }
        with open(cache_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return quantized_model
        
    except Exception as e:
        print(f"Warning: Failed to quantize model: {e}")
        print("Returning original model without quantization.")
        return model

def get_quantization_stats(model, quantized_model):
    """
    Get quantization statistics comparing original and quantized models.
    
    Returns:
        Dictionary with model size and other statistics
    """
    # Calculate approximate model sizes
    def _estimate_model_size(model):
        """Estimate model size in MB."""
        # Use a more stable approach - estimate from model IR size
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "temp_model.xml"
                ov.save_model(model, str(temp_path))
                
                # Get XML file size
                xml_size = temp_path.stat().st_size
                
                # Get weights file size if it exists
                weights_path = temp_path.with_suffix('.bin')
                weights_size = weights_path.stat().st_size if weights_path.exists() else 0
                
                # Return total size in MB
                return (xml_size + weights_size) / (1024 * 1024)
        except:
            # Fallback: rough estimation based on input/output shapes
            total_size = 0
            for input_node in model.inputs:
                shape_size = 1
                for dim in input_node.shape:
                    if dim > 0:
                        shape_size *= dim
                total_size += shape_size * 4  # 4 bytes per float32
            
            return total_size / (1024 * 1024)
    
    try:
        original_size = _estimate_model_size(model)
        quantized_size = _estimate_model_size(quantized_model)
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
    except:
        # Fallback if estimation fails
        original_size = 0
        quantized_size = 0
        compression_ratio = 0
    
    return {
        "original_size_mb": round(original_size, 2),
        "quantized_size_mb": round(quantized_size, 2),
        "compression_ratio": round(compression_ratio, 2),
        "quantization_method": "NNCF compress_weights" if NNCF_AVAILABLE else "None"
    } 