"""Model loader for OpenVINO-Easy (HF/ONNX/IR to ov.Model)."""

from pathlib import Path
import openvino as ov
from huggingface_hub import snapshot_download, hf_hub_download
import hashlib
import json
import tempfile
import os

def _get_cache_key(model_id: str, dtype: str, ov_version: str) -> str:
    """Generate a cache key for the model."""
    key_data = f"{model_id}:{dtype}:{ov_version}"
    return hashlib.sha256(key_data.encode()).hexdigest()

def _is_diffusers_model(model_path: Path) -> bool:
    """Check if the model is a diffusers pipeline."""
    # Check for diffusers-specific files
    diffusers_files = [
        "model_index.json",
        "scheduler/scheduler_config.json", 
        "text_encoder/config.json",
        "unet/config.json",
        "vae/config.json"
    ]
    return any((model_path / file).exists() for file in diffusers_files)

def _convert_with_optimum_intel(model_path: str, output_dir: str, dtype: str = "fp16"):
    """Convert diffusers model using optimum-intel."""
    try:
        from optimum.intel import OVStableDiffusionPipeline
        
        # Load and convert using optimum-intel
        pipeline = OVStableDiffusionPipeline.from_pretrained(
            model_path,
            export=True,
            compile=False,
            device="CPU"  # We'll compile later on target device
        )
        
        # Save to output directory
        pipeline.save_pretrained(output_dir)
        
        # Return path to the main model file
        return str(Path(output_dir) / "unet" / "openvino_model.xml")
        
    except ImportError:
        raise RuntimeError(
            "optimum-intel is required for diffusers models. "
            "Install with: pip install optimum[openvino]"
        )

def load_model(model_id_or_path: str, dtype: str = "fp16", cache_dir: str | Path = "~/.cache/oe"):
    """
    Load a model from Hugging Face Hub, ONNX, or OpenVINO IR.
    Returns an ov.Model (uncompiled).
    """
    model_path = Path(model_id_or_path)
    cache_dir = Path(cache_dir).expanduser()
    
    # Check if it's a local IR model
    if model_path.exists() and model_path.suffix == ".xml":
        return ov.Core().read_model(str(model_path))
    
    # Check if it's a local ONNX model
    if model_path.exists() and model_path.suffix == ".onnx":
        return ov.Core().read_model(str(model_path))
    
    # Assume it's a Hugging Face model ID
    # Generate cache key
    ov_version = ov.__version__
    cache_key = _get_cache_key(model_id_or_path, dtype, ov_version)
    cache_path = cache_dir / cache_key
    
    # Check if model is already cached
    if cache_path.exists():
        model_xml = cache_path / "model.xml"
        if model_xml.exists():
            return ov.Core().read_model(str(model_xml))
    
    # Download and convert Hugging Face model
    try:
        # Download model files
        local_model_path = snapshot_download(
            repo_id=model_id_or_path,
            cache_dir=cache_dir / "hf_cache",
            allow_patterns=["*.safetensors", "*.bin", "*.json", "*.txt", "*.md", "*.py"]
        )
        
        local_path = Path(local_model_path)
        
        # Check if it's a diffusers model
        if _is_diffusers_model(local_path):
            # Use optimum-intel for diffusers models
            with tempfile.TemporaryDirectory() as temp_dir:
                model_xml_path = _convert_with_optimum_intel(
                    str(local_path), temp_dir, dtype
                )
                model = ov.Core().read_model(model_xml_path)
        else:
            # Convert standard model to OpenVINO format
            # Create output directory for conversion
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "model.xml"
                
                try:
                    # Try direct conversion
                    model = ov.convert_model(
                        str(local_path),
                        compress_to_fp16=(dtype == "fp16")
                    )
                except Exception as e:
                    # Fallback: try with specific input shape if needed
                    try:
                        # For some models, we need to specify example input
                        model = ov.convert_model(
                            str(local_path),
                            compress_to_fp16=(dtype == "fp16"),
                            example_input=None  # Could be enhanced with model-specific logic
                        )
                    except Exception as e2:
                        raise RuntimeError(
                            f"Failed to convert model '{model_id_or_path}': {e}. "
                            f"Fallback also failed: {e2}"
                        )
        
        # Cache the converted model
        cache_path.mkdir(parents=True, exist_ok=True)
        ov.save_model(model, str(cache_path / "model.xml"))
        
        # Save metadata
        metadata = {
            "model_id": model_id_or_path,
            "dtype": dtype,
            "ov_version": ov_version,
            "cache_key": cache_key
        }
        with open(cache_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load Hugging Face model '{model_id_or_path}': {e}") 