"""Quantization utilities for OpenVINO-Easy (NNCF auto-INT8 pipeline)."""

from pathlib import Path
import openvino as ov
import numpy as np
import hashlib
import json
import tempfile
from typing import Optional, Union, List, Dict, Any
import warnings
import logging

# Import proper NNCF 2025 APIs
try:
    import nncf
    from nncf import compress_weights, quantize
    from nncf.common.utils.api_marker import api

    NNCF_AVAILABLE = True
except ImportError:
    NNCF_AVAILABLE = False
    nncf = None
    compress_weights = None
    quantize = None


def _generate_smart_calibration_data(
    model, model_info: Optional[Dict[str, Any]] = None, num_samples: int = 100
):
    """Generate smarter calibration data based on model type and available information."""

    # Try to use model-specific calibration if we have model info
    if model_info and model_info.get("source_path"):
        calibration_data = _try_dataset_calibration(model, model_info, num_samples)
        if calibration_data:
            logging.info(
                f"✅ Generated {len(calibration_data)} calibration samples using real dataset"
            )
            return calibration_data

    # Try to detect model type and generate appropriate data
    calibration_data = _generate_typed_calibration_data(model, num_samples)
    if calibration_data:
        logging.info(
            f"⚠️  Generated {len(calibration_data)} calibration samples using typed heuristics"
        )
        return calibration_data

    # Fallback to random data with warning
    logging.warning(
        "⚠️  Using random calibration data - this may reduce quantization accuracy"
    )
    warnings.warn(
        "Using random calibration data for quantization. "
        "This may reduce accuracy. Consider providing real dataset samples or installing datasets: "
        "pip install datasets",
        UserWarning,
    )
    return _generate_random_calibration_data(model, num_samples)


def _try_dataset_calibration(model, model_info: Dict[str, Any], num_samples: int):
    """Try to use real dataset for calibration based on model type."""
    try:
        # Try to detect if it's a text model
        if _is_text_model(model, model_info):
            return _generate_text_calibration_data(model, model_info, num_samples)

        # Try to detect if it's a vision model
        if _is_vision_model(model):
            return _generate_vision_calibration_data(model, num_samples)

    except Exception as e:
        logging.warning(f"⚠️  Failed to generate dataset calibration: {e}")
        return None

    return None


def _is_text_model(model, model_info: Dict[str, Any]) -> bool:
    """Detect if model is a text/NLP model."""
    # Check input shapes (text models typically have 1D or 2D integer inputs)
    for input_node in model.inputs:
        shape = input_node.shape
        if len(shape) <= 2 and any(dim > 100 for dim in shape if dim > 0):
            return True

    # Check if we have tokenizer info
    source_path = model_info.get("source_path", "")
    text_indicators = [
        "bert",
        "gpt",
        "dialogpt",
        "roberta",
        "distilbert",
        "albert",
        "bloom",
        "opt",
    ]
    return any(indicator in source_path.lower() for indicator in text_indicators)


def _is_vision_model(model) -> bool:
    """Detect if model is a vision model."""
    for input_node in model.inputs:
        shape = input_node.shape
        # Vision models typically have 4D inputs: [batch, channels, height, width]
        if len(shape) == 4 and shape[-2:] == [224, 224] or shape[-2:] == [512, 512]:
            return True
    return False


def _generate_text_calibration_data(
    model, model_info: Dict[str, Any], num_samples: int
):
    """Generate calibration data for text models using real datasets when available."""
    try:
        # Import tokenizer if available
        from transformers import AutoTokenizer

        source_path = model_info.get("source_path")
        if not source_path:
            return None

        tokenizer = AutoTokenizer.from_pretrained(source_path)

        # Try to use real dataset first
        real_texts = _try_real_text_dataset(model_info, num_samples)

        if real_texts:
            logging.info(f"📚 Using real dataset with {len(real_texts)} text samples")
            text_samples = real_texts
        else:
            # Fallback to curated text samples that are more representative
            logging.info(
                "📝 Using curated text samples (consider installing 'datasets' for better accuracy)"
            )
            diverse_samples = [
                # News/factual
                "Scientists have discovered a new method for converting plastic waste into useful materials.",
                "The stock market experienced significant volatility following the latest economic reports.",
                "Climate change continues to affect weather patterns across the globe.",
                # Technical/AI
                "Machine learning algorithms require large amounts of training data to achieve optimal performance.",
                "Neural network architectures have evolved significantly over the past decade.",
                "Quantization techniques help reduce model size while maintaining accuracy.",
                # Conversational
                "Hello, how can I help you today? I'm here to assist with any questions you might have.",
                "Thank you for reaching out. Let me provide you with the information you need.",
                "I understand your concern. Let's work together to find a solution.",
                # Literary/creative
                "The ancient library stood majestically against the backdrop of the setting sun.",
                "In the heart of the bustling city, a small garden provided a peaceful retreat.",
                "The artist carefully mixed colors on her palette, preparing to capture the scene.",
                # Technical documentation
                "To configure the system, first ensure all dependencies are properly installed.",
                "The function takes two parameters and returns a processed result.",
                "Error handling is crucial for maintaining system stability and user experience.",
                # Educational
                "Learning new skills requires practice, patience, and dedication to improvement.",
                "The scientific method involves forming hypotheses and testing them systematically.",
                "Historical events often have complex causes and far-reaching consequences.",
            ]

            # Repeat and shuffle to get enough samples
            text_samples = (
                diverse_samples * (num_samples // len(diverse_samples) + 1)
            )[:num_samples]

        calibration_data = []
        input_names = [node.get_any_name() for node in model.inputs]

        for i in range(min(num_samples, len(text_samples))):
            text = text_samples[i]

            # Tokenize
            encoded = tokenizer(
                text, return_tensors="np", padding=True, truncation=True, max_length=512
            )

            # Map to model inputs
            sample = {}
            for input_name in input_names:
                if input_name in ["input_ids", "input"] and "input_ids" in encoded:
                    sample[input_name] = encoded["input_ids"].astype(np.float32)
                elif (
                    input_name in ["attention_mask", "mask"]
                    and "attention_mask" in encoded
                ):
                    sample[input_name] = encoded["attention_mask"].astype(np.float32)
                elif (
                    input_name in ["token_type_ids", "segment_ids"]
                    and "token_type_ids" in encoded
                ):
                    sample[input_name] = encoded["token_type_ids"].astype(np.float32)
                else:
                    # Use first available tensor
                    sample[input_name] = list(encoded.values())[0].astype(np.float32)

            calibration_data.append(sample)

        return calibration_data

    except Exception as e:
        warnings.warn(f"Failed to generate text calibration data: {e}")
        return None


def _generate_vision_calibration_data(model, num_samples: int):
    """Generate calibration data for vision models using real datasets when available."""
    try:
        # Try to use real image dataset first
        real_images = _try_real_vision_dataset(num_samples)

        calibration_data = []

        for input_node in model.inputs:
            input_name = input_node.get_any_name()
            shape = list(input_node.shape)

            # Replace dynamic dimensions
            fixed_shape = [dim if dim > 0 else 1 for dim in shape]

            if real_images and len(fixed_shape) == 4:
                logging.info(
                    f"🖼️  Using real image dataset with {len(real_images)} samples"
                )

                for i in range(min(num_samples, len(real_images))):
                    real_img = real_images[i]

                    # Reshape to match model input shape
                    if fixed_shape[1] == 3:  # CHW format [B, C, H, W]
                        if real_img.shape != (3, fixed_shape[2], fixed_shape[3]):
                            # Resize if needed
                            from PIL import Image

                            img_hwc = np.transpose(real_img, (1, 2, 0))  # CHW -> HWC
                            img_pil = Image.fromarray((img_hwc * 255).astype(np.uint8))
                            img_pil = img_pil.resize((fixed_shape[3], fixed_shape[2]))
                            img_array = np.array(img_pil).astype(np.float32) / 255.0
                            sample_data = np.transpose(
                                img_array, (2, 0, 1)
                            )  # HWC -> CHW
                        else:
                            sample_data = real_img

                        # Add batch dimension
                        sample_data = sample_data.reshape(fixed_shape)

                    elif fixed_shape[3] == 3:  # HWC format [B, H, W, C]
                        img_hwc = np.transpose(real_img, (1, 2, 0))  # CHW -> HWC
                        if img_hwc.shape != (fixed_shape[1], fixed_shape[2], 3):
                            from PIL import Image

                            img_pil = Image.fromarray((img_hwc * 255).astype(np.uint8))
                            img_pil = img_pil.resize((fixed_shape[2], fixed_shape[1]))
                            img_hwc = np.array(img_pil).astype(np.float32) / 255.0

                        sample_data = img_hwc.reshape(fixed_shape)
                    else:
                        # Unknown format, use synthetic data
                        sample_data = np.random.normal(0.45, 0.25, fixed_shape).astype(
                            np.float32
                        )
                        sample_data = np.clip(sample_data, 0, 1)

                    if i < len(calibration_data):
                        calibration_data[i][input_name] = sample_data
                    else:
                        calibration_data.append({input_name: sample_data})
            else:
                # Fallback to synthetic data with better statistics
                logging.info(
                    "🎨 Using synthetic image data with ImageNet statistics (consider installing 'datasets' for better accuracy)"
                )

                for i in range(num_samples):
                    # Generate data with ImageNet-like statistics
                    if len(fixed_shape) == 4:  # [B, C, H, W] or [B, H, W, C]
                        if fixed_shape[1] == 3:  # CHW format
                            # ImageNet mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]
                            sample_data = np.random.normal(
                                loc=[0.485, 0.456, 0.406],
                                scale=[0.229, 0.224, 0.225],
                                size=fixed_shape,
                            ).astype(np.float32)
                        else:  # HWC format assumed
                            sample_data = np.random.normal(
                                loc=0.45, scale=0.25, size=fixed_shape
                            ).astype(np.float32)
                    else:
                        # Fallback for other shapes
                        sample_data = np.random.normal(0, 0.25, fixed_shape).astype(
                            np.float32
                        )

                    # Clip to reasonable range
                    sample_data = np.clip(sample_data, 0, 1)

                    if i < len(calibration_data):
                        calibration_data[i][input_name] = sample_data
                    else:
                        calibration_data.append({input_name: sample_data})

        return calibration_data

    except Exception as e:
        warnings.warn(f"Failed to generate vision calibration data: {e}")
        return None


def _generate_typed_calibration_data(model, num_samples: int):
    """Generate calibration data based on input characteristics."""
    calibration_data = []

    for i in range(num_samples):
        sample = {}

        for input_node in model.inputs:
            input_name = input_node.get_any_name()
            shape = list(input_node.shape)
            fixed_shape = [dim if dim > 0 else 1 for dim in shape]

            # Generate data based on shape characteristics
            if len(fixed_shape) == 4:  # Likely vision
                # Use ImageNet-like distribution
                data = np.random.normal(0.45, 0.25, fixed_shape).astype(np.float32)
                data = np.clip(data, 0, 1)
            elif len(fixed_shape) <= 2:  # Likely text/embeddings
                # Use smaller range for text-like data
                data = np.random.normal(0, 0.1, fixed_shape).astype(np.float32)
            else:
                # General case
                data = np.random.normal(0, 0.5, fixed_shape).astype(np.float32)

            sample[input_name] = data

        calibration_data.append(sample)

    return calibration_data


def _generate_random_calibration_data(model, num_samples: int = 100):
    """Generate random calibration data (fallback method)."""
    input_info = {}
    for input_node in model.inputs:
        shape = input_node.shape
        fixed_shape = [dim if dim > 0 else 1 for dim in shape]
        input_info[input_node.get_any_name()] = fixed_shape

    calibration_data = []
    for _ in range(num_samples):
        sample = {}
        for input_name, shape in input_info.items():
            sample[input_name] = np.random.randn(*shape).astype(np.float32)
        calibration_data.append(sample)

    return calibration_data


def _try_real_text_dataset(
    model_info: Dict[str, Any], num_samples: int
) -> Optional[List[str]]:
    """Try to load real text dataset for better calibration."""
    try:
        from datasets import load_dataset

        # Determine appropriate dataset based on model type
        source_path = model_info.get("source_path", "").lower()

        # Choose dataset based on model characteristics
        if any(
            indicator in source_path for indicator in ["gpt", "bloom", "opt", "llama"]
        ):
            # Generative models - use diverse text
            try:
                dataset = load_dataset(
                    "wikitext", "wikitext-2-raw-v1", split="train", streaming=True
                )
                texts = []
                for i, item in enumerate(dataset):
                    if i >= num_samples:
                        break
                    text = item.get("text", "").strip()
                    if len(text) > 50:  # Skip very short texts
                        texts.append(text[:500])  # Limit length
                return texts
            except:
                pass

        elif any(
            indicator in source_path for indicator in ["bert", "roberta", "distilbert"]
        ):
            # BERT-style models - use sentence classification data
            try:
                dataset = load_dataset("glue", "sst2", split="train", streaming=True)
                texts = []
                for i, item in enumerate(dataset):
                    if i >= num_samples:
                        break
                    texts.append(item.get("sentence", ""))
                return texts
            except:
                pass

        # Fallback: use a general text dataset
        try:
            dataset = load_dataset("c4", "realnewslike", split="train", streaming=True)
            texts = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                text = item.get("text", "").strip()
                if len(text) > 100:
                    texts.append(text[:300])  # Shorter for better performance
            return texts
        except:
            pass

    except ImportError:
        logging.info(
            "💡 Install 'datasets' for better calibration: pip install datasets"
        )
    except Exception as e:
        logging.warning(f"⚠️  Could not load real dataset: {e}")

    return None


def _try_real_vision_dataset(num_samples: int) -> Optional[List[np.ndarray]]:
    """Try to load real vision dataset for better calibration."""
    try:
        from datasets import load_dataset
        from PIL import Image

        # Use ImageNet validation set if available
        try:
            dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
            images = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                try:
                    img = item["image"]
                    if isinstance(img, Image.Image):
                        # Convert to numpy and normalize
                        img_array = np.array(img.resize((224, 224)).convert("RGB"))
                        img_array = img_array.astype(np.float32) / 255.0
                        # Convert to CHW format
                        img_array = np.transpose(img_array, (2, 0, 1))
                        images.append(img_array)
                except:
                    continue
            return images if images else None
        except:
            pass

        # Fallback to CIFAR-10
        try:
            dataset = load_dataset("cifar10", split="train", streaming=True)
            images = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                try:
                    img = item["img"]
                    if isinstance(img, Image.Image):
                        # Resize to 224x224 and normalize
                        img_array = np.array(img.resize((224, 224)).convert("RGB"))
                        img_array = img_array.astype(np.float32) / 255.0
                        img_array = np.transpose(img_array, (2, 0, 1))
                        images.append(img_array)
                except:
                    continue
            return images if images else None
        except:
            pass

    except ImportError:
        logging.info(
            "💡 Install 'datasets' for better vision calibration: pip install datasets pillow"
        )
    except Exception as e:
        logging.warning(f"⚠️  Could not load real vision dataset: {e}")

    return None


def _get_model_checksum(model) -> str:
    """Generate a stable checksum from the model's IR representation."""
    # Save model to temporary buffer and hash the IR content
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "temp_model.xml"
        ov.save_model(model, str(temp_path))

        # Read and hash the IR content
        with open(temp_path, "rb") as f:
            model_content = f.read()

        # Also hash the weights file if it exists
        weights_path = temp_path.with_suffix(".bin")
        if weights_path.exists():
            with open(weights_path, "rb") as f:
                model_content += f.read()

        return hashlib.sha256(model_content).hexdigest()


def _get_quant_cache_key(
    model_checksum: str, quant_config: dict, ov_version: str
) -> str:
    """Generate a cache key for quantized model."""
    config_str = json.dumps(quant_config, sort_keys=True)
    key_data = f"{model_checksum}:{config_str}:{ov_version}"
    return hashlib.sha256(key_data.encode()).hexdigest()


def quantize_model(
    model: "ov.Model",
    dtype: str = "int8",
    cache_dir: Union[str, Path] = "~/.cache/oe",
    model_info: Optional[Dict[str, Any]] = None,
    calibration_data: Optional[List[Dict[str, np.ndarray]]] = None,
    quantization_method: str = "auto",
    **kwargs,
) -> "ov.Model":
    """
    Quantize a model using NNCF (Neural Network Compression Framework).

    Args:
        model: OpenVINO model to quantize
        dtype: Target precision ("int8" or "fp16")
        cache_dir: Directory for caching quantized models
        model_info: Model metadata for smart calibration
        calibration_data: Custom calibration dataset
        quantization_method: "auto", "weights_only", or "full_quantization"
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
        warnings.warn(
            "NNCF not available. Install with 'pip install openvino-easy[quant]' for INT8 quantization support."
        )
        return model

    # Generate stable model checksum
    model_checksum = _get_model_checksum(model)

    # Determine quantization method
    if quantization_method == "auto":
        # Use full quantization if we have good calibration data, otherwise weights only
        quantization_method = (
            "full_quantization" if calibration_data or model_info else "weights_only"
        )

    # Quantization configuration
    quant_config = {
        "method": quantization_method,
        "preset": kwargs.get("preset", "mixed"),
        "stat_subset_size": kwargs.get("stat_subset_size", 300),
        "fast_bias_correction": kwargs.get("fast_bias_correction", True),
        "ratio": kwargs.get("ratio", 1.0),
        "group_size": kwargs.get("group_size", -1),
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
        if quantization_method == "weights_only":
            # Simple weight compression
            quantized_model = compress_weights(
                model,
                mode=nncf.CompressWeightsMode.INT8,
                ratio=quant_config["ratio"],
                group_size=quant_config["group_size"],
            )
        else:
            # Full quantization with calibration
            if not calibration_data:
                logging.info("Generating calibration data for full quantization...")
                calibration_data = _generate_smart_calibration_data(
                    model, model_info, quant_config["stat_subset_size"]
                )

            # Create calibration dataset
            def calibration_dataset():
                for sample in calibration_data:
                    yield sample

            # Full quantization
            quantized_model = quantize(
                model,
                calibration_dataset,
                preset=nncf.QuantizationPreset.MIXED,
                subset_size=min(
                    len(calibration_data), quant_config["stat_subset_size"]
                ),
                fast_bias_correction=quant_config["fast_bias_correction"],
            )

        # Cache the quantized model
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            writable_path = cache_path
        except Exception:
            # Fall back to a temporary directory if cache is not writable
            writable_path = Path(tempfile.mkdtemp(prefix="oe_quant_"))

        ov.save_model(quantized_model, str(writable_path / "model.xml"))

        # Save metadata
        metadata = {
            "original_model_checksum": model_checksum,
            "quant_config": quant_config,
            "ov_version": ov_version,
            "cache_key": cache_key,
            "dtype": dtype,
            "quantization_method": f"NNCF {quantization_method}",
            "calibration_samples": len(calibration_data) if calibration_data else 0,
        }
        with open(writable_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logging.info(
            f"✅ Model quantized using {quantization_method} with {len(calibration_data) if calibration_data else 0} calibration samples"
        )
        return quantized_model

    except Exception as e:
        logging.warning(f"Failed to quantize model: {e}")
        logging.info("Returning original model without quantization.")
        return model


def get_quantization_stats(
    model: "ov.Model", quantized_model: "ov.Model"
) -> Dict[str, Any]:
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
                weights_path = temp_path.with_suffix(".bin")
                weights_size = (
                    weights_path.stat().st_size if weights_path.exists() else 0
                )

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
        "quantization_method": "NNCF compress_weights" if NNCF_AVAILABLE else "None",
    }
