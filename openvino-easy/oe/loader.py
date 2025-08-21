"""Model loader for OpenVINO-Easy (HF/ONNX/IR to ov.Model)."""

from pathlib import Path
from typing import Union
import openvino as ov
from huggingface_hub import snapshot_download
import hashlib
import json
import tempfile
import time
import warnings
import logging


# Custom exceptions for better error handling
class ModelLoadError(Exception):
    """Base exception for model loading errors."""

    pass


class ModelNotFoundError(ModelLoadError):
    """Model not found or inaccessible."""

    pass


class ModelConversionError(ModelLoadError):
    """Model conversion failed."""

    pass


class NetworkError(ModelLoadError):
    """Network/download related error."""

    pass


class UnsupportedModelError(ModelLoadError):
    """Model format not supported."""

    pass


class CorruptedModelError(ModelLoadError):
    """Model file is corrupted or invalid."""

    pass


def _download_with_retry(
    repo_id: str, cache_dir: Path, max_retries: int = 3, retry_delay: float = 1.0
) -> str:
    """Download model with retry logic and better error handling."""
    last_exception = None

    for attempt in range(max_retries):
        try:
            logging.info(
                f"📥 Downloading {repo_id} (attempt {attempt + 1}/{max_retries})"
            )

            # Add retry parameters
            local_model_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir / "hf_cache",
                allow_patterns=[
                    "*.safetensors",
                    "*.bin",
                    "*.json",
                    "*.txt",
                    "*.md",
                    "*.py",
                    "*.onnx",
                ],
                resume_download=True,  # Resume partial downloads
                local_files_only=False,
            )

            # Verify download succeeded
            local_path = Path(local_model_path)
            if not local_path.exists():
                raise ModelNotFoundError(
                    f"Downloaded path does not exist: {local_path}"
                )

            # Basic integrity check
            if not any(local_path.iterdir()):
                raise CorruptedModelError(
                    f"Downloaded model directory is empty: {local_path}"
                )

            logging.info(f"✅ Successfully downloaded {repo_id}")
            return local_model_path

        except Exception as e:
            last_exception = e

            # Classify the error
            if "not found" in str(e).lower() or "404" in str(e):
                raise ModelNotFoundError(
                    f"Model '{repo_id}' not found on Hugging Face Hub: {e}"
                )
            elif (
                "network" in str(e).lower()
                or "connection" in str(e).lower()
                or "timeout" in str(e).lower()
            ):
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)  # Exponential backoff
                    logging.warning(
                        f"⚠️  Network error, retrying in {wait_time:.1f}s: {e}"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    raise NetworkError(
                        f"Failed to download '{repo_id}' after {max_retries} attempts: {e}"
                    )
            elif "permission" in str(e).lower() or "forbidden" in str(e).lower():
                raise ModelNotFoundError(
                    f"Access denied for model '{repo_id}'. Model may be private or require authentication: {e}"
                )
            else:
                # Unknown error - retry if we have attempts left
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    logging.warning(
                        f"⚠️  Download failed, retrying in {wait_time:.1f}s: {e}"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    break

    # All retries exhausted
    if last_exception:
        raise NetworkError(
            f"Failed to download '{repo_id}' after {max_retries} attempts. Last error: {last_exception}"
        )
    else:
        raise ModelLoadError(f"Unknown error downloading '{repo_id}'")


def _safe_model_conversion(convert_func, model_path: str, *args, **kwargs):
    """Safely execute model conversion with detailed error handling."""
    try:
        return convert_func(model_path, *args, **kwargs)
    except ImportError as e:
        # Missing dependency
        missing_dep = str(e).split("'")[1] if "'" in str(e) else "unknown"
        raise ModelConversionError(
            f"Missing dependency '{missing_dep}' for model conversion. "
            f"Install with: pip install 'openvino-easy[text,stable-diffusion]'"
        ) from e
    except FileNotFoundError as e:
        raise CorruptedModelError(f"Model file not found during conversion: {e}") from e
    except MemoryError as e:
        raise ModelConversionError(
            "Insufficient memory for model conversion. "
            "Try reducing batch size or converting on a machine with more RAM."
        ) from e
    except Exception as e:
        # Generic conversion error with helpful context
        error_msg = str(e)
        if "config.json" in error_msg:
            raise ModelConversionError(
                f"Model configuration error: {e}. "
                f"The model may have an invalid or incompatible config.json file."
            ) from e
        elif "shape" in error_msg.lower() or "dimension" in error_msg.lower():
            raise ModelConversionError(
                f"Model shape/dimension error: {e}. "
                f"This model architecture may not be supported by OpenVINO."
            ) from e
        else:
            raise ModelConversionError(f"Model conversion failed: {e}") from e


def _verify_model_integrity(model, model_path: str):
    """Verify that the loaded model is valid."""
    try:
        # Basic model validation
        if model is None:
            raise CorruptedModelError(f"Model loaded as None from {model_path}")

        # Check if model has inputs and outputs
        if not hasattr(model, "inputs") or not hasattr(model, "outputs"):
            raise CorruptedModelError(f"Model missing inputs/outputs: {model_path}")

        if len(model.inputs) == 0:
            raise CorruptedModelError(f"Model has no inputs: {model_path}")

        if len(model.outputs) == 0:
            raise CorruptedModelError(f"Model has no outputs: {model_path}")

        # Check input/output shapes are reasonable
        for i, input_node in enumerate(model.inputs):
            if not hasattr(input_node, "shape"):
                raise CorruptedModelError(
                    f"Input {i} missing shape information: {model_path}"
                )

        for i, output_node in enumerate(model.outputs):
            if not hasattr(output_node, "shape"):
                raise CorruptedModelError(
                    f"Output {i} missing shape information: {model_path}"
                )

        logging.info(
            f"✅ Model integrity verified: {len(model.inputs)} inputs, {len(model.outputs)} outputs"
        )

    except Exception as e:
        if isinstance(e, CorruptedModelError):
            raise
        else:
            raise CorruptedModelError(f"Model integrity check failed: {e}") from e


def _get_cache_key(model_id: str, dtype: str, ov_version: str) -> str:
    """Generate a cache key for the model."""
    key_data = f"{model_id}:{dtype}:{ov_version}"
    return hashlib.sha256(key_data.encode()).hexdigest()


def _detect_model_type(model_path: Path) -> str:
    """Detect the type of model based on files and structure."""
    # Check for diffusers models
    diffusers_files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "unet/config.json",
        "vae/config.json",
    ]
    if any((model_path / file).exists() for file in diffusers_files):
        return "diffusers"

    # Check for transformers models with enhanced detection
    if (model_path / "config.json").exists():
        try:
            with open(model_path / "config.json", "r") as f:
                config = json.load(f)

            # Check model architecture
            model_type = config.get("model_type", "").lower()
            architectures = config.get("architectures", [])
            task = config.get("task", "").lower()

            # Enhanced architecture detection with more models
            optimum_compatible = [
                "bert",
                "roberta",
                "distilbert",
                "albert",
                "electra",
                "deberta",
                "camembert",
                "gpt2",
                "gpt_neo",
                "gpt_neox",
                "gpt_j",
                "codegen",
                "opt",
                "bloom",
                "llama",
                "t5",
                "mt5",
                "bart",
                "pegasus",
                "marian",
                "blenderbot",
                "whisper",
                "wav2vec2",
            ]

            # Check for audio models specifically
            audio_indicators = [
                "whisper",
                "wav2vec2",
                "wavlm",
                "hubert",
                "speecht5",
                "bark",
                "vall-e",
            ]
            if (
                model_type in audio_indicators
                or any(
                    arch.lower().startswith(tuple(audio_indicators))
                    for arch in architectures
                )
                or any(audio_arch in model_type for audio_arch in audio_indicators)
                or task
                in [
                    "automatic-speech-recognition",
                    "text-to-speech",
                    "audio-classification",
                ]
            ):
                return "transformers_audio"

            # Check if it's a known optimum-compatible model
            if (
                model_type in optimum_compatible
                or any(
                    arch.lower().startswith(tuple(optimum_compatible))
                    for arch in architectures
                )
                or any(opt_arch in model_type for opt_arch in optimum_compatible)
            ):
                return "transformers_optimum"

            # Check for vision transformers and other vision models
            vision_indicators = [
                "vit",
                "deit",
                "swin",
                "beit",
                "convnext",
                "resnet",
                "efficientnet",
            ]
            if any(indicator in model_type for indicator in vision_indicators) or any(
                any(indicator in arch.lower() for indicator in vision_indicators)
                for arch in architectures
            ):
                return "transformers_vision"

            # Check for multimodal models
            multimodal_indicators = ["clip", "blip", "llava", "flamingo", "kosmos"]
            if any(
                indicator in model_type for indicator in multimodal_indicators
            ) or any(
                any(indicator in arch.lower() for indicator in multimodal_indicators)
                for arch in architectures
            ):
                return "transformers_multimodal"

            # Enhanced detection based on task type
            if task in ["text-generation", "text2text-generation", "conversational"]:
                return "transformers_optimum"  # These usually work well with optimum
            elif task in [
                "image-classification",
                "object-detection",
                "image-segmentation",
            ]:
                return "transformers_vision"

            # Check for tokenizer files to confirm it's a text model
            has_tokenizer = any(
                (model_path / f).exists()
                for f in [
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.txt",
                    "vocab.json",
                    "merges.txt",
                    "special_tokens_map.json",
                ]
            )

            if has_tokenizer:
                return (
                    "transformers_optimum"  # Likely text model that works with optimum
                )

            # Check input shape hints in config
            if "max_position_embeddings" in config or "n_positions" in config:
                return "transformers_optimum"  # Sequence models

            # Other transformers models - try direct conversion first
            return "transformers_direct"
        except Exception as e:
            logging.warning(f"⚠️  Error parsing config.json: {e}")

    # Enhanced ONNX detection with metadata checking
    onnx_files = list(model_path.glob("*.onnx"))
    if onnx_files:
        # Try to get more info about the ONNX model
        try:
            # Check for accompanying files that might give hints
            if (model_path / "tokenizer.json").exists():
                return "onnx_text"
            elif any(
                (model_path / f).exists()
                for f in ["labels.txt", "imagenet_classes.txt"]
            ):
                return "onnx_vision"
            else:
                return "onnx"
        except:
            return "onnx"

    # Enhanced PyTorch detection
    pytorch_files = (
        list(model_path.glob("*.pt"))
        + list(model_path.glob("*.pth"))
        + list(model_path.glob("*.bin"))
    )
    if pytorch_files:
        # Check if it's a transformers model with PyTorch weights
        if (model_path / "config.json").exists():
            return "pytorch_transformers"
        else:
            return "pytorch_native"

    # Enhanced safetensors detection
    safetensors_files = list(model_path.glob("*.safetensors"))
    if safetensors_files:
        if (model_path / "config.json").exists():
            return "safetensors_transformers"
        else:
            return "safetensors_native"

    # Check for TensorFlow models
    tf_files = (
        list(model_path.glob("*.pb"))
        + list(model_path.glob("saved_model.pb"))
        + list(model_path.glob("*.h5"))
    )
    if tf_files:
        return "tensorflow"

    # Check for other common model formats
    if list(model_path.glob("*.tflite")):
        return "tflite"
    if list(model_path.glob("*.mlmodel")):
        return "coreml"

    return "unknown"


def _is_diffusers_model(model_path: Path) -> bool:
    """Check if the model is a diffusers pipeline."""
    return _detect_model_type(model_path) == "diffusers"


def _convert_diffusers_with_optimum(
    model_path: str, output_dir: str, dtype: str = "fp16"
):
    """Convert diffusers model using optimum-intel."""
    try:
        from optimum.intel import OVStableDiffusionPipeline

        # Load and convert using optimum-intel
        pipeline = OVStableDiffusionPipeline.from_pretrained(
            model_path,
            export=True,
            compile=False,
            device="CPU",  # We'll compile later on target device
        )

        # Save to output directory
        pipeline.save_pretrained(output_dir)

        # Return path to the main model file
        return str(Path(output_dir) / "unet" / "openvino_model.xml")

    except ImportError:
        raise RuntimeError(
            "optimum-intel is required for diffusers models. "
            "Install with: pip install 'openvino-easy[stable-diffusion]'"
        )


def _convert_transformers_with_optimum(
    model_path: str, output_dir: str, model_id: str, dtype: str = "fp16"
):
    """Convert transformers model using optimum-intel."""
    try:
        from optimum.intel import (
            OVModelForCausalLM,
            OVModelForSequenceClassification,
            OVModelForQuestionAnswering,
        )
        from transformers import AutoTokenizer, AutoConfig

        # Try to determine the model task
        config = AutoConfig.from_pretrained(model_path)

        # Select appropriate OV model class based on architecture
        if hasattr(config, "num_labels") and config.num_labels > 1:
            # Classification model
            ov_model = OVModelForSequenceClassification.from_pretrained(
                model_path, export=True, compile=False
            )
        elif any(
            arch.lower().startswith(("gpt", "bloom", "opt"))
            for arch in getattr(config, "architectures", [])
        ):
            # Causal LM
            ov_model = OVModelForCausalLM.from_pretrained(
                model_path, export=True, compile=False
            )
        else:
            # Try causal LM as default
            ov_model = OVModelForCausalLM.from_pretrained(
                model_path, export=True, compile=False
            )

        # Save to output directory
        ov_model.save_pretrained(output_dir)

        # Find the OpenVINO model file
        model_files = list(Path(output_dir).glob("openvino_model.xml"))
        if not model_files:
            model_files = list(Path(output_dir).glob("*.xml"))

        if model_files:
            return str(model_files[0])
        else:
            raise RuntimeError("Could not find converted OpenVINO model file")

    except ImportError:
        raise RuntimeError(
            "optimum-intel is required for transformers model conversion. "
            "Install with: pip install 'openvino-easy[text]'"
        )


def _convert_with_direct_ov(model_path: str, model_format: str, dtype: str = "fp16"):
    """Convert model using direct OpenVINO conversion."""
    try:
        if model_format == "onnx":
            # Find ONNX file
            onnx_files = list(Path(model_path).glob("*.onnx"))
            if not onnx_files:
                raise RuntimeError("No ONNX files found")

            model = ov.Core().read_model(str(onnx_files[0]))

        else:
            # Try direct conversion for other formats
            model = ov.convert_model(model_path, compress_to_fp16=(dtype == "fp16"))

        return model

    except Exception as e:
        raise RuntimeError(f"Direct OpenVINO conversion failed: {e}")


# Keep legacy function name for compatibility
def _convert_with_optimum_intel(model_path: str, output_dir: str, dtype: str = "fp16"):
    """Legacy function - use _convert_diffusers_with_optimum instead."""
    return _convert_diffusers_with_optimum(model_path, output_dir, dtype)


def load_model(
    model_id_or_path: str, dtype: str = "fp16", cache_dir: Union[str, Path] = "~/.cache/oe"
):
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
        # Check if we can reach Hugging Face before downloading
        try:
            import requests

            response = requests.head("https://huggingface.co", timeout=10)
            if response.status_code >= 400:
                raise NetworkError("Cannot reach Hugging Face Hub")
        except Exception as e:
            raise NetworkError(f"Network connectivity issue: {e}")

        # Download model files
        local_model_path = _download_with_retry(model_id_or_path, cache_dir)

        local_path = Path(local_model_path)

        # Detect model type and use appropriate conversion method
        model_type = _detect_model_type(local_path)
        logging.info(f"🔍 Detected model type: {model_type}")

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                if model_type == "diffusers":
                    # Use optimum-intel for diffusers models
                    model_xml_path = _safe_model_conversion(
                        _convert_diffusers_with_optimum,
                        str(local_path),
                        temp_dir,
                        dtype,
                    )
                    model = ov.Core().read_model(model_xml_path)

                elif model_type in [
                    "transformers_optimum",
                    "transformers_vision",
                    "transformers_multimodal",
                    "transformers_audio",
                ]:
                    # Use optimum-intel for known compatible transformers models
                    logging.info(
                        f"🔄 Converting {model_type} model with optimum-intel..."
                    )
                    model_xml_path = _safe_model_conversion(
                        _convert_transformers_with_optimum,
                        str(local_path),
                        temp_dir,
                        model_id_or_path,
                        dtype,
                    )
                    model = ov.Core().read_model(model_xml_path)

                elif model_type in ["onnx", "onnx_text", "onnx_vision"]:
                    # Direct loading for ONNX models
                    logging.info(f"📥 Loading {model_type} model...")
                    model = _safe_model_conversion(
                        _convert_with_direct_ov, str(local_path), "onnx", dtype
                    )

                elif model_type in ["tensorflow"]:
                    # TensorFlow models - try direct conversion
                    logging.info("🔄 Converting TensorFlow model...")

                    def _convert_tensorflow_model(model_path, dtype):
                        # Find the main model file
                        pb_files = list(Path(model_path).glob("*.pb"))
                        if pb_files:
                            model_file = str(pb_files[0])
                        else:
                            model_file = str(model_path)

                        return ov.convert_model(
                            model_file, compress_to_fp16=(dtype == "fp16")
                        )

                    model = _safe_model_conversion(
                        _convert_tensorflow_model, str(local_path), dtype
                    )

                elif model_type in [
                    "transformers_direct",
                    "pytorch_transformers",
                    "safetensors_transformers",
                ]:
                    # Try direct OpenVINO conversion first, fallback to optimum-intel
                    logging.info(
                        f"⚠️  Attempting direct conversion for {model_type} model..."
                    )
                    try:
                        model = ov.convert_model(
                            str(local_path), compress_to_fp16=(dtype == "fp16")
                        )
                        logging.info("✅ Direct conversion succeeded")
                    except Exception as e:
                        # Try with optimum-intel as fallback
                        logging.warning(f"❌ Direct conversion failed: {e}")
                        logging.info("🔄 Trying optimum-intel conversion...")
                        try:
                            model_xml_path = _safe_model_conversion(
                                _convert_transformers_with_optimum,
                                str(local_path),
                                temp_dir,
                                model_id_or_path,
                                dtype,
                            )
                            model = ov.Core().read_model(model_xml_path)
                            logging.info("✅ Optimum-intel fallback succeeded")
                        except ModelConversionError as e2:
                            raise ModelConversionError(
                                f"Failed to convert {model_type} model '{model_id_or_path}': {e}. "
                                f"Optimum-intel fallback also failed: {e2}. "
                                f"This model format may not be supported."
                            ) from e2

                elif model_type in ["pytorch_native", "safetensors_native"]:
                    # Native PyTorch/safetensors models without transformers config
                    logging.info(f"⚠️  Converting native {model_type} model...")
                    try:
                        model = ov.convert_model(
                            str(local_path), compress_to_fp16=(dtype == "fp16")
                        )
                    except Exception as e:
                        model_name = (
                            model_id_or_path.split("/")[-1]
                            if "/" in model_id_or_path
                            else model_id_or_path
                        )
                        raise ModelConversionError(
                            f"❌ Native PyTorch model '{model_name}' conversion failed: {e}\n\n"
                            f"🔄 **Recommended Solutions:**\n"
                            f"1. **Convert to ONNX format** (most compatible):\n"
                            f"   ```python\n"
                            f"   import torch\n"
                            f"   model = torch.load('{model_name}')\n"
                            f"   dummy_input = torch.randn(1, 3, 224, 224)  # Adjust shape\n"
                            f"   torch.onnx.export(model, dummy_input, '{model_name}.onnx')\n"
                            f"   oe.load('{model_name}.onnx')  # Then use OpenVINO-Easy\n"
                            f"   ```\n\n"
                            f"2. **Upload to Hugging Face Hub** with config.json:\n"
                            f"   - Add model configuration and tokenizer files\n"
                            f"   - OpenVINO-Easy works best with HF Hub models\n\n"
                            f"3. **Use TorchScript** (.pt with torch.jit.save):\n"
                            f"   ```python\n"
                            f"   traced_model = torch.jit.trace(model, dummy_input)\n"
                            f"   torch.jit.save(traced_model, '{model_name}_scripted.pt')\n"
                            f"   ```\n\n"
                            f"📖 **Why this happens:** Native .pt/.pth files need model architecture info.\n"
                            f"💡 **Best practice:** Use Hugging Face Hub or ONNX for production models.\n"
                            f"🔗 **More help:** https://docs.openvino.ai/2025/openvino-workflow/model-preparation.html"
                        ) from e

                elif model_type in ["tflite", "coreml"]:
                    # Unsupported formats that need special handling
                    raise RuntimeError(
                        f"Model format '{model_type}' is not directly supported. "
                        f"Please convert to ONNX, OpenVINO IR, or a supported format first. "
                        f"For {model_type} conversion, see OpenVINO Model Optimizer documentation."
                    )

                else:  # model_type == "unknown"
                    # Last resort: try multiple conversion strategies with detailed guidance
                    logging.info(
                        "⚠️  Unknown model format. Analyzing structure and trying conversion strategies..."
                    )

                    # Provide detailed analysis of what was found
                    files_found = []
                    for pattern in [
                        "*.json",
                        "*.bin",
                        "*.pt",
                        "*.pth",
                        "*.onnx",
                        "*.pb",
                        "*.h5",
                        "*.safetensors",
                    ]:
                        files = list(Path(local_path).glob(pattern))
                        if files:
                            files_found.extend(
                                [f.name for f in files[:3]]
                            )  # Limit to first 3

                    logging.info(
                        f"📁 Files found: {', '.join(files_found[:10])}"
                    )  # Show first 10

                    conversion_attempts = []

                    # Strategy 1: Direct OpenVINO conversion
                    try:
                        logging.info("🔄 Strategy 1: Direct OpenVINO conversion...")
                        model = ov.convert_model(
                            str(local_path), compress_to_fp16=(dtype == "fp16")
                        )
                        logging.info("✅ Direct OpenVINO conversion succeeded")
                    except Exception as e1:
                        conversion_attempts.append(f"Direct conversion: {e1}")

                        # Strategy 2: Try optimum-intel if config.json exists
                        if (Path(local_path) / "config.json").exists():
                            try:
                                logging.info(
                                    "🔄 Strategy 2: Optimum-intel conversion (config.json found)..."
                                )
                                model_xml_path = _convert_transformers_with_optimum(
                                    str(local_path), temp_dir, model_id_or_path, dtype
                                )
                                model = ov.Core().read_model(model_xml_path)
                                logging.info("✅ Optimum-intel conversion succeeded")
                            except Exception as e2:
                                conversion_attempts.append(f"Optimum-intel: {e2}")
                        else:
                            conversion_attempts.append(
                                "Optimum-intel: No config.json found"
                            )

                        # Strategy 3: Try ONNX loading if ONNX files exist
                        onnx_files = list(Path(local_path).glob("*.onnx"))
                        if onnx_files and len(conversion_attempts) == 2:
                            try:
                                logging.info("🔄 Strategy 3: ONNX model loading...")
                                model = ov.Core().read_model(str(onnx_files[0]))
                                logging.info("✅ ONNX loading succeeded")
                            except Exception as e3:
                                conversion_attempts.append(f"ONNX loading: {e3}")

                        # All strategies failed - provide actionable error
                        if len(conversion_attempts) >= 2:
                            supported_formats = [
                                "🤗 **Transformers models** (Hugging Face Hub) - RECOMMENDED",
                                "🎨 **Diffusers pipelines** (Stable Diffusion, etc.)",
                                "🔄 **ONNX models** (.onnx files) - Universal format",
                                "🧠 **OpenVINO IR** (.xml/.bin files) - Fastest loading",
                                "🔧 **TensorFlow SavedModel** (.pb files)",
                            ]

                            model_name = (
                                model_id_or_path.split("/")[-1]
                                if "/" in model_id_or_path
                                else model_id_or_path
                            )

                            workflow_suggestions = []

                            # PyTorch-specific suggestions
                            if any(".pt" in f or ".pth" in f for f in files_found):
                                workflow_suggestions.extend(
                                    [
                                        "🔄 **For PyTorch models (.pt/.pth):**",
                                        f"   export to ONNX: torch.onnx.export(model, dummy_input, '{model_name}.onnx')",
                                        f"   then: oe.load('{model_name}.onnx')",
                                    ]
                                )

                            # SafeTensors suggestions
                            if any(".safetensors" in f for f in files_found):
                                workflow_suggestions.extend(
                                    [
                                        "🔒 **For SafeTensors models:** Add config.json file or upload to Hugging Face Hub"
                                    ]
                                )

                            # TensorFlow suggestions
                            if any(".h5" in f or ".pb" in f for f in files_found):
                                workflow_suggestions.extend(
                                    [
                                        "🔧 **For TensorFlow models:** Use SavedModel format or convert to ONNX"
                                    ]
                                )

                            # General suggestions
                            workflow_suggestions.extend(
                                [
                                    "📦 **Best Practice:** Upload models to Hugging Face Hub with proper config files",
                                    "🔗 **Model Conversion Guide:** https://docs.openvino.ai/2025/openvino-workflow/model-preparation.html",
                                    "🛠️ **Missing dependencies?** Try: pip install 'openvino-easy[full]'",
                                ]
                            )

                            error_msg = (
                                f"❌ **Model Conversion Failed**\n"
                                f"Model: '{model_id_or_path}'\n"
                                f"Files: {', '.join(files_found[:5])}{'...' if len(files_found) > 5 else ''}\n\n"
                                f"🔍 **Attempted conversions:**\n"
                                + "\n".join(
                                    f"  • {attempt}" for attempt in conversion_attempts
                                )
                                + "\n\n✅ **Supported formats:**\n"
                                + "\n".join(f"  {fmt}" for fmt in supported_formats)
                                + "\n\n🚀 **Recommended workflows:**\n"
                                + "\n".join(f"  {sug}" for sug in workflow_suggestions)
                                + "\n\n💬 **Need help?** Join our community: https://github.com/openvinotoolkit/openvino/discussions"
                            )
                            raise ModelConversionError(error_msg)

            except Exception as e:
                # Re-raise with model type context if not already a RuntimeError
                if not isinstance(e, RuntimeError):
                    raise RuntimeError(
                        f"Conversion failed for {model_type} model '{model_id_or_path}': {e}"
                    ) from e
                else:
                    raise

        # Verify model integrity before caching
        _verify_model_integrity(model, model_id_or_path)

        # Cache the converted model
        cache_path.mkdir(parents=True, exist_ok=True)
        try:
            ov.save_model(model, str(cache_path / "model.xml"))
        except Exception as e:
            raise ModelLoadError(
                f"Failed to cache model '{model_id_or_path}': {e}"
            ) from e

        # Save metadata
        metadata = {
            "model_id": model_id_or_path,
            "dtype": dtype,
            "ov_version": ov_version,
            "cache_key": cache_key,
            "model_type": model_type,
            "conversion_time": time.time(),
        }
        try:
            with open(cache_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save metadata for '{model_id_or_path}': {e}")

        return model

    except (
        ModelLoadError,
        ModelNotFoundError,
        ModelConversionError,
        NetworkError,
        UnsupportedModelError,
        CorruptedModelError,
    ):
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        # Catch any other unexpected errors
        raise ModelLoadError(
            f"Unexpected error loading model '{model_id_or_path}': {e}"
        ) from e
