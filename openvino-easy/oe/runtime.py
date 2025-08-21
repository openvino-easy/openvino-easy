"""Runtime wrapper for OpenVINO-Easy (ov.CompiledModel abstraction)."""

import numpy as np
from typing import Dict, Any, Union, Optional, List
from pathlib import Path
import warnings
import asyncio
import concurrent.futures
import threading
import os
import logging


class RuntimeWrapper:
    """Unified runtime wrapper for OpenVINO compiled models."""

    def __init__(
        self, compiled_model, device: str, model_info: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the runtime wrapper.

        Args:
            compiled_model: Compiled OpenVINO model
            device: Device name (NPU, GPU, CPU, etc.)
            model_info: Optional model metadata
        """
        self.compiled_model = compiled_model
        self.device = device
        self.model_info = model_info or {}

        # Create infer request for inference (OpenVINO 2025 API)
        self.infer_request = compiled_model.create_infer_request()

        # Create optimal number of async infer requests based on device and CPU cores
        optimal_pool_size = self._get_optimal_pool_size()
        self.async_infer_requests = [
            compiled_model.create_infer_request() for _ in range(optimal_pool_size)
        ]
        self._request_pool_lock = threading.Lock()
        self._available_requests = list(self.async_infer_requests)

        # Performance optimizations
        self._preprocessing_cache = {}  # Cache for repeated preprocessing
        self._cache_lock = threading.Lock()
        self._cache_max_size = 100  # Limit cache size to prevent memory bloat
        self._cache_hits = 0  # Track cache hit statistics
        self._cache_attempts = 0  # Track total cache attempts

        # Extract input/output information
        self.input_info = self._extract_input_info()
        self.output_info = self._extract_output_info()

        # Initialize tokenizer if model supports text
        self.tokenizer = None
        self._init_tokenizer()

    def _get_optimal_pool_size(self) -> int:
        """Get optimal size for inference request pool based on device and CPU cores."""
        try:
            cpu_count = os.cpu_count() or 4

            if self.device == "NPU":
                # NPU can handle more concurrent requests efficiently
                return min(cpu_count, 8)
            elif self.device == "GPU":
                # GPU benefits from multiple requests for batching
                return min(cpu_count // 2, 6)
            else:  # CPU
                # CPU performance scales with cores but with diminishing returns
                return min(cpu_count, 4)
        except:
            return 4  # Safe fallback

    def _get_optimal_worker_count(self, batch_size: int = 1) -> int:
        """Get optimal number of workers for batch processing."""
        try:
            cpu_count = os.cpu_count() or 4

            # Consider device capabilities
            if self.device == "NPU":
                # NPU can handle higher concurrency
                base_workers = min(cpu_count, 8)
            elif self.device == "GPU":
                # GPU benefits from parallel requests
                base_workers = min(cpu_count // 2, 6)
            else:  # CPU
                # CPU scaling depends on model complexity and cores
                base_workers = min(cpu_count, 4)

            # Adjust based on batch size - smaller batches can use more workers
            if batch_size <= 4:
                return base_workers
            elif batch_size <= 16:
                return max(base_workers // 2, 2)
            else:
                return max(base_workers // 4, 1)

        except:
            return 4  # Safe fallback

    def _extract_input_info(self) -> Dict[str, Dict[str, Any]]:
        """Extract input information from the compiled model."""
        input_info = {}
        for input_node in self.compiled_model.inputs:
            name = input_node.get_any_name()
            input_info[name] = {
                "shape": list(input_node.shape),
                "dtype": str(input_node.get_element_type()),
                "node": input_node,
            }
        return input_info

    def _extract_output_info(self) -> Dict[str, Dict[str, Any]]:
        """Extract output information from the compiled model."""
        output_info = {}
        for output_node in self.compiled_model.outputs:
            name = output_node.get_any_name()
            output_info[name] = {
                "shape": list(output_node.shape),
                "dtype": str(output_node.get_element_type()),
                "node": output_node,
            }
        return output_info

    def _init_tokenizer(self):
        """Initialize tokenizer if model supports text input."""
        try:
            # Check if we have model source path for tokenizer
            source_path = self.model_info.get("source_path")
            if not source_path:
                return

            # Try to load tokenizer from transformers
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(source_path)
                logging.info(f"✅ Loaded tokenizer for {source_path}")
            except Exception as e:
                # Check if it's a local path with tokenizer files
                local_path = Path(source_path)
                if local_path.exists():
                    try:
                        from transformers import AutoTokenizer

                        self.tokenizer = AutoTokenizer.from_pretrained(str(local_path))
                        logging.info(f"✅ Loaded local tokenizer from {local_path}")
                    except Exception:
                        if self._is_likely_text_model():
                            logging.warning(
                                f"⚠️  Failed to load tokenizer from {local_path}: {e}"
                            )
                            logging.info(
                                "💡 Consider installing transformers for better text processing: pip install transformers"
                            )

        except ImportError:
            # transformers not available - only warn if this looks like a text model
            if self._is_likely_text_model():
                logging.warning("⚠️  Transformers not available for tokenization")
                logging.info(
                    "💡 For better text processing, install transformers: pip install transformers"
                )
                logging.info("📝 Using basic tokenization fallback (reduced accuracy)")
            # For non-text models, silently continue without tokenizer

    def _get_optimal_sequence_length(self) -> int:
        """Determine optimal sequence length based on model inputs and tokenizer."""
        # First, try to get from tokenizer if available
        if self.tokenizer is not None:
            try:
                # Check tokenizer's model_max_length
                if (
                    hasattr(self.tokenizer, "model_max_length")
                    and self.tokenizer.model_max_length < 1000000
                ):
                    return min(
                        self.tokenizer.model_max_length, 2048
                    )  # Cap at 2048 for performance
            except:
                pass

        # Check model input shapes for sequence dimension
        for input_name, input_info in self.input_info.items():
            shape = input_info["shape"]

            # Look for sequence dimension in common positions
            if len(shape) == 2:  # [batch, seq_len]
                seq_dim = shape[1]
                if seq_dim > 0 and seq_dim <= 4096:  # Reasonable sequence length
                    return seq_dim

            elif len(shape) == 3:  # [batch, seq_len, hidden] or similar
                seq_dim = shape[1]
                if seq_dim > 0 and seq_dim <= 4096:
                    return seq_dim

        # Model-specific defaults based on common architectures
        source_path = self.model_info.get("source_path", "").lower()

        if any(arch in source_path for arch in ["bert", "roberta", "distilbert"]):
            return 512  # BERT family default
        elif any(arch in source_path for arch in ["gpt", "bloom", "opt"]):
            return 1024  # GPT family can handle longer sequences
        elif "t5" in source_path:
            return 512  # T5 default
        elif any(arch in source_path for arch in ["albert", "electra"]):
            return 512  # Similar to BERT
        else:
            return 256  # Conservative default for unknown models

    def _preprocess_input(
        self, input_data: Union[str, np.ndarray, Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """
        Preprocess input data based on model type and requirements.

        Args:
            input_data: Raw input (string prompt, numpy array, or dict)

        Returns:
            Preprocessed input dictionary
        """
        # For string inputs, check cache first (strings are hashable and commonly repeated)
        if (
            isinstance(input_data, str) and len(input_data) < 1000
        ):  # Only cache reasonably sized strings
            cache_key = hash(input_data)

            with self._cache_lock:
                self._cache_attempts += 1
                if cache_key in self._preprocessing_cache:
                    self._cache_hits += 1
                    return self._preprocessing_cache[
                        cache_key
                    ].copy()  # Return copy to avoid mutation

        # If input is already a dict, validate and return
        if isinstance(input_data, dict):
            return self._validate_input_dict(input_data)

        # If input is a string, treat as prompt for text-based models
        if isinstance(input_data, str):
            result = self._preprocess_text_input(input_data)

            # Cache the result for future use (if string is reasonable size)
            if len(input_data) < 1000:
                cache_key = hash(input_data)
                with self._cache_lock:
                    # Implement LRU-like cache eviction
                    if len(self._preprocessing_cache) >= self._cache_max_size:
                        # Remove oldest entry (simple FIFO for now)
                        oldest_key = next(iter(self._preprocessing_cache))
                        del self._preprocessing_cache[oldest_key]

                    self._preprocessing_cache[cache_key] = result.copy()

            return result

        # If input is numpy array, handle single input models
        if isinstance(input_data, np.ndarray):
            return self._preprocess_array_input(input_data)

        # Check if input is an audio file path
        if isinstance(input_data, (str, Path)) and self._is_audio_file(input_data):
            return self._preprocess_audio_input(input_data)

        raise ValueError(f"Unsupported input type: {type(input_data)}")

    def _validate_input_dict(self, input_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Validate and convert input dictionary."""
        processed_inputs = {}

        for input_name, input_value in input_dict.items():
            if input_name not in self.input_info:
                raise ValueError(f"Unknown input name: {input_name}")

            # Convert to numpy array if needed
            if not isinstance(input_value, np.ndarray):
                input_value = np.array(input_value)

            # Validate shape
            expected_shape = self.input_info[input_name]["shape"]
            if not self._shapes_compatible(input_value.shape, expected_shape):
                raise ValueError(
                    f"Input shape mismatch for {input_name}: "
                    f"expected {expected_shape}, got {input_value.shape}"
                )

            processed_inputs[input_name] = input_value.astype(np.float32)

        return processed_inputs

    def _preprocess_text_input(self, text: str) -> Dict[str, np.ndarray]:
        """Preprocess text input for text-based models."""
        # Use real tokenizer if available
        if self.tokenizer is not None:
            return self._tokenize_with_transformers(text)

        # Fallback: try to detect model type and use appropriate preprocessing
        return self._tokenize_fallback(text)

    def _tokenize_with_transformers(self, text: str) -> Dict[str, np.ndarray]:
        """Tokenize text using transformers tokenizer."""
        try:
            # Get model input requirements
            input_names = list(self.input_info.keys())

            # Get model-specific max length or use reasonable default
            max_length = self._get_optimal_sequence_length()

            # Tokenize text
            encoded = self.tokenizer(
                text,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            # Map tokenizer outputs to model inputs
            processed_inputs = {}

            for input_name in input_names:
                expected_shape = self.input_info[input_name]["shape"]

                # Common input name mappings
                if input_name in ["input_ids", "input"] and "input_ids" in encoded:
                    tensor = encoded["input_ids"]
                elif (
                    input_name in ["attention_mask", "mask"]
                    and "attention_mask" in encoded
                ):
                    tensor = encoded["attention_mask"]
                elif (
                    input_name in ["token_type_ids", "segment_ids"]
                    and "token_type_ids" in encoded
                ):
                    tensor = encoded["token_type_ids"]
                else:
                    # Use first available tensor as fallback
                    tensor = list(encoded.values())[0]

                # Reshape to match expected shape
                tensor = self._reshape_to_expected(tensor, expected_shape)
                processed_inputs[input_name] = tensor.astype(np.float32)

            return processed_inputs

        except Exception as e:
            warnings.warn(
                f"Tokenizer failed: {e}. Falling back to simple tokenization."
            )
            return self._tokenize_fallback(text)

    def _tokenize_fallback(self, text: str) -> Dict[str, np.ndarray]:
        """Enhanced fallback tokenization with better subword approximation."""
        warnings.warn(
            "Using basic tokenization fallback. For better accuracy, install transformers: "
            "pip install transformers",
            UserWarning,
        )

        # First, check if this is likely a text model
        if not self._is_likely_text_model():
            warnings.warn(
                f"Text input '{text[:50]}...' provided to likely non-text model. "
                f"Model input shapes: {[info['shape'] for info in self.input_info.values()]}",
                UserWarning,
            )
            # Still try to process it, but warn the user

        # Enhanced tokenization with basic subword handling
        processed_inputs = {}

        for input_name, input_info in self.input_info.items():
            expected_shape = input_info["shape"]

            # Generate tokens with basic BPE-like splitting
            tokens = self._basic_tokenize(text)

            # Handle different input shapes
            if len(expected_shape) == 2:  # [batch_size, seq_len]
                max_length = (
                    expected_shape[1]
                    if expected_shape[1] > 0
                    else self._get_optimal_sequence_length()
                )

                # Pad or truncate
                if len(tokens) > max_length:
                    tokens = tokens[: max_length - 1] + [
                        tokens[-1]
                    ]  # Keep last token (often EOS)
                else:
                    tokens.extend([0] * (max_length - len(tokens)))

                input_tensor = np.array([tokens], dtype=np.float32)

            elif len(expected_shape) == 1:  # [seq_len]
                max_length = expected_shape[0] if expected_shape[0] > 0 else len(tokens)

                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                else:
                    tokens.extend([0] * (max_length - len(tokens)))

                input_tensor = np.array(tokens, dtype=np.float32)

            else:
                # For non-standard shapes, try to create reasonable data
                if len(expected_shape) == 3:  # Maybe embeddings? [batch, seq, dim]
                    (
                        expected_shape[1]
                        if expected_shape[1] > 0
                        else min(len(tokens), 128)
                    )
                    # Create simple embeddings (normalized random vectors)
                    embeddings = np.random.normal(0, 0.1, expected_shape).astype(
                        np.float32
                    )
                    input_tensor = embeddings
                elif len(expected_shape) == 4:  # Likely vision model
                    input_tensor = np.random.normal(0.45, 0.25, expected_shape).astype(
                        np.float32
                    )
                    input_tensor = np.clip(input_tensor, 0, 1)
                else:
                    # Unknown shape, create random data
                    input_tensor = np.random.normal(0, 0.1, expected_shape).astype(
                        np.float32
                    )

            processed_inputs[input_name] = input_tensor

        return processed_inputs

    def _basic_tokenize(self, text: str) -> List[int]:
        """Enhanced basic tokenization with improved subword approximation."""
        # Clean and normalize text
        text = text.strip()

        # Handle contractions and common patterns
        import re

        # Normalize contractions before tokenization
        text = re.sub(r"won't", "will not", text, flags=re.IGNORECASE)
        text = re.sub(r"can't", "can not", text, flags=re.IGNORECASE)
        text = re.sub(r"n't", " not", text, flags=re.IGNORECASE)
        text = re.sub(r"'re", " are", text, flags=re.IGNORECASE)
        text = re.sub(r"'ve", " have", text, flags=re.IGNORECASE)
        text = re.sub(r"'ll", " will", text, flags=re.IGNORECASE)
        text = re.sub(r"'d", " would", text, flags=re.IGNORECASE)

        # Now convert to lowercase
        text = text.lower()

        # Enhanced tokenization with better pattern recognition
        # Split on whitespace and punctuation, preserving structure
        tokens = re.findall(r"\w+|[.,!?;:\-\(\)\[\]\"\'`]", text)

        # Enhanced vocabulary with more common words and subword patterns
        common_vocab = {
            # Special tokens
            "<pad>": 0,
            "<unk>": 1,
            "<cls>": 101,
            "<sep>": 102,
            # Most common English words
            "the": 2,
            "a": 3,
            "an": 4,
            "and": 5,
            "or": 6,
            "but": 7,
            "in": 8,
            "on": 9,
            "at": 10,
            "to": 11,
            "of": 12,
            "for": 13,
            "with": 14,
            "by": 15,
            "from": 16,
            "about": 17,
            "into": 18,
            "through": 19,
            "during": 20,
            "before": 21,
            "after": 22,
            "is": 23,
            "are": 24,
            "was": 25,
            "were": 26,
            "be": 27,
            "been": 28,
            "being": 29,
            "have": 30,
            "has": 31,
            "had": 32,
            "do": 33,
            "does": 34,
            "did": 35,
            "will": 36,
            "would": 37,
            "could": 38,
            "should": 39,
            "may": 40,
            "might": 41,
            "must": 42,
            "can": 43,
            "not": 44,
            "no": 45,
            "yes": 46,
            "all": 47,
            "this": 48,
            "that": 49,
            "these": 50,
            "those": 51,
            "i": 52,
            "you": 53,
            "he": 54,
            "she": 55,
            "it": 56,
            "we": 57,
            "they": 58,
            "me": 59,
            "him": 60,
            "her": 61,
            "us": 62,
            "them": 63,
            "my": 64,
            "your": 65,
            "his": 66,
            "our": 67,
            "their": 68,
            "what": 69,
            "where": 70,
            "when": 71,
            "why": 72,
            "how": 73,
            "who": 74,
            "which": 75,
            # Common punctuation
            ".": 76,
            ",": 77,
            "!": 78,
            "?": 79,
            ";": 80,
            ":": 81,
            "-": 82,
            "(": 83,
            ")": 84,
            "[": 85,
            "]": 86,
            '"': 87,
            "'": 88,
        }

        token_ids = []
        vocab_base = 200  # Start custom IDs after reserved range

        for token in tokens:
            if token in common_vocab:
                token_ids.append(common_vocab[token])
            elif len(token) == 1:
                # Single character fallback - use ASCII with offset
                token_ids.append(min(ord(token) % 100 + 100, 199))
            else:
                # For unknown words, use character-aware hashing for more stability
                # Include token length to reduce collisions
                char_sum = sum(ord(c) for c in token[:8])  # Limit to first 8 chars
                length_factor = min(len(token), 20)  # Cap length factor
                token_id = ((char_sum * 31 + length_factor) % 9800) + vocab_base
                token_ids.append(token_id)

        # Add special tokens with better padding for empty inputs
        if token_ids:
            token_ids = [common_vocab["<cls>"]] + token_ids + [common_vocab["<sep>"]]
        else:
            # For empty input, return meaningful tokens
            token_ids = [
                common_vocab["<cls>"],
                common_vocab["<pad>"],
                common_vocab["<sep>"],
            ]

        return token_ids

    def _is_likely_text_model(self) -> bool:
        """Check if this is likely a text model based on input characteristics."""
        for input_info in self.input_info.values():
            shape = input_info["shape"]

            # Text models typically have:
            # - 1D or 2D inputs
            # - Sequence-like dimensions (often 512, 1024, etc.)
            # - Not image-like dimensions (224x224, etc.)

            if len(shape) <= 2:
                # Check for sequence-like dimensions
                for dim in shape:
                    if dim > 0 and dim in [128, 256, 512, 1024, 2048]:
                        return True
                    elif dim > 0 and 50 <= dim <= 5000:  # Reasonable sequence length
                        return True
            elif len(shape) == 4:
                # Definitely looks like a vision model
                return False

        # If we have a tokenizer, it's likely text
        if self.tokenizer is not None:
            return True

        # Check source path for text model indicators
        source_path = self.model_info.get("source_path", "").lower()
        text_indicators = [
            "bert",
            "gpt",
            "roberta",
            "distilbert",
            "albert",
            "bloom",
            "opt",
            "t5",
        ]
        return any(indicator in source_path for indicator in text_indicators)

    def _reshape_to_expected(
        self, tensor: np.ndarray, expected_shape: list
    ) -> np.ndarray:
        """Reshape tensor to match expected shape, handling dynamic dimensions."""
        # Handle dynamic dimensions (-1)
        target_shape = []
        for dim in expected_shape:
            if dim == -1:
                # Use corresponding dimension from tensor, or 1 if not available
                if len(target_shape) < len(tensor.shape):
                    target_shape.append(tensor.shape[len(target_shape)])
                else:
                    target_shape.append(1)
            else:
                target_shape.append(dim)

        # Reshape or pad/truncate as needed
        if tensor.shape == tuple(target_shape):
            return tensor

        # For sequence models, handle length dimension
        if len(target_shape) == 2 and len(tensor.shape) == 2:
            batch_size, seq_len = target_shape
            if tensor.shape[1] != seq_len and seq_len > 0:
                if tensor.shape[1] > seq_len:
                    # Truncate
                    tensor = tensor[:, :seq_len]
                else:
                    # Pad
                    padding = np.zeros(
                        (tensor.shape[0], seq_len - tensor.shape[1]), dtype=tensor.dtype
                    )
                    tensor = np.concatenate([tensor, padding], axis=1)

        # Final reshape attempt
        try:
            return tensor.reshape(target_shape)
        except ValueError:
            # If reshape fails, return tensor as-is and let the model handle it
            warnings.warn(
                f"Could not reshape tensor from {tensor.shape} to {target_shape}"
            )
            return tensor

    def _preprocess_array_input(self, array: np.ndarray) -> Dict[str, np.ndarray]:
        """Preprocess numpy array input."""
        if len(self.input_info) != 1:
            raise ValueError(
                f"Model has {len(self.input_info)} inputs, but single array provided"
            )

        input_name = list(self.input_info.keys())[0]
        expected_shape = self.input_info[input_name]["shape"]

        # Reshape if needed
        if array.shape != tuple(expected_shape):
            array = self._reshape_to_expected(array, expected_shape)

        return {input_name: array.astype(np.float32)}

    def _shapes_compatible(self, actual_shape: tuple, expected_shape: list) -> bool:
        """Check if shapes are compatible (handling dynamic dimensions)."""
        if len(actual_shape) != len(expected_shape):
            return False

        for actual, expected in zip(actual_shape, expected_shape):
            if expected == -1:  # Dynamic dimension
                continue
            if actual != expected:
                return False

        return True

    def _postprocess_output(self, output_data: Dict[str, np.ndarray]) -> Any:
        """
        Postprocess output data based on model type.

        Args:
            output_data: Raw model output

        Returns:
            Postprocessed output
        """
        # For single output models, return the output directly
        if len(output_data) == 1:
            output = list(output_data.values())[0]
            return self._format_single_output(output)

        # For multiple outputs, return as dict
        return {
            name: self._format_single_output(output)
            for name, output in output_data.items()
        }

    def _format_single_output(self, output: np.ndarray) -> Any:
        """Format a single output tensor."""
        # Remove batch dimension if it's 1
        if output.ndim > 1 and output.shape[0] == 1:
            output = output.squeeze(0)

        # For classification-like outputs, return probabilities
        if output.ndim == 1 and len(output) > 1:
            # Check if it looks like logits (values can be negative, not normalized)
            if np.any(output < 0) or not np.allclose(output.sum(), 1.0, atol=1e-3):
                # Apply softmax to convert logits to probabilities
                output = self._softmax(output)
            return output.tolist()

        # For single values, return scalar
        if output.size == 1:
            return float(output.item())

        # Otherwise return as list
        return output.tolist()

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to array."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def infer(
        self, input_data: Union[str, np.ndarray, Dict[str, Any]], **kwargs
    ) -> Any:
        """
        Run inference on the model.

        Args:
            input_data: Input data (string, numpy array, or dict)
            **kwargs: Additional inference parameters

        Returns:
            Model output

        Raises:
            RuntimeError: If model has been unloaded
        """
        if not self.is_loaded():
            raise RuntimeError(
                "Model has been unloaded and cannot be used for inference. "
                "Please load a new model using oe.load() or create a new RuntimeWrapper instance."
            )

        # Preprocess input
        processed_input = self._preprocess_input(input_data)

        # Run inference using infer_request (OpenVINO 2025 API)
        self.infer_request.infer(processed_input)

        # Get output data from infer_request
        output_data = {}
        for output_node in self.compiled_model.outputs:
            output_name = output_node.get_any_name()
            output_data[output_name] = self.infer_request.get_output_tensor(
                output_node.index
            ).data

        # Postprocess output
        return self._postprocess_output(output_data)

    async def infer_async(
        self, input_data: Union[str, np.ndarray, Dict[str, Any]], **kwargs
    ) -> Any:
        """
        Run asynchronous inference on the model for better throughput.

        Args:
            input_data: Input data (string, numpy array, or dict)
            **kwargs: Additional inference parameters

        Returns:
            Model output (awaitable)

        Raises:
            RuntimeError: If model has been unloaded
        """
        if not self.is_loaded():
            raise RuntimeError(
                "Model has been unloaded and cannot be used for inference. "
                "Please load a new model using oe.load() or create a new RuntimeWrapper instance."
            )

        # Preprocess input
        processed_input = self._preprocess_input(input_data)

        # Run async inference in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._run_single_inference, processed_input)
            result = await loop.run_in_executor(None, future.result)

        return result

    def _run_single_inference(self, processed_input: Dict[str, np.ndarray]) -> Any:
        """Helper method to run single inference with request pooling."""
        # Get an available request from the pool
        infer_request = self._get_available_request()

        try:
            # Run inference
            infer_request.infer(processed_input)

            # Get output data
            output_data = {}
            for output_node in self.compiled_model.outputs:
                output_name = output_node.get_any_name()
                output_data[output_name] = infer_request.get_output_tensor(
                    output_node.index
                ).data.copy()

            # Postprocess output
            return self._postprocess_output(output_data)

        finally:
            # Return request to pool
            self._return_request(infer_request)

    def _get_available_request(self):
        """Get an available inference request from the pool with improved management."""
        with self._request_pool_lock:
            if self._available_requests:
                return self._available_requests.pop()
            else:
                # If no requests available, create a temporary one
                # This handles high concurrency gracefully
                temp_request = self.compiled_model.create_infer_request()
                temp_request._is_temporary = True  # Mark as temporary
                return temp_request

    def _return_request(self, request):
        """Return an inference request to the pool with better management."""
        with self._request_pool_lock:
            # Only return permanent requests to the pool
            if not getattr(request, "_is_temporary", False):
                if len(self._available_requests) < len(self.async_infer_requests):
                    self._available_requests.append(request)
            # Temporary requests are automatically garbage collected

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about the request pool for monitoring."""
        with self._request_pool_lock:
            return {
                "pool_size": len(self.async_infer_requests),
                "available_requests": len(self._available_requests),
                "active_requests": len(self.async_infer_requests)
                - len(self._available_requests),
                "device": self.device,
                "cache_size": len(self._preprocessing_cache),
                "cache_hit_rate": getattr(self, "_cache_hits", 0)
                / max(getattr(self, "_cache_attempts", 1), 1),
            }

    async def infer_batch_async(
        self,
        input_batch: List[Union[str, np.ndarray, Dict[str, Any]]],
        max_concurrent: Optional[int] = None,
        **kwargs,
    ) -> List[Any]:
        """
        Run asynchronous batch inference with controlled concurrency.

        Args:
            input_batch: List of input data
            max_concurrent: Maximum number of concurrent inference requests (auto-detected if None)
            **kwargs: Additional inference parameters

        Returns:
            List of model outputs in the same order as inputs
        """
        if not input_batch:
            return []

        # Auto-detect optimal concurrency if not specified
        if max_concurrent is None:
            max_concurrent = self._get_optimal_worker_count(len(input_batch))

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single(input_data):
            async with semaphore:
                return await self.infer_async(input_data, **kwargs)

        # Create tasks for all inputs
        tasks = [process_single(input_data) for input_data in input_batch]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                warnings.warn(f"Batch item {i} failed: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)

        return processed_results

    def infer_batch(
        self,
        input_batch: List[Union[str, np.ndarray, Dict[str, Any]]],
        max_workers: Optional[int] = None,
        **kwargs,
    ) -> List[Any]:
        """
        Run synchronous batch inference with thread pool for better throughput.

        Args:
            input_batch: List of input data
            max_workers: Maximum number of worker threads (auto-detected if None)
            **kwargs: Additional inference parameters

        Returns:
            List of model outputs in the same order as inputs
        """
        if not input_batch:
            return []

        # Auto-detect optimal worker count if not specified
        if max_workers is None:
            max_workers = self._get_optimal_worker_count(len(input_batch))

        # Use ThreadPoolExecutor for CPU-bound inference tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._safe_infer, input_data, **kwargs): i
                for i, input_data in enumerate(input_batch)
            }

            # Collect results in order
            results = [None] * len(input_batch)
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    warnings.warn(f"Batch item {index} failed: {e}")
                    results[index] = None

        return results

    def _safe_infer(
        self, input_data: Union[str, np.ndarray, Dict[str, Any]], **kwargs
    ) -> Any:
        """Safe inference wrapper that handles exceptions."""
        try:
            # Preprocess input
            processed_input = self._preprocess_input(input_data)
            return self._run_single_inference(processed_input)
        except Exception as e:
            # Re-raise the exception to be handled by the caller
            raise e

    def get_input_info(self) -> Dict[str, Dict[str, Any]]:
        """Get input information."""
        return self.input_info.copy()

    def get_output_info(self) -> Dict[str, Dict[str, Any]]:
        """Get output information."""
        return self.output_info.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information including performance stats."""
        base_info = {
            "device": self.device,
            "input_info": self.input_info,
            "output_info": self.output_info,
            "has_tokenizer": self.tokenizer is not None,
            **self.model_info,
        }

        # Add performance statistics
        base_info.update(self.get_pool_stats())

        return base_info

    def clear_cache(self):
        """Clear preprocessing cache to free memory."""
        with self._cache_lock:
            cache_size = len(self._preprocessing_cache)
            self._preprocessing_cache.clear()
            self._cache_hits = 0
            self._cache_attempts = 0
            logging.info(f"Cleared preprocessing cache ({cache_size} entries)")

    def unload(self):
        """
        Explicitly unload the model and free all associated resources.

        This method:
        - Releases OpenVINO compiled model and inference requests
        - Clears preprocessing cache
        - Frees tokenizer resources
        - Marks the runtime as unloaded

        After calling unload(), this RuntimeWrapper instance should not be used
        for inference until a new model is loaded.
        """
        logging.info(f"Unloading model from {self.device}...")

        # Clear inference requests pool
        with self._request_pool_lock:
            self._available_requests.clear()
            self.async_infer_requests.clear()

        # Release main inference request
        if hasattr(self, "infer_request") and self.infer_request is not None:
            del self.infer_request
            self.infer_request = None

        # Release compiled model
        if hasattr(self, "compiled_model") and self.compiled_model is not None:
            del self.compiled_model
            self.compiled_model = None

        # Clear tokenizer
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Clear preprocessing cache
        self.clear_cache()

        # Clear model info
        self.input_info.clear()
        self.output_info.clear()
        self.model_info.clear()

        logging.info("Model unloaded successfully")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically unload model."""
        self.unload()
        return False  # Don't suppress exceptions

    def is_loaded(self) -> bool:
        """Check if the model is currently loaded and ready for inference."""
        return (
            hasattr(self, "compiled_model")
            and self.compiled_model is not None
            and hasattr(self, "infer_request")
            and self.infer_request is not None
        )

    def _is_audio_file(self, file_path: Union[str, Path]) -> bool:
        """Check if the given path is an audio file."""
        if not isinstance(file_path, (str, Path)):
            return False

        path = Path(file_path)
        if not path.exists():
            return False

        audio_extensions = {
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".m4a",
            ".aac",
            ".wma",
            ".mp4",
            ".avi",
        }
        return path.suffix.lower() in audio_extensions

    def _preprocess_audio_input(
        self, audio_path: Union[str, Path]
    ) -> Dict[str, np.ndarray]:
        """Preprocess audio file for audio models."""
        # Try to use librosa for audio processing if available
        try:
            import librosa

            # Load audio file - Whisper expects 16kHz sampling rate
            audio_data, sample_rate = librosa.load(str(audio_path), sr=16000)

            # Convert to float32 and ensure correct shape
            audio_data = audio_data.astype(np.float32)

            # For most audio models, input shape is [batch_size, sequence_length]
            if len(audio_data.shape) == 1:
                audio_data = audio_data[np.newaxis, :]  # Add batch dimension

            # Find the audio input name (usually 'input' or 'input_features')
            audio_input_name = self._get_audio_input_name()

            return {audio_input_name: audio_data}

        except ImportError:
            # Fallback: try basic audio loading with wave for WAV files
            import wave

            path = Path(audio_path)
            if path.suffix.lower() != ".wav":
                raise ValueError(
                    f"Audio file '{audio_path}' is not supported. "
                    f"Install librosa for broader audio format support: "
                    f"pip install librosa"
                )

            try:
                with wave.open(str(audio_path), "rb") as wav_file:
                    frames = wav_file.readframes(-1)
                    sample_rate = wav_file.getframerate()
                    n_channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()

                    # Convert to numpy array
                    if sample_width == 1:
                        dtype = np.uint8
                    elif sample_width == 2:
                        dtype = np.int16
                    elif sample_width == 4:
                        dtype = np.int32
                    else:
                        raise ValueError(f"Unsupported sample width: {sample_width}")

                    audio_data = np.frombuffer(frames, dtype=dtype)

                    # Handle stereo to mono conversion
                    if n_channels == 2:
                        audio_data = audio_data.reshape(-1, 2).mean(axis=1)

                    # Normalize to [-1, 1] range
                    if dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                        if dtype == np.int16:
                            audio_data /= 32768.0
                        elif dtype == np.int32:
                            audio_data /= 2147483648.0
                        elif dtype == np.uint8:
                            audio_data = (audio_data - 128) / 128.0

                    # Resample to 16kHz if needed (basic resampling)
                    if sample_rate != 16000:
                        logging.warning(
                            f"⚠️  Audio sample rate is {sample_rate}Hz, expected 16kHz. "
                            f"Install librosa for proper resampling: pip install librosa"
                        )
                        # Basic resampling (not ideal but functional)
                        target_length = int(len(audio_data) * 16000 / sample_rate)
                        audio_data = np.interp(
                            np.linspace(0, len(audio_data), target_length),
                            np.arange(len(audio_data)),
                            audio_data,
                        ).astype(np.float32)

                    # Add batch dimension
                    audio_data = audio_data[np.newaxis, :]

                    # Find the audio input name
                    audio_input_name = self._get_audio_input_name()

                    return {audio_input_name: audio_data}

            except Exception as e:
                raise ValueError(f"Failed to process WAV file '{audio_path}': {e}")

        except Exception as e:
            raise ValueError(f"Failed to process audio file '{audio_path}': {e}")

    def _get_audio_input_name(self) -> str:
        """Get the name of the audio input tensor."""
        # Common names for audio inputs in different models
        common_audio_names = ["input_features", "input", "audio", "waveform", "speech"]

        for name in common_audio_names:
            if name in self.input_info:
                return name

        # If no common name found, use the first input
        if self.input_info:
            return next(iter(self.input_info.keys()))

        raise ValueError("No suitable audio input found in model")
