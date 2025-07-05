"""Runtime wrapper for OpenVINO-Easy (ov.CompiledModel abstraction)."""

import numpy as np
from typing import Dict, Any, Union, Optional
from pathlib import Path
import json
import warnings

class RuntimeWrapper:
    """Unified runtime wrapper for OpenVINO compiled models."""
    
    def __init__(self, compiled_model, device: str, model_info: Optional[Dict[str, Any]] = None):
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
        
        # Create infer request for inference (OpenVINO 2024 API)
        self.infer_request = compiled_model.create_infer_request()
        
        # Extract input/output information
        self.input_info = self._extract_input_info()
        self.output_info = self._extract_output_info()
        
        # Initialize tokenizer if model supports text
        self.tokenizer = None
        self._init_tokenizer()
        
    def _extract_input_info(self) -> Dict[str, Dict[str, Any]]:
        """Extract input information from the compiled model."""
        input_info = {}
        for input_node in self.compiled_model.inputs:
            name = input_node.get_any_name()
            input_info[name] = {
                "shape": list(input_node.shape),
                "dtype": str(input_node.get_element_type()),
                "node": input_node
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
                "node": output_node
            }
        return output_info
    
    def _init_tokenizer(self):
        """Initialize tokenizer if model supports text input."""
        try:
            # Check if we have model source path for tokenizer
            source_path = self.model_info.get('source_path')
            if not source_path:
                return
                
            # Try to load tokenizer from transformers
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(source_path)
                print(f"✅ Loaded tokenizer for {source_path}")
            except Exception as e:
                # Check if it's a local path with tokenizer files
                local_path = Path(source_path)
                if local_path.exists():
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(str(local_path))
                        print(f"✅ Loaded local tokenizer from {local_path}")
                    except Exception:
                        pass
                        
        except ImportError:
            # transformers not available
            pass
    
    def _preprocess_input(self, input_data: Union[str, np.ndarray, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Preprocess input data based on model type and requirements.
        
        Args:
            input_data: Raw input (string prompt, numpy array, or dict)
            
        Returns:
            Preprocessed input dictionary
        """
        # If input is already a dict, validate and return
        if isinstance(input_data, dict):
            return self._validate_input_dict(input_data)
        
        # If input is a string, treat as prompt for text-based models
        if isinstance(input_data, str):
            return self._preprocess_text_input(input_data)
        
        # If input is numpy array, handle single input models
        if isinstance(input_data, np.ndarray):
            return self._preprocess_array_input(input_data)
        
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
                raise ValueError(f"Input shape mismatch for {input_name}: "
                               f"expected {expected_shape}, got {input_value.shape}")
            
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
            
            # Tokenize text
            encoded = self.tokenizer(
                text,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512  # Common default
            )
            
            # Map tokenizer outputs to model inputs
            processed_inputs = {}
            
            for input_name in input_names:
                expected_shape = self.input_info[input_name]["shape"]
                
                # Common input name mappings
                if input_name in ["input_ids", "input"] and "input_ids" in encoded:
                    tensor = encoded["input_ids"]
                elif input_name in ["attention_mask", "mask"] and "attention_mask" in encoded:
                    tensor = encoded["attention_mask"]
                elif input_name in ["token_type_ids", "segment_ids"] and "token_type_ids" in encoded:
                    tensor = encoded["token_type_ids"]
                else:
                    # Use first available tensor as fallback
                    tensor = list(encoded.values())[0]
                
                # Reshape to match expected shape
                tensor = self._reshape_to_expected(tensor, expected_shape)
                processed_inputs[input_name] = tensor.astype(np.float32)
            
            return processed_inputs
            
        except Exception as e:
            warnings.warn(f"Tokenizer failed: {e}. Falling back to simple tokenization.")
            return self._tokenize_fallback(text)
    
    def _tokenize_fallback(self, text: str) -> Dict[str, np.ndarray]:
        """Fallback tokenization for when transformers tokenizer is not available."""
        # Simple word-level tokenization
        words = text.lower().split()
        
        # Create a simple vocabulary mapping
        vocab = {word: i + 1 for i, word in enumerate(set(words))}  # Start from 1, 0 for padding
        token_ids = [vocab.get(word, 0) for word in words]
        
        # Get expected input shape
        input_name = list(self.input_info.keys())[0]
        expected_shape = self.input_info[input_name]["shape"]
        
        # Handle different input shapes
        if len(expected_shape) == 2:  # [batch_size, seq_len]
            max_length = expected_shape[1] if expected_shape[1] > 0 else 128
            
            # Pad or truncate
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            else:
                token_ids.extend([0] * (max_length - len(token_ids)))
            
            input_tensor = np.array([token_ids], dtype=np.float32)
            
        elif len(expected_shape) == 1:  # [seq_len]
            max_length = expected_shape[0] if expected_shape[0] > 0 else len(token_ids)
            
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            else:
                token_ids.extend([0] * (max_length - len(token_ids)))
            
            input_tensor = np.array(token_ids, dtype=np.float32)
            
        else:
            # For other shapes, create random data (vision models)
            input_tensor = np.random.randn(*expected_shape).astype(np.float32)
            warnings.warn(f"Text input provided to non-text model. Using random data for shape {expected_shape}")
        
        return {input_name: input_tensor}
    
    def _reshape_to_expected(self, tensor: np.ndarray, expected_shape: list) -> np.ndarray:
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
                    padding = np.zeros((tensor.shape[0], seq_len - tensor.shape[1]), dtype=tensor.dtype)
                    tensor = np.concatenate([tensor, padding], axis=1)
        
        # Final reshape attempt
        try:
            return tensor.reshape(target_shape)
        except ValueError:
            # If reshape fails, return tensor as-is and let the model handle it
            warnings.warn(f"Could not reshape tensor from {tensor.shape} to {target_shape}")
            return tensor
    
    def _preprocess_array_input(self, array: np.ndarray) -> Dict[str, np.ndarray]:
        """Preprocess numpy array input."""
        if len(self.input_info) != 1:
            raise ValueError(f"Model has {len(self.input_info)} inputs, "
                           f"but single array provided")
        
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
        return {name: self._format_single_output(output) 
                for name, output in output_data.items()}
    
    def _format_single_output(self, output: np.ndarray) -> Any:
        """Format a single output tensor."""
        # Remove batch dimension if it's 1
        if output.ndim > 1 and output.shape[0] == 1:
            output = output.squeeze(0)
        
        # For classification-like outputs, return probabilities
        if output.ndim == 1 and len(output) > 1:
            # Apply softmax if not already applied
            if not np.allclose(output.sum(), 1.0, atol=1e-6):
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
    
    def infer(self, input_data: Union[str, np.ndarray, Dict[str, Any]], **kwargs) -> Any:
        """
        Run inference on the model.
        
        Args:
            input_data: Input data (string, numpy array, or dict)
            **kwargs: Additional inference parameters
            
        Returns:
            Model output
        """
        # Preprocess input
        processed_input = self._preprocess_input(input_data)
        
        # Run inference using infer_request (OpenVINO 2024 API)
        self.infer_request.infer(processed_input)
        
        # Get output data from infer_request
        output_data = {}
        for output_node in self.compiled_model.outputs:
            output_name = output_node.get_any_name()
            output_data[output_name] = self.infer_request.get_output_tensor(output_node.index).data
        
        # Postprocess output
        return self._postprocess_output(output_data)
    
    def get_input_info(self) -> Dict[str, Dict[str, Any]]:
        """Get input information."""
        return self.input_info.copy()
    
    def get_output_info(self) -> Dict[str, Dict[str, Any]]:
        """Get output information."""
        return self.output_info.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "device": self.device,
            "input_info": self.input_info,
            "output_info": self.output_info,
            "has_tokenizer": self.tokenizer is not None,
            **self.model_info
        } 