"""Unit tests for oe.runtime."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from oe.runtime import RuntimeWrapper

class TestRuntimeWrapper:
    """Test RuntimeWrapper functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock compiled model
        self.mock_compiled_model = MagicMock()
        
        # Mock input nodes
        self.mock_input1 = MagicMock()
        self.mock_input1.get_any_name.return_value = "input1"
        self.mock_input1.shape = [1, 3, 224, 224]
        self.mock_input1.get_element_type.return_value = "f32"
        
        self.mock_input2 = MagicMock()
        self.mock_input2.get_any_name.return_value = "input2"
        self.mock_input2.shape = [1, 10]
        self.mock_input2.get_element_type.return_value = "f32"
        
        # Mock output nodes
        self.mock_output1 = MagicMock()
        self.mock_output1.get_any_name.return_value = "output1"
        self.mock_output1.shape = [1, 1000]
        self.mock_output1.get_element_type.return_value = "f32"
        
        self.mock_compiled_model.inputs = [self.mock_input1, self.mock_input2]
        self.mock_compiled_model.outputs = [self.mock_output1]
        
        # Create wrapper
        self.wrapper = RuntimeWrapper(self.mock_compiled_model, "NPU")
    
    def test_init(self):
        """Test wrapper initialization."""
        assert self.wrapper.device == "NPU"
        assert self.wrapper.compiled_model == self.mock_compiled_model
        assert "input1" in self.wrapper.input_info
        assert "input2" in self.wrapper.input_info
        assert "output1" in self.wrapper.output_info
    
    def test_extract_input_info(self):
        """Test input information extraction."""
        input_info = self.wrapper.get_input_info()
        
        assert "input1" in input_info
        assert "input2" in input_info
        assert input_info["input1"]["shape"] == [1, 3, 224, 224]
        assert input_info["input1"]["dtype"] == "f32"
        assert input_info["input2"]["shape"] == [1, 10]
        assert input_info["input2"]["dtype"] == "f32"
    
    def test_extract_output_info(self):
        """Test output information extraction."""
        output_info = self.wrapper.get_output_info()
        
        assert "output1" in output_info
        assert output_info["output1"]["shape"] == [1, 1000]
        assert output_info["output1"]["dtype"] == "f32"
    
    def test_shapes_compatible(self):
        """Test shape compatibility checking."""
        # Test compatible shapes
        assert self.wrapper._shapes_compatible((1, 3, 224, 224), [1, 3, 224, 224])
        assert self.wrapper._shapes_compatible((2, 3, 224, 224), [-1, 3, 224, 224])  # Dynamic batch
        
        # Test incompatible shapes
        assert not self.wrapper._shapes_compatible((1, 3, 224, 224), [1, 3, 224, 225])
        assert not self.wrapper._shapes_compatible((1, 3, 224), [1, 3, 224, 224])  # Wrong dims
    
    def test_validate_input_dict(self):
        """Test input dictionary validation."""
        # Valid input
        input_dict = {
            "input1": np.random.randn(1, 3, 224, 224).astype(np.float32),
            "input2": np.random.randn(1, 10).astype(np.float32)
        }
        
        processed = self.wrapper._validate_input_dict(input_dict)
        assert "input1" in processed
        assert "input2" in processed
        assert processed["input1"].dtype == np.float32
        assert processed["input2"].dtype == np.float32
    
    def test_validate_input_dict_unknown_input(self):
        """Test validation with unknown input name."""
        input_dict = {"unknown_input": np.random.randn(1, 3, 224, 224)}
        
        with pytest.raises(ValueError, match="Unknown input name"):
            self.wrapper._validate_input_dict(input_dict)
    
    def test_validate_input_dict_shape_mismatch(self):
        """Test validation with shape mismatch."""
        input_dict = {"input1": np.random.randn(1, 3, 224, 225)}  # Wrong shape
        
        with pytest.raises(ValueError, match="Input shape mismatch"):
            self.wrapper._validate_input_dict(input_dict)
    
    def test_preprocess_array_input_single_input(self):
        """Test array preprocessing for single input model."""
        # Create wrapper with single input
        single_input_model = MagicMock()
        single_input_model.inputs = [self.mock_input1]
        single_input_model.outputs = [self.mock_output1]
        
        wrapper = RuntimeWrapper(single_input_model, "CPU")
        
        # Test array input
        input_array = np.random.randn(3, 224, 224)
        processed = wrapper._preprocess_array_input(input_array)
        
        assert "input1" in processed
        assert processed["input1"].shape == (1, 3, 224, 224)
        assert processed["input1"].dtype == np.float32
    
    def test_preprocess_array_input_multiple_inputs(self):
        """Test array preprocessing with multiple inputs (should fail)."""
        input_array = np.random.randn(3, 224, 224)
        
        with pytest.raises(ValueError, match="Model has 2 inputs"):
            self.wrapper._preprocess_array_input(input_array)
    
    def test_preprocess_text_input(self):
        """Test text preprocessing."""
        # Create wrapper with single input for text
        single_input_model = MagicMock()
        single_input_model.inputs = [self.mock_input1]
        single_input_model.outputs = [self.mock_output1]
        
        wrapper = RuntimeWrapper(single_input_model, "CPU")
        
        text = "hello world test"
        processed = wrapper._preprocess_text_input(text)
        
        assert "input1" in processed
        assert processed["input1"].shape == (1, 3, 224, 224)
        assert processed["input1"].dtype == np.float32
    
    def test_preprocess_input_string(self):
        """Test string input preprocessing."""
        # Mock the text preprocessing method
        with patch.object(self.wrapper, '_preprocess_text_input') as mock_text:
            mock_text.return_value = {"input1": np.random.randn(1, 3, 224, 224)}
            
            result = self.wrapper._preprocess_input("test prompt")
            mock_text.assert_called_once_with("test prompt")
    
    def test_preprocess_input_array(self):
        """Test array input preprocessing."""
        with patch.object(self.wrapper, '_preprocess_array_input') as mock_array:
            mock_array.return_value = {"input1": np.random.randn(1, 3, 224, 224)}
            
            input_array = np.random.randn(3, 224, 224)
            result = self.wrapper._preprocess_input(input_array)
            mock_array.assert_called_once_with(input_array)
    
    def test_preprocess_input_dict(self):
        """Test dict input preprocessing."""
        with patch.object(self.wrapper, '_validate_input_dict') as mock_dict:
            mock_dict.return_value = {"input1": np.random.randn(1, 3, 224, 224)}
            
            input_dict = {"input1": np.random.randn(1, 3, 224, 224)}
            result = self.wrapper._preprocess_input(input_dict)
            mock_dict.assert_called_once_with(input_dict)
    
    def test_preprocess_input_unsupported(self):
        """Test unsupported input type."""
        with pytest.raises(ValueError, match="Unsupported input type"):
            self.wrapper._preprocess_input(123)  # int is not supported
    
    def test_format_single_output_scalar(self):
        """Test single output formatting for scalar."""
        output = np.array([42.0])
        result = self.wrapper._format_single_output(output)
        assert result == 42.0
    
    def test_format_single_output_classification(self):
        """Test single output formatting for classification."""
        output = np.array([1.0, 2.0, 3.0])
        result = self.wrapper._format_single_output(output)
        
        # Should apply softmax and return list
        assert isinstance(result, list)
        assert len(result) == 3
        assert abs(sum(result) - 1.0) < 1e-6  # Probabilities sum to 1
    
    def test_format_single_output_already_probabilities(self):
        """Test single output formatting for already normalized probabilities."""
        output = np.array([0.2, 0.3, 0.5])  # Already sums to 1
        result = self.wrapper._format_single_output(output)
        
        assert isinstance(result, list)
        assert result == [0.2, 0.3, 0.5]
    
    def test_format_single_output_batch_dimension(self):
        """Test single output formatting with batch dimension."""
        output = np.array([[1.0, 2.0, 3.0]])  # Batch dimension of 1
        result = self.wrapper._format_single_output(output)
        
        # Should remove batch dimension
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]
    
    def test_postprocess_output_single(self):
        """Test postprocessing for single output."""
        output_data = {"output1": np.array([1.0, 2.0, 3.0])}
        
        with patch.object(self.wrapper, '_format_single_output') as mock_format:
            mock_format.return_value = [0.1, 0.2, 0.7]
            
            result = self.wrapper._postprocess_output(output_data)
            mock_format.assert_called_once_with(output_data["output1"])
            assert result == [0.1, 0.2, 0.7]
    
    def test_postprocess_output_multiple(self):
        """Test postprocessing for multiple outputs."""
        output_data = {
            "output1": np.array([1.0, 2.0]),
            "output2": np.array([3.0])
        }
        
        with patch.object(self.wrapper, '_format_single_output') as mock_format:
            mock_format.side_effect = [[0.1, 0.9], 3.0]
            
            result = self.wrapper._postprocess_output(output_data)
            
            assert mock_format.call_count == 2
            assert result == {"output1": [0.1, 0.9], "output2": 3.0}
    
    def test_softmax(self):
        """Test softmax function."""
        x = np.array([1.0, 2.0, 3.0])
        result = self.wrapper._softmax(x)
        
        # Check that probabilities sum to 1
        assert abs(result.sum() - 1.0) < 1e-6
        
        # Check that larger values get higher probabilities
        assert result[2] > result[1] > result[0]
    
    def test_infer(self):
        """Test inference method."""
        # Mock preprocessing and postprocessing
        mock_input = {"input1": np.random.randn(1, 3, 224, 224)}
        mock_output = {"output1": np.array([0.1, 0.2, 0.7])}
        
        with patch.object(self.wrapper, '_preprocess_input', return_value=mock_input):
            with patch.object(self.wrapper, '_postprocess_output', return_value=mock_output):
                self.mock_compiled_model.return_value = mock_output
                
                result = self.wrapper.infer("test input")
                
                self.mock_compiled_model.assert_called_once_with(mock_input)
                assert result == mock_output
    
    def test_get_model_info(self):
        """Test model info retrieval."""
        model_info = self.wrapper.get_model_info()
        
        assert model_info["device"] == "NPU"
        assert "input_info" in model_info
        assert "output_info" in model_info
        assert "input1" in model_info["input_info"]
        assert "output1" in model_info["output_info"]
    
    def test_get_model_info_with_custom_info(self):
        """Test model info with custom metadata."""
        custom_info = {"model_type": "classification", "version": "1.0"}
        wrapper = RuntimeWrapper(self.mock_compiled_model, "GPU", custom_info)
        
        model_info = wrapper.get_model_info()
        
        assert model_info["device"] == "GPU"
        assert model_info["model_type"] == "classification"
        assert model_info["version"] == "1.0" 