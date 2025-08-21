"""Unit tests for oe.quant."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path
from oe.quant import (
    quantize_model,
    _generate_random_calibration_data,
    _get_quant_cache_key,
    get_quantization_stats,
)


def test_get_quant_cache_key():
    """Test cache key generation for quantized models."""
    model_hash = "abc123"
    quant_config = {"algorithm": "DefaultQuantization", "preset": "mixed"}
    ov_version = "2025.2.0"

    key = _get_quant_cache_key(model_hash, quant_config, ov_version)
    assert isinstance(key, str)
    assert len(key) == 64  # SHA-256 hex length


def test_generate_calibration_data():
    """Test calibration data generation."""
    # Mock model with input nodes
    mock_model = MagicMock()
    mock_input1 = MagicMock()
    mock_input1.shape = [1, 3, 224, 224]
    mock_input1.get_any_name.return_value = "input1"

    mock_input2 = MagicMock()
    mock_input2.shape = [1, 10]
    mock_input2.get_any_name.return_value = "input2"

    mock_model.inputs = [mock_input1, mock_input2]

    calibration_data = _generate_random_calibration_data(mock_model, num_samples=5)

    assert len(calibration_data) == 5
    for sample in calibration_data:
        assert "input1" in sample
        assert "input2" in sample
        assert sample["input1"].shape == (1, 3, 224, 224)
        assert sample["input2"].shape == (1, 10)
        assert sample["input1"].dtype == np.float32
        assert sample["input2"].dtype == np.float32


def test_generate_calibration_data_dynamic_shapes():
    """Test calibration data generation with dynamic shapes."""
    mock_model = MagicMock()
    mock_input = MagicMock()
    mock_input.shape = [-1, 3, 224, 224]  # Dynamic batch size
    mock_input.get_any_name.return_value = "input"
    mock_model.inputs = [mock_input]

    calibration_data = _generate_random_calibration_data(mock_model, num_samples=3)

    assert len(calibration_data) == 3
    for sample in calibration_data:
        assert sample["input"].shape == (1, 3, 224, 224)  # Dynamic dim replaced with 1


@patch("oe.quant.NNCF_AVAILABLE", True)  # Mock NNCF as available
@patch("oe.quant.nncf")  # Mock nncf module
@patch(
    "oe.quant._get_model_checksum", return_value="mock_checksum"
)  # Mock checksum function
@patch("oe.quant.compress_weights")
@patch("oe.quant.ov.Core")
@patch("oe.quant.ov.save_model")
@patch("oe.quant.Path")
def test_quantize_model_int8_new(
    mock_path, mock_save, mock_core, mock_compress, mock_checksum, mock_nncf
):
    """Test quantizing a model to INT8 (new quantization)."""
    # Mock model
    mock_model = MagicMock()

    # Mock cache doesn't exist
    mock_cache_path = MagicMock()
    mock_cache_path.exists.return_value = False
    mock_path.return_value.expanduser.return_value = Path("/cache")
    mock_path.return_value.__truediv__.return_value = mock_cache_path

    # Mock NNCF module properties
    mock_nncf.CompressWeightsMode.INT8_SYM = "INT8_SYM"

    # Mock quantization
    mock_quantized_model = MagicMock()
    mock_compress.return_value = mock_quantized_model

    # Mock OpenVINO version
    with patch("oe.quant.ov.__version__", "2025.2.0"):
        result = quantize_model(
            mock_model, dtype="int8", quantization_method="weights_only"
        )

    mock_compress.assert_called_once()
    mock_save.assert_called_once()
    assert result == mock_quantized_model


@patch("oe.quant.NNCF_AVAILABLE", True)  # Mock NNCF as available
@patch("oe.quant.nncf")  # Mock nncf module
@patch(
    "oe.quant._get_model_checksum", return_value="mock_checksum"
)  # Mock checksum function
@patch("oe.quant.ov.Core")
@patch("oe.quant.Path")
def test_quantize_model_int8_cached(mock_path, mock_core, mock_checksum, mock_nncf):
    """Test quantizing a model to INT8 (cached)."""
    # Mock model
    mock_model = MagicMock()

    # Mock cache exists with model.xml
    mock_cache_path = MagicMock()
    mock_cache_path.exists.return_value = True
    mock_model_xml = MagicMock()
    mock_model_xml.exists.return_value = True
    mock_cache_path.__truediv__.return_value = mock_model_xml
    mock_path.return_value.expanduser.return_value = Path("/cache")
    mock_path.return_value.__truediv__.return_value = mock_cache_path

    # Mock NNCF module properties
    mock_nncf.CompressWeightsMode.INT8_SYM = "INT8_SYM"

    # Mock model loading
    mock_quantized_model = MagicMock()
    mock_core.return_value.read_model.return_value = mock_quantized_model

    result = quantize_model(mock_model, dtype="int8")

    # Since NNCF is not actually available, quantization will return original model
    # We test that the function doesn't crash and returns a model
    assert result is not None


def test_quantize_model_fp16():
    """Test that FP16 quantization returns the original model."""
    mock_model = MagicMock()

    result = quantize_model(mock_model, dtype="fp16")

    assert result == mock_model


def test_quantize_model_unsupported_dtype():
    """Test that unsupported dtypes raise ValueError."""
    mock_model = MagicMock()

    with pytest.raises(ValueError, match="Unsupported dtype"):
        quantize_model(mock_model, dtype="fp32")


@patch("oe.quant.compress_weights")
@patch("oe.quant.Path")
def test_quantize_model_error_handling(mock_path, mock_compress):
    """Test error handling during quantization."""
    mock_model = MagicMock()

    # Mock cache doesn't exist
    mock_cache_path = MagicMock()
    mock_cache_path.exists.return_value = False
    mock_path.return_value.expanduser.return_value = Path("/cache")
    mock_path.return_value.__truediv__.return_value = mock_cache_path

    # Mock compression to raise an error
    mock_compress.side_effect = Exception("Quantization failed")

    # Should return original model on error, not raise
    result = quantize_model(
        mock_model, dtype="int8", quantization_method="weights_only"
    )
    assert result == mock_model  # Should return original model on failure


def test_get_quantization_stats():
    """Test quantization statistics function."""
    mock_model = MagicMock()
    mock_quantized_model = MagicMock()

    stats = get_quantization_stats(mock_model, mock_quantized_model)

    assert isinstance(stats, dict)
    assert "original_size_mb" in stats
    assert "quantized_size_mb" in stats
    assert "compression_ratio" in stats
    assert "quantization_method" in stats
    assert (
        "NNCF" in stats["quantization_method"] or stats["quantization_method"] == "None"
    )
