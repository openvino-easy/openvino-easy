"""Unit tests for oe.loader."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from oe.loader import load_model, _get_cache_key

def test_load_model_signature():
    """Test that load_model has the correct signature."""
    import inspect
    sig = inspect.signature(load_model)
    assert "model_id_or_path" in sig.parameters
    assert "dtype" in sig.parameters
    assert "cache_dir" in sig.parameters

def test_get_cache_key():
    """Test cache key generation."""
    key = _get_cache_key("test/model", "fp16", "2024.0.0")
    assert isinstance(key, str)
    assert len(key) == 64  # SHA-256 hex length

@patch('oe.loader.ov.Core')
@patch('oe.loader.Path')
def test_load_local_ir_model(mock_path, mock_core):
    """Test loading a local IR model."""
    mock_path.return_value.exists.return_value = True
    mock_path.return_value.suffix = ".xml"
    mock_path.return_value.__str__ = lambda x: "/path/to/model.xml"
    
    mock_model = MagicMock()
    mock_core.return_value.read_model.return_value = mock_model
    
    result = load_model("/path/to/model.xml")
    
    mock_core.return_value.read_model.assert_called_once_with("/path/to/model.xml")
    assert result == mock_model

@patch('oe.loader.ov.Core')
@patch('oe.loader.Path')
def test_load_local_onnx_model(mock_path, mock_core):
    """Test loading a local ONNX model."""
    mock_path.return_value.exists.return_value = True
    mock_path.return_value.suffix = ".onnx"
    mock_path.return_value.__str__ = lambda x: "/path/to/model.onnx"
    
    mock_model = MagicMock()
    mock_core.return_value.read_model.return_value = mock_model
    
    result = load_model("/path/to/model.onnx")
    
    mock_core.return_value.read_model.assert_called_once_with("/path/to/model.onnx")
    assert result == mock_model

@patch('oe.loader.snapshot_download')
@patch('oe.loader.ov.convert_model')
@patch('oe.loader.ov.save_model')
@patch('oe.loader.ov.Core')
@patch('oe.loader.Path')
def test_load_hf_model_new(mock_path, mock_core, mock_save, mock_convert, mock_download):
    """Test loading a new Hugging Face model."""
    # Mock path doesn't exist (not local file)
    mock_path.return_value.exists.return_value = False
    mock_path.return_value.expanduser.return_value = Path("/cache")
    
    # Mock cache doesn't exist
    mock_cache_path = MagicMock()
    mock_cache_path.exists.return_value = False
    mock_path.return_value.__truediv__.return_value = mock_cache_path
    
    # Mock download
    mock_download.return_value = "/local/model/path"
    
    # Mock conversion
    mock_model = MagicMock()
    mock_convert.return_value = mock_model
    
    # Mock OpenVINO version
    with patch('oe.loader.ov.__version__', '2024.0.0'):
        result = load_model("test/model")
    
    mock_download.assert_called_once()
    mock_convert.assert_called_once_with("/local/model/path", from_hf=True, dtype="fp16")
    assert result == mock_model

@patch('oe.loader.ov.Core')
@patch('oe.loader.Path')
def test_load_hf_model_cached(mock_path, mock_core):
    """Test loading a cached Hugging Face model."""
    # Mock path doesn't exist (not local file)
    mock_path.return_value.exists.return_value = False
    mock_path.return_value.expanduser.return_value = Path("/cache")
    
    # Mock cache exists with model.xml
    mock_cache_path = MagicMock()
    mock_cache_path.exists.return_value = True
    mock_model_xml = MagicMock()
    mock_model_xml.exists.return_value = True
    mock_cache_path.__truediv__.return_value = mock_model_xml
    mock_path.return_value.__truediv__.return_value = mock_cache_path
    
    # Mock model loading
    mock_model = MagicMock()
    mock_core.return_value.read_model.return_value = mock_model
    
    result = load_model("test/model")
    
    mock_core.return_value.read_model.assert_called_once()
    assert result == mock_model 