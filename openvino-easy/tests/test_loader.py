"""Unit tests for oe.loader."""

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
    key = _get_cache_key("test/model", "fp16", "2025.2.0")
    assert isinstance(key, str)
    assert len(key) == 64  # SHA-256 hex length


@patch("oe.loader.ov.Core")
@patch("oe.loader.normalize_path")
def test_load_local_ir_model(mock_normalize_path, mock_core):
    """Test loading a local IR model."""
    mock_path = MagicMock()
    mock_path.exists.return_value = True
    mock_path.suffix = ".xml"
    mock_path.__str__ = lambda self: "/path/to/model.xml"
    mock_normalize_path.return_value = mock_path

    mock_model = MagicMock()
    mock_core.return_value.read_model.return_value = mock_model

    result = load_model("/path/to/model.xml")

    mock_core.return_value.read_model.assert_called_once_with("/path/to/model.xml")
    assert result == mock_model


@patch("oe.loader.ov.Core")
@patch("oe.loader.normalize_path")
def test_load_local_onnx_model(mock_normalize_path, mock_core):
    """Test loading a local ONNX model."""
    mock_path = MagicMock()
    mock_path.exists.return_value = True
    mock_path.suffix = ".onnx"
    mock_path.__str__ = lambda self: "/path/to/model.onnx"
    mock_normalize_path.return_value = mock_path

    mock_model = MagicMock()
    mock_core.return_value.read_model.return_value = mock_model

    result = load_model("/path/to/model.onnx")

    mock_core.return_value.read_model.assert_called_once_with("/path/to/model.onnx")
    assert result == mock_model


@patch("oe.loader._check_dependencies_for_model_type")
@patch("oe.loader._detect_model_type")
@patch("oe.loader._download_with_retry")
@patch("oe.loader.ov.Core")
def test_load_hf_model_new(mock_core, mock_download, mock_detect_type, mock_check_deps):
    """Test loading a new Hugging Face model."""
    # Mock download returning a model path
    mock_download.return_value = "/tmp/downloaded/model"

    # Mock model type detection
    mock_detect_type.return_value = "transformers_optimum"

    # Mock dependency check to pass
    mock_check_deps.return_value = None

    # Mock model loading
    mock_model = MagicMock()
    mock_model.inputs = [MagicMock()]  # Mock having at least one input
    mock_model.outputs = [MagicMock()]  # Mock having at least one output
    mock_core.return_value.read_model.return_value = mock_model

    # Mock offline mode and no cached model
    with patch("oe.loader.ov.__version__", "2025.2.0"):
        with patch("oe.loader._convert_transformers_with_optimum") as mock_convert:
            with patch("oe.loader.ov.save_model"):
                mock_convert.return_value = "/tmp/converted/model.xml"
                result = load_model("test-model", offline=False)

    # Verify dependencies were checked
    mock_check_deps.assert_called_once_with("transformers_optimum", auto_install=False)
    # Verify download was called
    mock_download.assert_called_once()
    # Verify model was returned
    assert result == mock_model


@patch("oe.loader.ov.Core")
@patch("oe.loader.normalize_path")
def test_load_hf_model_cached(mock_normalize_path, mock_core):
    """Test loading a cached Hugging Face model."""
    # Mock model loading
    mock_model = MagicMock()
    mock_model.inputs = [MagicMock()]  # Mock having at least one input
    mock_model.outputs = [MagicMock()]  # Mock having at least one output
    mock_core.return_value.read_model.return_value = mock_model

    # Mock cache directory setup
    cache_dir = MagicMock()
    mock_normalize_path.return_value = cache_dir

    # Mock cache path exists
    cache_path = MagicMock()
    cache_path.exists.return_value = True  # Cache exists
    model_xml = MagicMock()
    model_xml.exists.return_value = True  # model.xml exists
    model_xml.__str__ = lambda self: "/cache/model.xml"
    cache_path.__truediv__.return_value = model_xml
    cache_dir.__truediv__.return_value = cache_path

    with patch("oe.loader.ov.__version__", "2025.2.0"):
        result = load_model("test-model")

    mock_core.return_value.read_model.assert_called_once()
    assert result == mock_model
