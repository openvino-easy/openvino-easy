"""Tests for model management API."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import oe


class TestModelSearch:
    """Test model search functionality."""

    @patch("huggingface_hub.HfApi")
    def test_search_basic(self, mock_hf_api):
        """Test basic model search."""
        # Mock HF API response
        mock_model = Mock()
        mock_model.id = "test/model"
        mock_model.downloads = 1000
        mock_model.likes = 50
        mock_model.pipeline_tag = "text-generation"
        mock_model.tags = ["pytorch", "text"]
        mock_model.created_at = "2024-01-01"
        mock_model.last_modified = "2024-01-02"
        mock_model.private = False
        mock_model.gated = False

        mock_api_instance = Mock()
        mock_api_instance.list_models.return_value = [mock_model]
        mock_hf_api.return_value = mock_api_instance

        # Test search
        results = oe.models.search("test query", limit=5)

        assert len(results) == 1
        assert results[0]["id"] == "test/model"
        assert results[0]["downloads"] == 1000
        assert results[0]["likes"] == 50
        assert results[0]["pipeline_tag"] == "text-generation"

        # Verify API was called correctly
        mock_api_instance.list_models.assert_called_once()
        call_args = mock_api_instance.list_models.call_args
        assert call_args[1]["search"] == "test query"
        assert call_args[1]["limit"] == 5

    @patch("huggingface_hub.HfApi")
    def test_search_with_type_filter(self, mock_hf_api):
        """Test search with model type filtering."""
        mock_api_instance = Mock()
        mock_api_instance.list_models.return_value = []
        mock_hf_api.return_value = mock_api_instance

        # Test with text type
        oe.models.search("query", model_type="text")

        call_args = mock_api_instance.list_models.call_args
        assert "pipeline_tag" in call_args[1]
        assert "text-generation" in call_args[1]["pipeline_tag"]

    def test_search_missing_dependency(self):
        """Test search without huggingface_hub installed."""
        with patch("huggingface_hub.HfApi", side_effect=ImportError):
            with pytest.raises(RuntimeError, match="Hugging Face Hub client required"):
                oe.models.search("query")


class TestModelInfo:
    """Test model info functionality."""

    def test_info_local_only(self):
        """Test getting info for local-only model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake local model
            models_dir = Path(temp_dir) / "models"
            models_dir.mkdir()
            model_dir = models_dir / "test--model--fp16--hash123"
            model_dir.mkdir()
            (model_dir / "model.xml").touch()
            (model_dir / "model.bin").touch()

            # Mock the models.list to return our fake model
            with patch.object(oe.models, "list") as mock_list:
                mock_list.return_value = [
                    {
                        "name": "test--model--fp16--hash123",
                        "path": str(model_dir),
                        "size_mb": 100.5,
                        "files": ["model.xml", "model.bin"],
                    }
                ]

                info = oe.models.info("test--model", temp_dir)

                assert info["local"]
                assert info["local_info"]["size_mb"] == 100.5
                assert "test--model" in info["local_info"]["name"]

    @patch("huggingface_hub.model_info")
    def test_info_remote_only(self, mock_hf_info):
        """Test getting info for remote-only model."""
        # Mock HF model info
        mock_remote = Mock()
        mock_remote.id = "test/model"
        mock_remote.downloads = 5000
        mock_remote.likes = 100
        mock_remote.pipeline_tag = "text-generation"
        mock_remote.tags = ["llm"]
        mock_remote.library_name = "transformers"
        mock_remote.created_at = "2024-01-01"
        mock_remote.last_modified = "2024-01-02"
        mock_remote.private = False
        mock_remote.gated = False

        mock_hf_info.return_value = mock_remote

        with patch.object(oe.models, "list", return_value=[]):
            info = oe.models.info("test/model")

            assert not info["local"]
            assert info["remote"]
            assert info["remote_info"]["downloads"] == 5000
            assert info["requirements"]["min_memory_mb"] == 2000  # LLM default


class TestModelInstall:
    """Test model installation functionality."""

    @patch("oe.load_model")
    @patch("oe._get_models_dir")
    def test_install_success(self, mock_get_dir, mock_load_model):
        """Test successful model installation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_dir.return_value = Path(temp_dir)
            mock_load_model.return_value = Mock()  # Mock OV model

            # Mock the models.list to be empty initially, then show the installed model
            with patch.object(oe.models, "list") as mock_list:
                mock_list.side_effect = [
                    [],  # First call: no models installed initially
                    [    # Second call: model appears after installation
                        {
                            "name": "test--model--fp16--hash123",
                            "size_mb": 150.0,
                            "files": ["model.xml", "model.bin"],
                        }
                    ]
                ]

                result = oe.models.install("test/model", dtype="fp16")

                assert result["installed"]
                assert result["model_id"] == "test/model"
                assert result["dtype"] == "fp16"
                assert "Successfully installed" in result["message"]

    def test_install_already_exists(self):
        """Test installing model that already exists."""
        with patch.object(oe.models, "list") as mock_list:
            mock_list.return_value = [
                {
                    "name": "test--model--fp16--hash123",
                    "size_mb": 150.0,
                    "files": ["model.xml", "model.bin"],
                }
            ]

            result = oe.models.install("test/model", dtype="fp16", force=False)

            assert not result["installed"]
            assert result["already_exists"]
            assert "already installed" in result["message"]

    @patch("oe.load_model")
    def test_install_failure(self, mock_load_model):
        """Test failed model installation."""
        mock_load_model.side_effect = Exception("Network error")

        result = oe.models.install("test/model")

        assert not result["installed"]
        assert "error" in result
        assert "Network error" in result["error"]


class TestModelValidation:
    """Test model validation functionality."""

    def test_validate_valid_model(self):
        """Test validation of valid model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create valid model structure
            model_dir = Path(temp_dir) / "test_model"
            model_dir.mkdir()
            (model_dir / "model.xml").touch()
            (model_dir / "model.bin").touch()

            with patch.object(oe.models, "list") as mock_list:
                mock_list.return_value = [
                    {"name": "test_model", "path": str(model_dir), "size_mb": 100.0}
                ]

                # Mock OpenVINO model loading
                with patch("openvino.Core") as mock_core:
                    mock_model = Mock()
                    mock_model.inputs = [Mock()]  # Has inputs
                    mock_model.outputs = [Mock()]  # Has outputs

                    mock_core_instance = Mock()
                    mock_core_instance.read_model.return_value = mock_model
                    mock_core.return_value = mock_core_instance

                    results = oe.models.validate()

                    assert results["validated"] == 1
                    assert results["passed"] == 1
                    assert results["failed"] == 0
                    assert results["models"][0]["valid"]

    def test_validate_invalid_model(self):
        """Test validation of invalid model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid model structure (missing files)
            model_dir = Path(temp_dir) / "test_model"
            model_dir.mkdir()
            # No model.xml file

            with patch.object(oe.models, "list") as mock_list:
                mock_list.return_value = [
                    {"name": "test_model", "path": str(model_dir), "size_mb": 100.0}
                ]

                results = oe.models.validate()

                assert results["validated"] == 1
                assert results["passed"] == 0
                assert results["failed"] == 1
                assert not results["models"][0]["valid"]
                assert any(
                    "No .xml files found" in error
                    for error in results["models"][0]["errors"]
                )


class TestModelBenchmarkAll:
    """Test benchmarking all models functionality."""

    @patch("oe.load")
    @patch("oe.benchmark")
    @patch("oe.unload")
    def test_benchmark_all_success(self, mock_unload, mock_benchmark, mock_load):
        """Test successful benchmarking of all models."""
        # Mock benchmark results
        mock_benchmark.return_value = {"fps": 25.5, "mean_ms": 39.2, "device": "CPU"}

        with patch.object(oe.models, "list") as mock_list:
            mock_list.return_value = [
                {"name": "test--model1--fp16--hash1", "size_mb": 100.0},
                {"name": "test--model2--int8--hash2", "size_mb": 50.0},
            ]

            results = oe.models.benchmark_all(warmup=2, runs=5)

            assert results["total_models"] == 2
            assert results["benchmarked"] == 2
            assert results["failed"] == 0
            assert len(results["results"]) == 2

            # Check summary stats
            assert results["summary"]["average_fps"] == 25.5
            assert results["summary"]["fastest_model"]["fps"] == 25.5
            assert results["summary"]["slowest_model"]["fps"] == 25.5

            # Verify load/benchmark/unload called for each model
            assert mock_load.call_count == 2
            assert mock_benchmark.call_count == 2
            assert mock_unload.call_count == 2

    @patch("oe.load")
    def test_benchmark_all_with_failures(self, mock_load):
        """Test benchmarking with some model failures."""
        # First model succeeds, second fails
        mock_load.side_effect = [None, Exception("Load failed")]

        with patch.object(oe.models, "list") as mock_list:
            mock_list.return_value = [
                {"name": "good--model--fp16--hash1", "size_mb": 100.0},
                {"name": "bad--model--fp16--hash2", "size_mb": 50.0},
            ]

            with patch("oe.benchmark", return_value={"fps": 30.0}):
                with patch("oe.unload"):
                    results = oe.models.benchmark_all()

                    assert results["total_models"] == 2
                    assert results["benchmarked"] == 1  # Only one succeeded
                    assert results["failed"] == 1

                    # Check that failure is recorded
                    failed_result = next(r for r in results["results"] if "error" in r)
                    assert "Load failed" in failed_result["error"]


class TestSafetyFeatures:
    """Test safety features in model management."""

    def test_path_validation_in_models_dir(self):
        """Test that path validation applies to models.* operations."""
        with pytest.raises(ValueError, match="Unsafe path pattern"):
            oe.models.list("../../../etc")

    def test_path_validation_in_install(self):
        """Test path validation in model installation."""
        with pytest.raises(ValueError, match="Unsafe path pattern"):
            oe.models.install("test/model", cache_dir="${HOME}/dangerous")


@pytest.mark.integration
class TestModelManagementIntegration:
    """Integration tests for model management."""

    def test_full_model_lifecycle(self):
        """Test complete model lifecycle: search -> install -> validate -> benchmark."""
        # This would be a real integration test that:
        # 1. Searches for a small test model
        # 2. Installs it
        # 3. Validates it
        # 4. Benchmarks it
        # 5. Removes it

        # Skip if no network or in CI
        pytest.skip("Integration test - requires network and time")

    def test_model_compatibility_validation(self):
        """Test automated model compatibility validation."""
        # This would test:
        # 1. Different model architectures
        # 2. Different precision formats
        # 3. Different device targets
        # 4. Edge cases and error conditions

        pytest.skip("Integration test - requires multiple models")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
