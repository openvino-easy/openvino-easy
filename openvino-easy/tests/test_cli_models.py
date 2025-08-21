"""Tests for CLI model management commands."""

import pytest
import json
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path

from oe.cli import (
    model_search,
    model_info,
    model_install,
    model_validate,
    model_benchmark,
)


class TestCLIModelSearch:
    """Test CLI model search command."""

    @patch("oe.models.search")
    def test_search_basic(self, mock_search, capsys):
        """Test basic model search command."""
        # Mock search results
        mock_search.return_value = [
            {
                "id": "test/model1",
                "downloads": 1000,
                "likes": 50,
                "pipeline_tag": "text-generation",
                "private": False,
                "gated": False,
            },
            {
                "id": "test/model2",
                "downloads": 500,
                "likes": 25,
                "pipeline_tag": "image-classification",
                "private": True,
                "gated": False,
            },
        ]

        # Mock args
        args = Mock()
        args.query = "test query"
        args.limit = 10
        args.type = None
        args.json = False

        model_search(args)

        # Verify search was called correctly
        mock_search.assert_called_once_with("test query", limit=10, model_type=None)

        # Check output
        captured = capsys.readouterr()
        assert "Search Results for 'test query'" in captured.out
        assert "test/model1" in captured.out
        assert "Downloads: 1,000" in captured.out
        assert "[PRIVATE MODEL]" in captured.out

    @patch("oe.models.search")
    def test_search_json_output(self, mock_search, capsys):
        """Test model search with JSON output."""
        mock_results = [{"id": "test/model", "downloads": 1000}]
        mock_search.return_value = mock_results

        args = Mock()
        args.query = "test"
        args.limit = 5
        args.type = "text"
        args.json = True

        model_search(args)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output == mock_results

    @patch("oe.models.search")
    def test_search_no_results(self, mock_search, capsys):
        """Test search with no results."""
        mock_search.return_value = []

        args = Mock()
        args.query = "nonexistent"
        args.limit = 10
        args.type = None
        args.json = False

        model_search(args)

        captured = capsys.readouterr()
        assert "No models found for query" in captured.out

    @patch("oe.models.search")
    def test_search_error_handling(self, mock_search):
        """Test search error handling."""
        mock_search.side_effect = Exception("Network error")

        args = Mock()
        args.query = "test"
        args.limit = 10
        args.type = None
        args.json = False

        with pytest.raises(SystemExit):
            model_search(args)


class TestCLIModelInfo:
    """Test CLI model info command."""

    @patch("oe.models.info")
    def test_info_local_and_remote(self, mock_info, capsys):
        """Test model info with both local and remote data."""
        mock_info.return_value = {
            "local": True,
            "remote": True,
            "local_info": {
                "size_mb": 150.5,
                "path": "/path/to/model",
                "files": ["model.xml", "model.bin"],
            },
            "remote_info": {
                "downloads": 10000,
                "likes": 200,
                "pipeline_tag": "text-generation",
                "library_name": "transformers",
                "last_modified": "2024-01-15T10:30:00Z",
                "private": False,
                "gated": True,
            },
            "requirements": {
                "min_memory_mb": 2000,
                "recommended_devices": ["NPU", "GPU", "CPU"],
                "supported_precisions": ["fp16", "int8"],
            },
        }

        args = Mock()
        args.model = "test/model"
        args.cache_dir = None
        args.json = False

        model_info(args)

        captured = capsys.readouterr()
        assert "Model Information: test/model" in captured.out
        assert "Installed locally: 150.5 MB" in captured.out
        assert "Downloads: 10,000" in captured.out
        assert "This is a gated model" in captured.out
        assert "Memory: 2000 MB" in captured.out

    @patch("oe.models.info")
    def test_info_local_only(self, mock_info, capsys):
        """Test model info with only local data."""
        mock_info.return_value = {
            "local": True,
            "remote": False,
            "local_info": {"size_mb": 100.0, "path": "/path", "files": ["model.xml"]},
            "remote_error": "Model not found on Hub",
            "requirements": {
                "min_memory_mb": "unknown",
                "recommended_devices": ["CPU"],
                "supported_precisions": ["fp16"],
            },
        }

        args = Mock()
        args.model = "local/model"
        args.cache_dir = None
        args.json = False

        model_info(args)

        captured = capsys.readouterr()
        assert "Installed locally: 100.0 MB" in captured.out
        assert "No remote information available" in captured.out
        assert "Error: Model not found on Hub" in captured.out

    @patch("oe.models.info")
    def test_info_remote_only_with_install_suggestion(self, mock_info, capsys):
        """Test model info with only remote data and install suggestion."""
        mock_info.return_value = {
            "local": False,
            "remote": True,
            "remote_info": {
                "downloads": 5000,
                "likes": 100,
                "pipeline_tag": "text-generation",
                "library_name": "transformers",
                "last_modified": "2024-01-01T00:00:00Z",
                "private": False,
                "gated": False,
            },
            "requirements": {
                "min_memory_mb": 1500,
                "recommended_devices": ["GPU", "CPU"],
                "supported_precisions": ["fp16", "int8"],
            },
        }

        args = Mock()
        args.model = "remote/model"
        args.cache_dir = None
        args.json = False

        model_info(args)

        captured = capsys.readouterr()
        assert "Not installed locally" in captured.out
        assert "Downloads: 5,000" in captured.out
        assert "Install with: oe models install remote/model" in captured.out


class TestCLIModelInstall:
    """Test CLI model install command."""

    @patch("oe.models.install")
    def test_install_success(self, mock_install, capsys):
        """Test successful model installation."""
        mock_install.return_value = {
            "installed": True,
            "model_id": "test/model",
            "dtype": "fp16",
            "size_mb": 200.5,
            "files": 15,
            "message": "Successfully installed test/model with fp16 precision",
        }

        args = Mock()
        args.model = "test/model"
        args.dtype = "fp16"
        args.cache_dir = None
        args.force = False
        args.json = False

        model_install(args)

        mock_install.assert_called_once_with(
            "test/model", dtype="fp16", cache_dir=None, force=False
        )

        captured = capsys.readouterr()
        assert "Successfully installed test/model" in captured.out
        assert "Size: 200.5 MB" in captured.out
        assert "Files: 15" in captured.out

    @patch("oe.models.install")
    def test_install_already_exists(self, mock_install, capsys):
        """Test installing model that already exists."""
        mock_install.return_value = {
            "installed": False,
            "already_exists": True,
            "model_name": "test--model--fp16--hash",
            "size_mb": 200.5,
            "message": "Model already installed. Use force=True to reinstall.",
        }

        args = Mock()
        args.model = "test/model"
        args.dtype = "fp16"
        args.cache_dir = None
        args.force = False
        args.json = False

        model_install(args)

        captured = capsys.readouterr()
        assert "Model already installed" in captured.out
        assert "Use --force to reinstall" in captured.out

    @patch("oe.models.install")
    def test_install_failure(self, mock_install):
        """Test failed model installation."""
        mock_install.return_value = {
            "installed": False,
            "error": "Network timeout",
            "message": "Failed to install: Network timeout",
        }

        args = Mock()
        args.model = "test/model"
        args.dtype = "fp16"
        args.cache_dir = None
        args.force = False
        args.json = False

        with pytest.raises(SystemExit):
            model_install(args)


class TestCLIModelValidate:
    """Test CLI model validate command."""

    @patch("oe.models.validate")
    def test_validate_all_models_success(self, mock_validate, capsys):
        """Test validation of all models with success."""
        mock_validate.return_value = {
            "validated": 3,
            "passed": 2,
            "failed": 1,
            "models": [
                {
                    "name": "model1",
                    "valid": True,
                    "errors": [],
                    "warnings": ["Minor warning"],
                },
                {"name": "model2", "valid": True, "errors": [], "warnings": []},
                {
                    "name": "model3",
                    "valid": False,
                    "errors": ["Missing XML file", "Corrupted data"],
                    "warnings": [],
                },
            ],
        }

        args = Mock()
        args.model = None
        args.cache_dir = None
        args.json = False

        model_validate(args)

        captured = capsys.readouterr()
        assert "Validation Results (3 models checked)" in captured.out
        assert "model1: Valid" in captured.out
        assert "model2: Valid" in captured.out
        assert "model3: Invalid" in captured.out
        assert "Error: Missing XML file" in captured.out
        assert "Error: Corrupted data" in captured.out
        assert "Warning: Minor warning" in captured.out
        assert "Summary: 2 passed, 1 failed" in captured.out
        assert "Some models failed validation" in captured.out

    @patch("oe.models.validate")
    def test_validate_specific_model(self, mock_validate):
        """Test validation of specific model."""
        mock_validate.return_value = {
            "validated": 1,
            "passed": 1,
            "failed": 0,
            "models": [
                {"name": "specific_model", "valid": True, "errors": [], "warnings": []}
            ],
        }

        args = Mock()
        args.model = "specific_model"
        args.cache_dir = "/custom/cache"
        args.json = False

        model_validate(args)

        mock_validate.assert_called_once_with("specific_model", "/custom/cache")


class TestCLIModelBenchmark:
    """Test CLI model benchmark command."""

    @patch("oe.models.benchmark_all")
    def test_benchmark_success(self, mock_benchmark, capsys):
        """Test successful benchmarking of all models."""
        mock_benchmark.return_value = {
            "total_models": 2,
            "benchmarked": 2,
            "failed": 0,
            "results": [
                {
                    "model_id": "test/model1",
                    "model_name": "test--model1--fp16--hash1",
                    "dtype": "fp16",
                    "size_mb": 100.0,
                    "benchmark": {"fps": 30.5, "mean_ms": 32.8, "device": "NPU"},
                },
                {
                    "model_id": "test/model2",
                    "model_name": "test--model2--int8--hash2",
                    "dtype": "int8",
                    "size_mb": 50.0,
                    "benchmark": {"fps": 45.2, "mean_ms": 22.1, "device": "GPU"},
                },
            ],
            "summary": {
                "fastest_model": {"id": "test/model2", "fps": 45.2, "device": "GPU"},
                "slowest_model": {"id": "test/model1", "fps": 30.5, "device": "NPU"},
                "average_fps": 37.85,
            },
        }

        args = Mock()
        args.cache_dir = None
        args.warmup = 3
        args.runs = 10
        args.output = None
        args.json = False

        model_benchmark(args)

        mock_benchmark.assert_called_once_with(cache_dir=None, warmup=3, runs=10)

        captured = capsys.readouterr()
        assert "Benchmark Results (2/2 models)" in captured.out
        assert "Fastest: test/model2 (45.2 FPS on GPU)" in captured.out
        assert "Slowest: test/model1 (30.5 FPS on NPU)" in captured.out
        assert "Average performance: 37.85 FPS" in captured.out
        assert "test/model1 (fp16)" in captured.out
        assert "Device: NPU | FPS: 30.5 | Latency: 32.8ms" in captured.out

    @patch("oe.models.benchmark_all")
    def test_benchmark_with_failures(self, mock_benchmark, capsys):
        """Test benchmarking with some failures."""
        mock_benchmark.return_value = {
            "total_models": 2,
            "benchmarked": 1,
            "failed": 1,
            "results": [
                {
                    "model_id": "test/good_model",
                    "benchmark": {"fps": 25.0, "device": "CPU"},
                },
                {
                    "model_name": "test--bad--model",
                    "error": "Failed to load model",
                    "benchmarked": False,
                },
            ],
            "summary": {
                "fastest_model": {
                    "id": "test/good_model",
                    "fps": 25.0,
                    "device": "CPU",
                },
                "slowest_model": {
                    "id": "test/good_model",
                    "fps": 25.0,
                    "device": "CPU",
                },
                "average_fps": 25.0,
            },
        }

        args = Mock()
        args.cache_dir = None
        args.warmup = 3
        args.runs = 10
        args.output = None
        args.json = False

        model_benchmark(args)

        captured = capsys.readouterr()
        assert "Benchmark Results (1/2 models)" in captured.out
        assert "test--bad--model: Failed to load model" in captured.out

    @patch("oe.models.benchmark_all")
    def test_benchmark_no_models(self, mock_benchmark, capsys):
        """Test benchmarking with no models."""
        mock_benchmark.return_value = {
            "total_models": 0,
            "benchmarked": 0,
            "failed": 0,
            "results": [],
            "summary": {},
        }

        args = Mock()
        args.cache_dir = None
        args.warmup = 3
        args.runs = 10
        args.output = None
        args.json = False

        model_benchmark(args)

        captured = capsys.readouterr()
        assert "No models were successfully benchmarked" in captured.out

    @patch("oe.models.benchmark_all")
    def test_benchmark_with_output_file(self, mock_benchmark):
        """Test benchmarking with output file."""
        mock_results = {
            "total_models": 1,
            "benchmarked": 1,
            "results": [],
            "summary": {},
        }
        mock_benchmark.return_value = mock_results

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            output_file = f.name

        args = Mock()
        args.cache_dir = None
        args.warmup = 3
        args.runs = 10
        args.output = output_file
        args.json = False

        model_benchmark(args)

        # Verify file was written
        with open(output_file, "r") as f:
            saved_results = json.load(f)

        assert saved_results == mock_results

        # Cleanup
        Path(output_file).unlink()


@pytest.mark.integration
class TestCLIModelsIntegration:
    """Integration tests for CLI model commands."""

    def test_full_cli_workflow(self):
        """Test complete CLI workflow: search -> info -> install -> validate -> benchmark."""
        # This would test the full workflow but requires network and time
        pytest.skip("Integration test - requires network access")

    def test_cli_error_handling_robustness(self):
        """Test CLI robustness with various error conditions."""
        # Test network failures, permission errors, corrupted models, etc.
        pytest.skip("Integration test - requires specific error conditions")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
