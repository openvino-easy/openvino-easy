"""Unit tests for oe.cli."""

import pytest
import json
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path
from io import StringIO
import sys
from oe.cli import main, run_inference, run_benchmark, list_devices

class TestCLI:
    """Test CLI functionality."""
    
    @patch('sys.argv', ['oe', '--help'])
    @patch('sys.stdout', new_callable=StringIO)
    def test_help_output(self, mock_stdout):
        """Test help output."""
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        help_text = mock_stdout.getvalue()
        assert "OpenVINO-Easy" in help_text
        assert "run" in help_text
        assert "bench" in help_text
        assert "devices" in help_text

    @patch('sys.argv', ['oe'])
    @patch('sys.stdout', new_callable=StringIO)
    def test_no_command(self, mock_stdout):
        """Test behavior when no command is provided."""
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        help_text = mock_stdout.getvalue()
        assert "OpenVINO-Easy" in help_text

    @patch('oe.devices')
    def test_devices_command(self, mock_devices):
        """Test devices command."""
        mock_devices.return_value = ["CPU", "GPU", "NPU"]
        
        args = MagicMock()
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            list_devices(args)
        
        output = mock_stdout.getvalue()
        assert "Scanning OpenVINO devices" in output
        assert "CPU" in output
        assert "GPU" in output
        assert "NPU" in output

    @patch('oe.load')
    def test_run_inference_success(self, mock_load):
        """Test successful inference run."""
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.infer.return_value = {"result": "test_output"}
        mock_load.return_value = mock_pipeline
        
        args = MagicMock()
        args.model = "test/model"
        args.dtype = "fp16"
        args.device_preference = None
        args.prompt = "test prompt"
        args.output = None
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            run_inference(args)
        
        output = mock_stdout.getvalue()
        assert "test_output" in output
        
        # Verify load was called correctly
        mock_load.assert_called_once_with(
            "test/model",
            device_preference=None,
            dtype="fp16"
        )

    @patch('oe.load')
    def test_run_inference_with_device_preference(self, mock_load):
        """Test inference with device preference."""
        mock_pipeline = MagicMock()
        mock_pipeline.infer.return_value = {"result": "test_output"}
        mock_load.return_value = mock_pipeline
        
        args = MagicMock()
        args.model = "test/model"
        args.dtype = "fp16"
        args.device_preference = "NPU,GPU,CPU"
        args.prompt = "test prompt"
        args.output = None
        
        with patch('sys.stdout', new_callable=StringIO):
            run_inference(args)
        
        # Verify device preference was parsed correctly
        mock_load.assert_called_once_with(
            "test/model",
            device_preference=["NPU", "GPU", "CPU"],
            dtype="fp16"
        )

    @patch('oe.load')
    def test_run_inference_with_output_file(self, mock_load):
        """Test inference with output file."""
        mock_pipeline = MagicMock()
        mock_pipeline.infer.return_value = {"result": "test_output"}
        mock_load.return_value = mock_pipeline
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            args = MagicMock()
            args.model = "test/model"
            args.dtype = "fp16"
            args.device_preference = None
            args.prompt = "test prompt"
            args.output = temp_path
            
            with patch('sys.stdout', new_callable=StringIO):
                run_inference(args)
            
            # Check that output was written to file
            with open(temp_path, 'r') as f:
                output_data = json.load(f)
            assert output_data["result"] == {"result": "test_output"}
            
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @patch('oe.load')
    def test_run_inference_error(self, mock_load):
        """Test inference error handling."""
        mock_load.side_effect = Exception("Model loading failed")
        
        args = MagicMock()
        args.model = "test/model"
        args.dtype = "fp16"
        args.device_preference = None
        args.prompt = "test prompt"
        args.output = None
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with pytest.raises(SystemExit) as exc_info:
                run_inference(args)
        
        assert exc_info.value.code == 1
        error_output = mock_stdout.getvalue()
        assert "Error: Model loading failed" in error_output

    @patch('oe.load')
    def test_run_benchmark_success(self, mock_load):
        """Test successful benchmark run."""
        mock_pipeline = MagicMock()
        mock_pipeline.benchmark.return_value = {
            "device": "NPU",
            "mean_ms": 5.3,
            "fps": 188.7
        }
        mock_load.return_value = mock_pipeline
        
        args = MagicMock()
        args.model = "test/model"
        args.dtype = "fp16"
        args.device_preference = None
        args.warmup_runs = 3
        args.benchmark_runs = 10
        args.output = None
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            run_benchmark(args)
        
        output = mock_stdout.getvalue()
        assert "NPU" in output
        assert "5.3" in output
        assert "188.7" in output
        
        # Verify benchmark was called correctly
        mock_pipeline.benchmark.assert_called_once_with(warmup_runs=3, benchmark_runs=10)

    @patch('oe.load')
    def test_run_benchmark_with_output_file(self, mock_load):
        """Test benchmark with output file."""
        mock_pipeline = MagicMock()
        mock_pipeline.benchmark.return_value = {
            "device": "NPU",
            "mean_ms": 5.3,
            "fps": 188.7
        }
        mock_load.return_value = mock_pipeline
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            args = MagicMock()
            args.model = "test/model"
            args.dtype = "fp16"
            args.device_preference = None
            args.warmup_runs = 3
            args.benchmark_runs = 10
            args.output = temp_path
            
            with patch('sys.stdout', new_callable=StringIO):
                run_benchmark(args)
            
            # Check that output was written to file
            with open(temp_path, 'r') as f:
                output_data = json.load(f)
            assert output_data["device"] == "NPU"
            assert output_data["mean_ms"] == 5.3
            assert output_data["fps"] == 188.7
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @patch('oe.load')
    def test_run_benchmark_error(self, mock_load):
        """Test benchmark error handling."""
        mock_load.side_effect = Exception("Benchmark failed")
        
        args = MagicMock()
        args.model = "test/model"
        args.dtype = "fp16"
        args.device_preference = None
        args.warmup_runs = 3
        args.benchmark_runs = 10
        args.output = None
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with pytest.raises(SystemExit) as exc_info:
                run_benchmark(args)
        
        assert exc_info.value.code == 1
        error_output = mock_stdout.getvalue()
        assert "Error: Benchmark failed" in error_output

    @patch('oe.devices')
    def test_devices_command_error(self, mock_devices):
        """Test devices command error handling."""
        mock_devices.side_effect = Exception("Device detection failed")
        
        args = MagicMock()
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            list_devices(args)
        output = mock_stdout.getvalue()
        assert "Scanning OpenVINO devices" in output or "Device detection failed" in output

    @patch('sys.argv', ['oe', 'run', 'test/model', '--prompt', 'test prompt', '--dtype', 'int8'])
    @patch('oe.load')
    def test_cli_run_command(self, mock_load):
        """Test CLI run command parsing."""
        mock_pipeline = MagicMock()
        mock_pipeline.infer.return_value = {"result": "test"}
        mock_load.return_value = mock_pipeline
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            main()
        output = mock_stdout.getvalue()
        assert "test" in output

    @patch('sys.argv', ['oe', 'bench', 'test/model', '--warmup-runs', '10', '--benchmark-runs', '50'])
    @patch('oe.load')
    def test_cli_bench_command(self, mock_load):
        """Test CLI bench command parsing."""
        mock_pipeline = MagicMock()
        mock_pipeline.benchmark.return_value = {"device": "CPU", "fps": 100}
        mock_load.return_value = mock_pipeline
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            main()
        output = mock_stdout.getvalue()
        assert "CPU" in output
        assert "100" in output

    @patch('sys.argv', ['oe', 'devices'])
    @patch('oe.devices')
    def test_cli_devices_command(self, mock_devices):
        """Test CLI devices command."""
        mock_devices.return_value = ["CPU", "GPU"]
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            main()
        output = mock_stdout.getvalue()
        assert "Scanning OpenVINO devices" in output
        assert "CPU" in output
        assert "GPU" in output 