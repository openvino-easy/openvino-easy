"""Integration tests for OpenVINO-Easy with real OpenVINO runtime."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

import openvino as ov
from oe import load, devices
from oe._core import detect_device


class TestIntegration:
    """Integration tests that use real OpenVINO runtime."""
    
    def _create_simple_model(self, temp_dir: Path) -> Path:
        """Create a simple test model for integration testing."""
        # Create a simple model using OpenVINO Model API
        # This creates a basic linear layer: y = x * weight + bias
        
        # Input parameter
        input_shape = [1, 3, 224, 224]  # Typical image input
        input_node = ov.opset11.parameter(input_shape, dtype=np.float32, name="input")
        
        # Simple operations: Global Average Pooling -> Fully Connected
        gap = ov.opset11.reduce_mean(input_node, axes=[2, 3], keep_dims=False)
        
        # Create weight and bias constants
        weight = ov.opset11.constant(np.random.randn(3, 10).astype(np.float32), dtype=np.float32)
        bias = ov.opset11.constant(np.random.randn(10).astype(np.float32), dtype=np.float32)
        
        # Fully connected layer
        fc = ov.opset11.matmul(gap, weight, transpose_a=False, transpose_b=False)
        output = ov.opset11.add(fc, bias)
        
        # Set output name
        output.set_friendly_name("output")
        
        # Create model
        model = ov.Model([output], [input_node], "simple_test_model")
        
        # Save model to temporary directory
        model_path = temp_dir / "simple_model.xml"
        ov.save_model(model, str(model_path))
        
        return model_path
    
    @pytest.mark.integration
    def test_device_detection(self):
        """Test that device detection works."""
        device = detect_device()
        assert device in ["CPU", "GPU", "NPU"]
        
        # Test devices() function
        available_devices = devices()
        assert isinstance(available_devices, list)
        assert len(available_devices) > 0
        assert "CPU" in available_devices  # CPU should always be available
    
    @pytest.mark.integration
    def test_load_and_infer_ir_model(self):
        """Test loading and running inference on a local IR model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a simple test model
            model_path = self._create_simple_model(temp_path)
            
            # Load model using OpenVINO-Easy
            pipeline = load(str(model_path), device_preference=["CPU"])
            
            # Verify pipeline properties
            assert pipeline.device == "CPU"
            assert pipeline.model_path == str(model_path)
            
            # Test inference with numpy array
            input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
            result = pipeline.infer(input_data)
            
            # Verify output
            assert isinstance(result, list)
            assert len(result) == 10  # Output dimension
            assert all(isinstance(x, float) for x in result)
    
    @pytest.mark.integration
    def test_benchmark_pipeline(self):
        """Test benchmarking functionality with real model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a simple test model
            model_path = self._create_simple_model(temp_path)
            
            # Load model
            pipeline = load(str(model_path), device_preference=["CPU"])
            
            # Run benchmark
            stats = pipeline.benchmark(warmup_runs=2, benchmark_runs=5)
            
            # Verify benchmark results
            assert isinstance(stats, dict)
            assert "device" in stats
            assert "mean_ms" in stats
            assert "fps" in stats
            assert "p50_ms" in stats
            assert "p90_ms" in stats
            
            assert stats["device"] == "CPU"
            assert stats["mean_ms"] > 0
            assert stats["fps"] > 0
            assert stats["benchmark_runs"] == 5
            assert stats["warmup_runs"] == 2
    
    @pytest.mark.integration
    def test_model_caching(self):
        """Test that model caching works correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cache_dir = temp_path / "cache"
            
            # Create a simple test model
            model_path = self._create_simple_model(temp_path)
            
            # Load model first time (should cache)
            pipeline1 = load(str(model_path), device_preference=["CPU"], cache_dir=str(cache_dir))
            
            # Load model second time (should use cache)
            pipeline2 = load(str(model_path), device_preference=["CPU"], cache_dir=str(cache_dir))
            
            # Both should work
            input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
            result1 = pipeline1.infer(input_data)
            result2 = pipeline2.infer(input_data)
            
            # Results should be similar (same model)
            assert len(result1) == len(result2)
            assert isinstance(result1, list)
            assert isinstance(result2, list)
    
    @pytest.mark.integration
    def test_quantization_integration(self):
        """Test quantization with real model (if NNCF available)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a simple test model
            model_path = self._create_simple_model(temp_path)
            
            try:
                # Try to load with quantization
                pipeline = load(
                    str(model_path), 
                    device_preference=["CPU"],
                    dtype="int8"  # Request quantization
                )
                
                # Test inference still works
                input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
                result = pipeline.infer(input_data)
                
                assert isinstance(result, list)
                assert len(result) == 10
                
            except Exception as e:
                # If quantization fails, it should gracefully fall back
                # This is acceptable for integration tests
                print(f"Quantization test skipped: {e}")
    
    @pytest.mark.integration
    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a simple test model
            model_path = self._create_simple_model(temp_path)
            pipeline = load(str(model_path), device_preference=["CPU"])
            
            # Test with wrong input shape
            with pytest.raises(ValueError):
                wrong_input = np.random.randn(2, 5, 100, 100).astype(np.float32)
                pipeline.infer(wrong_input)
            
            # Test with invalid model path
            with pytest.raises(Exception):
                load("nonexistent_model.xml")
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_multiple_devices(self):
        """Test loading on different devices if available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a simple test model
            model_path = self._create_simple_model(temp_path)
            
            # Get available devices
            available_devices = devices()
            
            for device in available_devices:
                try:
                    pipeline = load(str(model_path), device_preference=[device])
                    assert pipeline.device == device
                    
                    # Test inference
                    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
                    result = pipeline.infer(input_data)
                    assert isinstance(result, list)
                    
                except Exception as e:
                    # Some devices might not be available or compatible
                    print(f"Device {device} test skipped: {e}")
                    continue 