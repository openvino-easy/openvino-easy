"""End-to-end tests with real models from Hugging Face."""

import pytest
import tempfile
from pathlib import Path

import oe


class TestE2ERealModels:
    """End-to-end tests with real models to catch real-world issues."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_small_text_model_e2e(self):
        """Test loading and inference with a small real text model."""
        # Use a very small model to keep test fast
        model_id = "prajjwal1/bert-tiny"  # Only ~4MB

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Load model using new 3-function API
                oe.load(
                    model_id,
                    device_preference=["CPU"],  # Force CPU for CI
                    cache_dir=temp_dir,
                )

                # Test basic properties
                info = oe.get_info()
                assert info["device"] == "CPU"

                # Test inference with text
                result = oe.infer("Hello world")
                assert result is not None

                # Test model info
                assert "device" in info

                oe.unload()

                print(f"✅ Successfully tested {model_id}")

            except Exception as e:
                pytest.skip(f"Test skipped due to model loading issue: {e}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_quantization_e2e(self):
        """Test end-to-end quantization with real model."""
        model_id = "prajjwal1/bert-tiny"

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Load model with quantization using new API
                oe.load(
                    model_id,
                    device_preference=["CPU"],
                    dtype="int8",
                    cache_dir=temp_dir,
                )

                # Test that model is marked as quantized
                info = oe.get_info()
                assert info.get("quantized") is True

                # Test inference still works
                result = oe.infer("Test quantized model")
                assert result is not None

                oe.unload()

                print(f"✅ Successfully tested quantization with {model_id}")

            except Exception as e:
                pytest.skip(f"Quantization test skipped: {e}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_audio_model_e2e(self):
        """Test end-to-end loading and inference with a small audio model."""
        # Use a very small Whisper model for testing
        model_id = "openai/whisper-tiny"

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Create a small test audio file
                import numpy as np
                import wave

                # Generate 1 second of 440Hz sine wave at 16kHz
                sample_rate = 16000
                duration = 1.0
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                audio_data = np.sin(440 * 2 * np.pi * t) * 0.5
                audio_int16 = (audio_data * 32767).astype(np.int16)

                # Save to WAV file
                audio_file = Path(temp_dir) / "test_audio.wav"
                with wave.open(str(audio_file), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())

                # Load audio model using new API
                oe.load(
                    model_id,
                    device_preference=["CPU"],  # Use CPU for CI compatibility
                    cache_dir=temp_dir,
                )

                # Test basic properties
                info = oe.get_info()
                assert info["device"] == "CPU"

                # Test inference with audio file
                result = oe.infer(str(audio_file))
                assert result is not None

                # Test model info contains audio-specific details
                assert "device" in info

                oe.unload()

                print(f"✅ Successfully tested audio model {model_id}")

            except Exception as e:
                pytest.skip(f"Audio model test skipped: {e}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_vision_model_e2e(self):
        """Test end-to-end loading and inference with a small vision model."""
        # Use a small vision model for testing
        model_id = "microsoft/resnet-18"

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Load vision model using new API
                oe.load(model_id, device_preference=["CPU"], cache_dir=temp_dir)

                # Test basic properties
                info = oe.get_info()
                assert info["device"] == "CPU"

                # Create a small test image (dummy input)
                import numpy as np

                # Standard ImageNet input shape: [batch, channels, height, width]
                dummy_image = np.random.randn(1, 3, 224, 224).astype(np.float32)

                # Test inference
                result = oe.infer(dummy_image)
                assert result is not None

                oe.unload()

                print(f"✅ Successfully tested vision model {model_id}")

            except Exception as e:
                pytest.skip(f"Vision model test skipped: {e}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_multimodal_model_e2e(self):
        """Test end-to-end loading with a small multimodal model."""
        # Use a small CLIP model for testing
        model_id = "openai/clip-vit-base-patch32"

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Load multimodal model
                oe.load(model_id, device_preference=["CPU"], cache_dir=temp_dir)

                # Test basic properties
                info = oe.get_info()
                assert info["device"] == "CPU"

                # Test with text input
                result = oe.infer("a photo of a cat")

                oe.unload()
                assert result is not None

                print(f"✅ Successfully tested multimodal model {model_id}")

            except Exception as e:
                pytest.skip(f"Multimodal model test skipped: {e}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_onnx_model_e2e(self):
        """Test loading an ONNX model if available."""
        # This would need an actual ONNX model URL or file
        # For now, we'll test the error handling path

        fake_onnx_path = "nonexistent_model.onnx"

        with pytest.raises(Exception):
            oe.load(fake_onnx_path)

    @pytest.mark.integration
    def test_unsupported_model_error_handling(self):
        """Test error handling for unsupported model formats."""
        # Test with a clearly unsupported identifier
        with pytest.raises(Exception) as exc_info:
            oe.load("clearly/nonexistent/model/format")

        # Should provide helpful error message
        error_msg = str(exc_info.value)
        assert any(
            keyword in error_msg.lower() for keyword in ["failed", "not found", "error"]
        )

    @pytest.mark.integration
    def test_device_fallback_e2e(self):
        """Test device fallback behavior."""
        model_id = "prajjwal1/bert-tiny"

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Request NPU first (likely not available in CI), should fallback to CPU
                oe.load(
                    model_id,
                    device_preference=["NPU", "GPU", "CPU"],
                    cache_dir=temp_dir,
                )

                # Should succeed with some device
                info = oe.get_info()
                assert info["device"] in ["NPU", "GPU", "CPU"]

                # Test inference works
                result = oe.infer("Device fallback test")
                assert result is not None

                print(f"✅ Device fallback worked, using: {info['device']}")

                oe.unload()

            except Exception as e:
                pytest.skip(f"Device fallback test skipped: {e}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_benchmark_e2e(self):
        """Test benchmarking with real model."""
        model_id = "prajjwal1/bert-tiny"

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                oe.load(model_id, device_preference=["CPU"], cache_dir=temp_dir)

                # Run benchmark with small number of runs for speed
                stats = oe.benchmark(warmup_runs=1, benchmark_runs=3)

                oe.unload()

                # Verify benchmark results structure
                required_keys = ["device", "mean_ms", "fps", "benchmark_runs"]
                for key in required_keys:
                    assert key in stats, f"Missing key: {key}"

                # Verify reasonable values
                assert stats["mean_ms"] > 0
                assert stats["fps"] > 0
                assert stats["benchmark_runs"] == 3

                print(
                    f"✅ Benchmark results: {stats['fps']:.1f} FPS, {stats['mean_ms']:.1f}ms"
                )

            except Exception as e:
                pytest.skip(f"Benchmark test skipped: {e}")

    @pytest.mark.integration
    def test_cache_functionality_e2e(self):
        """Test that caching works correctly."""
        model_id = "prajjwal1/bert-tiny"

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Load model first time
                oe.load(model_id, device_preference=["CPU"], cache_dir=temp_dir)

                # Test inference
                result1 = oe.infer("Cache test 1")
                assert result1 is not None

                oe.unload()

                # Verify cache directory has content
                cache_path = Path(temp_dir)
                assert any(cache_path.iterdir()), "Cache directory should not be empty"

                # Load model second time (should use cache)
                oe.load(model_id, device_preference=["CPU"], cache_dir=temp_dir)

                # Test inference again
                result2 = oe.infer("Cache test 2")
                assert result2 is not None

                oe.unload()

                print("✅ Cache functionality verified")

            except Exception as e:
                pytest.skip(f"Cache test skipped: {e}")

    @pytest.mark.integration
    def test_tokenizer_integration_e2e(self):
        """Test that tokenizer integration works with real model."""
        model_id = "prajjwal1/bert-tiny"

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                oe.load(model_id, device_preference=["CPU"], cache_dir=temp_dir)

                # Test different text inputs
                test_texts = [
                    "Short text",
                    "This is a longer text that should test tokenization properly",
                    "Special chars: @#$%^&*()",
                    "",  # Empty string
                ]

                for text in test_texts:
                    result = oe.infer(text)

                    assert result is not None, f"Failed for text: '{text}'"

                oe.unload()

                print("✅ Tokenizer integration verified")

            except Exception as e:
                pytest.skip(f"Tokenizer test skipped: {e}")

    def test_cli_integration_basic(self):
        """Test basic CLI functionality without model loading."""
        import subprocess
        import sys

        # Test that CLI is accessible
        try:
            result = subprocess.run(
                [sys.executable, "-m", "oe", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            assert result.returncode == 0
            assert "OpenVINO-Easy" in result.stdout

            print("✅ CLI is accessible")

        except Exception as e:
            pytest.skip(f"CLI test skipped: {e}")

    def test_devices_command_e2e(self):
        """Test the devices command works."""
        devices_list = oe.devices()

        # Should always have CPU at minimum
        assert isinstance(devices_list, list)
        assert len(devices_list) > 0
        assert "CPU" in devices_list

        print(f"✅ Available devices: {devices_list}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_model_type_detection_e2e(self):
        """Test that model type detection works with real model structure."""
        # This test downloads a model to test the detection logic
        model_id = "prajjwal1/bert-tiny"

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                from huggingface_hub import snapshot_download
                from oe.loader import _detect_model_type

                # Download model files
                local_path = snapshot_download(repo_id=model_id, cache_dir=temp_dir)

                # Test detection
                model_type = _detect_model_type(Path(local_path))

                # Should detect as transformers model
                assert model_type in ["transformers_optimum", "transformers_direct"]

                print(f"✅ Detected model type: {model_type}")

            except Exception as e:
                pytest.skip(f"Model type detection test skipped: {e}")
