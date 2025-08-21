"""Tests for audio model support in OpenVINO-Easy."""

import pytest
import numpy as np
import tempfile
import wave
from pathlib import Path
from unittest.mock import patch, MagicMock

from oe.runtime import RuntimeWrapper

# Check if optional dependencies are available
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class TestAudioSupport:
    """Test audio model loading and inference capabilities."""

    def create_test_audio_file(self, duration=1, sample_rate=16000):
        """Create a test WAV file."""
        # Generate a simple sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(440 * 2 * np.pi * t) * 0.5
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Save to temporary file
        temp_file = Path(tempfile.mktemp(suffix=".wav"))
        with wave.open(str(temp_file), "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return temp_file, audio_data.astype(np.float32)

    def test_audio_file_detection(self):
        """Test that audio files are correctly detected."""
        # Create mock runtime
        mock_compiled_model = MagicMock()
        runtime = RuntimeWrapper(mock_compiled_model, "CPU", {})

        # Test various audio file extensions
        test_files = [
            "/path/to/audio.wav",
            "/path/to/audio.mp3",
            "/path/to/audio.flac",
            "/path/to/audio.ogg",
        ]

        for test_file in test_files:
            with patch.object(Path, "exists", return_value=True):
                assert runtime._is_audio_file(test_file), (
                    f"Should detect {test_file} as audio"
                )

        # Test non-audio files
        non_audio_files = [
            "/path/to/text.txt",
            "/path/to/image.jpg",
            "/path/to/model.xml",
        ]

        for test_file in non_audio_files:
            with patch.object(Path, "exists", return_value=True):
                assert not runtime._is_audio_file(test_file), (
                    f"Should not detect {test_file} as audio"
                )

    @pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="librosa not available")
    def test_audio_preprocessing_with_librosa(self):
        """Test audio preprocessing with librosa."""
        # Create test audio file
        temp_file, expected_audio = self.create_test_audio_file()

        try:
            # Create mock runtime with audio input
            mock_compiled_model = MagicMock()
            mock_compiled_model.inputs = [MagicMock()]
            mock_compiled_model.inputs[0].any_name = "input_features"

            runtime = RuntimeWrapper(mock_compiled_model, "CPU", {})
            runtime.input_info = {"input_features": {"shape": [1, -1]}}

            # Mock librosa to control the loading
            with patch("librosa.load") as mock_librosa:
                mock_librosa.return_value = (expected_audio, 16000)

                result = runtime._preprocess_audio_input(temp_file)

                # Check that librosa was called correctly
                mock_librosa.assert_called_once_with(str(temp_file), sr=16000)

                # Check result format
                assert "input_features" in result
                assert isinstance(result["input_features"], np.ndarray)
                assert result["input_features"].dtype == np.float32
                assert result["input_features"].shape[0] == 1  # Batch dimension

        finally:
            # Clean up
            if temp_file.exists():
                temp_file.unlink()

    def test_audio_preprocessing_fallback_wav(self):
        """Test audio preprocessing fallback for WAV files without librosa."""
        # Skip if librosa is actually available - this test is for fallback behavior
        try:
            import librosa

            pytest.skip("librosa is available, skipping fallback test")
        except ImportError:
            pass  # Good, librosa not available, test the fallback

        temp_file, _ = self.create_test_audio_file()

        try:
            # Create mock runtime
            mock_compiled_model = MagicMock()
            runtime = RuntimeWrapper(mock_compiled_model, "CPU", {})
            runtime.input_info = {"input": {"shape": [1, -1]}}

            # Test fallback (librosa not available)
            result = runtime._preprocess_audio_input(temp_file)

            # Check result format
            assert "input" in result
            assert isinstance(result["input"], np.ndarray)
            assert result["input"].dtype == np.float32
            assert result["input"].shape[0] == 1  # Batch dimension

        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_audio_input_name_detection(self):
        """Test that audio input names are correctly detected."""
        mock_compiled_model = MagicMock()
        runtime = RuntimeWrapper(mock_compiled_model, "CPU", {})

        # Test with common audio input names
        test_cases = [
            ({"input_features": {}}, "input_features"),
            ({"audio": {}}, "audio"),
            ({"input": {}}, "input"),
            ({"waveform": {}}, "waveform"),
            ({"speech": {}}, "speech"),
            ({"some_other_name": {}}, "some_other_name"),  # Fallback to first
        ]

        for input_info, expected_name in test_cases:
            runtime.input_info = input_info
            result = runtime._get_audio_input_name()
            assert result == expected_name

    @pytest.mark.integration
    def test_audio_model_type_detection(self):
        """Test that audio models are correctly detected."""
        from oe.loader import _detect_model_type

        # Create mock model directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)

            # Create config.json for audio model
            config_content = {
                "model_type": "whisper",
                "architectures": ["WhisperForConditionalGeneration"],
                "task": "automatic-speech-recognition",
            }

            with open(model_path / "config.json", "w") as f:
                import json

                json.dump(config_content, f)

            model_type = _detect_model_type(model_path)
            assert model_type == "transformers_audio"

    @pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="librosa not available")
    def test_audio_file_path_handling(self):
        """Test that string paths that are audio files are handled correctly."""
        temp_file, _ = self.create_test_audio_file()

        try:
            # Create mock runtime
            mock_compiled_model = MagicMock()
            runtime = RuntimeWrapper(mock_compiled_model, "CPU", {})
            runtime.input_info = {"input": {"shape": [1, -1]}}

            # Test with Path object
            with patch("librosa.load") as mock_librosa:
                mock_librosa.return_value = (
                    np.random.randn(16000).astype(np.float32),
                    16000,
                )

                # Should detect as audio file and preprocess
                result = runtime._preprocess_input(temp_file)
                assert "input" in result
                assert isinstance(result["input"], np.ndarray)

                # Test with string path
                result = runtime._preprocess_input(str(temp_file))
                assert "input" in result
                assert isinstance(result["input"], np.ndarray)

        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_error_handling_missing_dependencies(self):
        """Test error handling when audio dependencies are missing."""
        # Skip if librosa is available - this test checks error handling without it
        try:
            import librosa

            pytest.skip("librosa is available, skipping missing dependency test")
        except ImportError:
            pass  # Good, librosa not available

        # Create an MP3 file which requires librosa (WAV can use builtin wave module)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_file = Path(f.name)
            f.write(b"fake mp3 content")  # Just some fake content

        try:
            mock_compiled_model = MagicMock()
            runtime = RuntimeWrapper(mock_compiled_model, "CPU", {})
            runtime.input_info = {"input": {"shape": [1, -1]}}

            # MP3 file without librosa should raise ValueError
            with pytest.raises(ValueError) as exc_info:
                runtime._preprocess_audio_input(temp_file)

            assert "not supported" in str(exc_info.value).lower()
            assert "librosa" in str(exc_info.value).lower()

        finally:
            if temp_file.exists():
                temp_file.unlink()

    @pytest.mark.skipif(
        LIBROSA_AVAILABLE, reason="librosa is available, skip fallback test"
    )
    def test_non_wav_file_without_librosa(self):
        """Test that non-WAV files require librosa when librosa is not available."""
        mock_compiled_model = MagicMock()
        runtime = RuntimeWrapper(mock_compiled_model, "CPU", {})

        # When librosa is not available, non-WAV files should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            runtime._preprocess_audio_input("/path/to/audio.mp3")

        assert "not supported" in str(exc_info.value)
        assert "librosa" in str(exc_info.value)

    def cleanup_files(self):
        """Clean up any temporary files created during tests."""
        # This would be called in teardown if needed
        pass
