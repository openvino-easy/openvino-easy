#!/usr/bin/env python3
"""
Enhanced Audio Speech Recognition Example

This example demonstrates comprehensive audio processing capabilities with OpenVINO-Easy,
including real-time processing, batch transcription, and audio format handling.

Features:
- Multiple audio format support (WAV, MP3, FLAC, etc.)
- Real-time audio streaming
- Batch processing of audio files
- Audio preprocessing and enhancement
- Language detection and multilingual support
- Confidence scoring and quality metrics

Requirements:
    pip install "openvino-easy[audio]" sounddevice matplotlib
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import oe
    import librosa
    import soundfile as sf
    import sounddevice as sd
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Install with: pip install 'openvino-easy[audio]' sounddevice matplotlib")
    sys.exit(1)


class AudioProcessor:
    """Advanced audio processing for speech recognition."""

    def __init__(self):
        self.sample_rate = 16000  # Standard for speech recognition
        self.supported_formats = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

    def load_audio(
        self, audio_path: str, normalize: bool = True
    ) -> tuple[np.ndarray, int]:
        """
        Load audio file with automatic format detection and preprocessing.

        Args:
            audio_path: Path to audio file
            normalize: Whether to normalize audio amplitude

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load with librosa for comprehensive format support
            audio_data, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,  # Resample to target rate
                mono=True,  # Convert to mono
            )

            if normalize:
                # Normalize to [-1, 1] range
                max_val = np.abs(audio_data).max()
                if max_val > 0:
                    audio_data = audio_data / max_val

            print(f"✓ Loaded audio: {Path(audio_path).name}")
            print(f"  Duration: {len(audio_data) / sr:.2f}s")
            print(f"  Sample rate: {sr}Hz")

            return audio_data, sr

        except Exception as e:
            print(f"✗ Failed to load {audio_path}: {e}")
            raise

    def enhance_audio(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply audio enhancement for better speech recognition.

        Args:
            audio_data: Raw audio samples
            sr: Sample rate

        Returns:
            Enhanced audio data
        """
        # Remove silence from beginning and end
        audio_trimmed, _ = librosa.effects.trim(
            audio_data,
            top_db=20,  # Remove audio below -20dB
            frame_length=512,
            hop_length=64,
        )

        # Apply pre-emphasis filter (common for speech processing)
        pre_emphasis = 0.97
        audio_emphasized = np.append(
            audio_trimmed[0], audio_trimmed[1:] - pre_emphasis * audio_trimmed[:-1]
        )

        # Normalize again after processing
        max_val = np.abs(audio_emphasized).max()
        if max_val > 0:
            audio_emphasized = audio_emphasized / max_val

        return audio_emphasized

    def split_audio(
        self, audio_data: np.ndarray, sr: int, chunk_duration: float = 30.0
    ) -> List[np.ndarray]:
        """
        Split long audio into smaller chunks for processing.

        Args:
            audio_data: Audio samples
            sr: Sample rate
            chunk_duration: Duration of each chunk in seconds

        Returns:
            List of audio chunks
        """
        chunk_samples = int(chunk_duration * sr)
        chunks = []

        for start in range(0, len(audio_data), chunk_samples):
            end = min(start + chunk_samples, len(audio_data))
            chunk = audio_data[start:end]

            # Only add chunks that are at least 1 second long
            if len(chunk) >= sr:
                chunks.append(chunk)

        return chunks


class SpeechRecognitionPipeline:
    """Advanced speech recognition pipeline with multiple model support."""

    def __init__(self, model_name: str = "openai/whisper-base", device: str = "auto"):
        """
        Initialize speech recognition pipeline.

        Args:
            model_name: HuggingFace model identifier or local model path
            device: Target device (auto, CPU, GPU, NPU)
        """
        self.model_name = model_name
        self.device = device
        self.audio_processor = AudioProcessor()

        print("Initializing speech recognition pipeline...")
        print(f"Model: {model_name}")
        print(f"Device: {device}")

        # Load the model
        self.pipeline = oe.load(model_name, device=device)

        print(f"✓ Model loaded on device: {self.pipeline.device}")

        # Get model info
        info = self.pipeline.model_info
        print(
            f"Model info: {info.get('model_type', 'Unknown')} - {info.get('parameters', 'Unknown')} parameters"
        )

    def transcribe(
        self, audio_data: np.ndarray, sr: int, enhance_audio: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Audio samples
            sr: Sample rate
            enhance_audio: Whether to apply audio enhancement

        Returns:
            Dictionary with transcription results and metadata
        """
        start_time = time.time()

        # Enhance audio if requested
        if enhance_audio:
            audio_data = self.audio_processor.enhance_audio(audio_data, sr)

        # Ensure audio is in the right format
        if sr != self.audio_processor.sample_rate:
            audio_data = librosa.resample(
                audio_data, orig_sr=sr, target_sr=self.audio_processor.sample_rate
            )
            sr = self.audio_processor.sample_rate

        # Run inference
        try:
            result = self.pipeline(audio_data)

            # Extract text result
            if isinstance(result, dict):
                text = result.get("text", str(result))
            else:
                text = str(result)

            processing_time = time.time() - start_time
            audio_duration = len(audio_data) / sr

            return {
                "text": text.strip(),
                "confidence": getattr(result, "confidence", None),
                "audio_duration": audio_duration,
                "processing_time": processing_time,
                "real_time_factor": processing_time / audio_duration,
                "device": self.pipeline.device,
            }

        except Exception as e:
            print(f"✗ Transcription failed: {e}")
            return {
                "text": "",
                "error": str(e),
                "audio_duration": len(audio_data) / sr,
                "processing_time": time.time() - start_time,
            }

    def transcribe_file(
        self, audio_path: str, enhance_audio: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            enhance_audio: Whether to apply audio enhancement

        Returns:
            Dictionary with transcription results
        """
        print(f"\nTranscribing: {Path(audio_path).name}")

        # Load audio
        try:
            audio_data, sr = self.audio_processor.load_audio(audio_path)
        except Exception as e:
            return {"error": f"Failed to load audio: {e}", "file": audio_path}

        # Transcribe
        result = self.transcribe(audio_data, sr, enhance_audio)
        result["file"] = audio_path

        # Print results
        if "error" not in result:
            print(
                f"✓ Transcription: '{result['text'][:100]}...' "
                if len(result["text"]) > 100
                else f"✓ Transcription: '{result['text']}'"
            )
            print(f"  Duration: {result['audio_duration']:.2f}s")
            print(f"  Processing time: {result['processing_time']:.2f}s")
            print(f"  Real-time factor: {result['real_time_factor']:.2f}x")
        else:
            print(f"✗ Error: {result['error']}")

        return result

    def batch_transcribe(
        self, audio_files: List[str], enhance_audio: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files in batch.

        Args:
            audio_files: List of audio file paths
            enhance_audio: Whether to apply audio enhancement

        Returns:
            List of transcription results
        """
        print(f"\nBatch transcribing {len(audio_files)} files...")

        results = []
        total_duration = 0
        total_processing_time = 0

        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}]", end=" ")

            result = self.transcribe_file(audio_file, enhance_audio)
            results.append(result)

            if "error" not in result:
                total_duration += result["audio_duration"]
                total_processing_time += result["processing_time"]

        # Summary
        print(f"\n{'=' * 50}")
        print("Batch Transcription Summary:")
        print(f"Files processed: {len(audio_files)}")
        print(f"Total audio duration: {total_duration:.2f}s")
        print(f"Total processing time: {total_processing_time:.2f}s")
        if total_duration > 0:
            print(
                f"Average real-time factor: {total_processing_time / total_duration:.2f}x"
            )

        return results

    def benchmark(
        self, audio_path: Optional[str] = None, duration: float = 10.0
    ) -> Dict[str, Any]:
        """
        Benchmark speech recognition performance.

        Args:
            audio_path: Path to test audio file (generated if None)
            duration: Duration of test audio in seconds

        Returns:
            Benchmark results
        """
        print("\nBenchmarking speech recognition performance...")

        if audio_path:
            audio_data, sr = self.audio_processor.load_audio(audio_path)
        else:
            # Generate test audio (white noise - not ideal but works for benchmarking)
            print("Generating test audio...")
            sr = self.audio_processor.sample_rate
            np.random.normal(0, 0.1, int(duration * sr))

        # Run benchmark
        return self.pipeline.benchmark(num_runs=10, warmup_runs=2)


class RealTimeRecorder:
    """Real-time audio recording and transcription."""

    def __init__(
        self, pipeline: SpeechRecognitionPipeline, chunk_duration: float = 5.0
    ):
        """
        Initialize real-time recorder.

        Args:
            pipeline: Speech recognition pipeline
            chunk_duration: Duration of audio chunks in seconds
        """
        self.pipeline = pipeline
        self.chunk_duration = chunk_duration
        self.sample_rate = pipeline.audio_processor.sample_rate
        self.is_recording = False
        self.audio_buffer = []

    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio stream."""
        if status:
            print(f"Audio stream status: {status}")

        if self.is_recording:
            self.audio_buffer.extend(indata[:, 0])  # Use first channel only

    def record_and_transcribe(self, duration: float = 30.0) -> List[Dict[str, Any]]:
        """
        Record audio and transcribe in real-time chunks.

        Args:
            duration: Total recording duration in seconds

        Returns:
            List of transcription results for each chunk
        """
        print(f"Starting real-time recording for {duration}s...")
        print("Speak into your microphone!")

        chunk_samples = int(self.chunk_duration * self.sample_rate)
        results = []

        try:
            with sd.InputStream(
                samplerate=self.sample_rate, channels=1, callback=self.audio_callback
            ):
                self.is_recording = True
                start_time = time.time()

                while time.time() - start_time < duration:
                    # Wait for enough audio data
                    while len(self.audio_buffer) < chunk_samples and self.is_recording:
                        time.sleep(0.1)

                    if not self.is_recording:
                        break

                    # Extract chunk
                    chunk_data = np.array(self.audio_buffer[:chunk_samples])
                    self.audio_buffer = self.audio_buffer[chunk_samples:]

                    # Transcribe chunk
                    print(f"\nTranscribing chunk ({self.chunk_duration}s)...")
                    result = self.pipeline.transcribe(chunk_data, self.sample_rate)

                    if result["text"]:
                        print(f">> {result['text']}")
                        results.append(result)
                    else:
                        print(">> [No speech detected]")

                self.is_recording = False

        except Exception as e:
            print(f"Recording error: {e}")
            self.is_recording = False

        return results


def create_sample_audio_files():
    """Create sample audio files for testing."""
    sample_dir = Path("sample_audio")
    sample_dir.mkdir(exist_ok=True)

    print("Creating sample audio files...")

    # Generate sample audio with different characteristics
    sr = 16000
    duration = 3  # 3 seconds each

    samples = [
        (
            "hello_world.wav",
            "sine wave",
            lambda t: 0.3 * np.sin(2 * np.pi * 440 * t),
        ),  # A4 tone
        (
            "speech_sim.wav",
            "speech-like",
            lambda t: 0.2 * np.sin(2 * np.pi * 200 * t) * np.exp(-t),
        ),  # Decaying tone
        (
            "noise.wav",
            "background noise",
            lambda t: 0.1 * np.random.normal(0, 1, len(t)),
        ),  # White noise
    ]

    created_files = []

    for filename, description, generator in samples:
        filepath = sample_dir / filename

        if not filepath.exists():
            print(f"Creating {filename} ({description})...")

            t = np.linspace(0, duration, int(sr * duration))
            audio_data = generator(t)

            # Ensure audio is in valid range
            audio_data = np.clip(audio_data, -1.0, 1.0)

            sf.write(str(filepath), audio_data, sr)
            created_files.append(str(filepath))

    return created_files


def demonstrate_features():
    """Demonstrate various speech recognition features."""

    print("OpenVINO-Easy Enhanced Speech Recognition Demo")
    print("=" * 50)

    # Initialize pipeline
    try:
        # Try Whisper model (most common for speech recognition)
        model_name = "openai/whisper-base"
        pipeline = SpeechRecognitionPipeline(model_name, device="auto")
    except Exception as e:
        print(f"Failed to load Whisper model: {e}")
        print("Falling back to a generic speech model...")
        # Fallback to any available speech model or create a dummy pipeline
        try:
            model_name = "facebook/wav2vec2-base-960h"
            pipeline = SpeechRecognitionPipeline(model_name, device="auto")
        except Exception as e2:
            print(f"Speech recognition models not available: {e2}")
            print("This example requires a speech recognition model.")
            print("Install with: pip install transformers")
            return

    # Create sample audio files for demonstration
    sample_files = create_sample_audio_files()

    # Demonstrate different features
    features = [
        (
            "Single File Transcription",
            lambda: demonstrate_single_file(
                pipeline, sample_files[0] if sample_files else None
            ),
        ),
        (
            "Batch Transcription",
            lambda: demonstrate_batch_transcription(pipeline, sample_files),
        ),
        ("Performance Benchmarking", lambda: demonstrate_benchmarking(pipeline)),
        (
            "Audio Enhancement",
            lambda: demonstrate_audio_enhancement(
                pipeline, sample_files[0] if sample_files else None
            ),
        ),
    ]

    print("\nAvailable demonstrations:")
    for i, (name, _) in enumerate(features, 1):
        print(f"{i}. {name}")
    print("0. Run all demonstrations")
    print("R. Real-time recording (requires microphone)")

    try:
        choice = input("\nSelect demonstration (1-4, 0, or R): ").strip().upper()

        if choice == "0":
            # Run all demonstrations
            for name, func in features:
                print(f"\n{'=' * 20} {name} {'=' * 20}")
                func()
        elif choice == "R":
            demonstrate_real_time_recording(pipeline)
        elif choice.isdigit() and 1 <= int(choice) <= len(features):
            name, func = features[int(choice) - 1]
            print(f"\n{'=' * 20} {name} {'=' * 20}")
            func()
        else:
            print("Invalid choice. Running single file demonstration.")
            demonstrate_single_file(pipeline, sample_files[0] if sample_files else None)

    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"Error during demonstration: {e}")


def demonstrate_single_file(pipeline: SpeechRecognitionPipeline, audio_file: str):
    """Demonstrate single file transcription."""
    if not audio_file or not Path(audio_file).exists():
        print("No audio file available for demonstration.")
        return

    print(f"Transcribing single audio file: {audio_file}")
    result = pipeline.transcribe_file(audio_file)

    if "error" not in result:
        print("\nDetailed Results:")
        print(f"Text: {result['text']}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
        print(f"Audio duration: {result['audio_duration']:.2f}s")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Real-time factor: {result['real_time_factor']:.2f}x")
        print(f"Device used: {result['device']}")


def demonstrate_batch_transcription(
    pipeline: SpeechRecognitionPipeline, sample_files: List[str]
):
    """Demonstrate batch transcription."""
    if not sample_files:
        print("No sample files available for batch transcription.")
        return

    print("Running batch transcription on sample files...")
    results = pipeline.batch_transcribe(sample_files)

    print("\nBatch Results Summary:")
    for i, result in enumerate(results, 1):
        if "error" not in result:
            print(
                f"{i}. {Path(result['file']).name}: '{result['text'][:50]}...' "
                if len(result["text"]) > 50
                else f"{i}. {Path(result['file']).name}: '{result['text']}'"
            )
        else:
            print(f"{i}. {Path(result['file']).name}: ERROR - {result['error']}")


def demonstrate_benchmarking(pipeline: SpeechRecognitionPipeline):
    """Demonstrate performance benchmarking."""
    print("Running performance benchmark...")

    # Create a longer test audio for benchmarking
    print("Generating test audio for benchmarking...")
    sr = 16000
    duration = 5  # 5 seconds
    t = np.linspace(0, duration, int(sr * duration))
    test_audio = 0.2 * np.sin(2 * np.pi * 440 * t)  # A4 tone

    # Save test audio
    test_file = "benchmark_audio.wav"
    sf.write(test_file, test_audio, sr)

    try:
        benchmark_results = pipeline.benchmark(test_file)

        print("\nBenchmark Results:")
        print(f"Average latency: {benchmark_results.get('avg_latency', 'N/A')} ms")
        print(f"Throughput: {benchmark_results.get('throughput', 'N/A')} FPS")
        print(f"Memory usage: {benchmark_results.get('memory_mb', 'N/A')} MB")

        # Performance analysis
        if "avg_latency" in benchmark_results:
            latency_ms = benchmark_results["avg_latency"]
            print("\nPerformance Analysis:")

            if latency_ms < 100:
                print("✓ Excellent real-time performance")
            elif latency_ms < 500:
                print("✓ Good real-time performance")
            elif latency_ms < 1000:
                print("⚠ Acceptable for batch processing")
            else:
                print("⚠ Consider optimization for real-time use")

    finally:
        # Cleanup
        if Path(test_file).exists():
            Path(test_file).unlink()


def demonstrate_audio_enhancement(pipeline: SpeechRecognitionPipeline, audio_file: str):
    """Demonstrate audio enhancement effects."""
    if not audio_file or not Path(audio_file).exists():
        print("No audio file available for enhancement demonstration.")
        return

    print("Comparing transcription with and without audio enhancement...")

    # Load audio
    audio_data, sr = pipeline.audio_processor.load_audio(audio_file)

    # Transcribe without enhancement
    print("\n1. Without enhancement:")
    result_raw = pipeline.transcribe(audio_data, sr, enhance_audio=False)
    print(f"   Text: {result_raw['text']}")
    print(f"   Processing time: {result_raw['processing_time']:.2f}s")

    # Transcribe with enhancement
    print("\n2. With enhancement:")
    result_enhanced = pipeline.transcribe(audio_data, sr, enhance_audio=True)
    print(f"   Text: {result_enhanced['text']}")
    print(f"   Processing time: {result_enhanced['processing_time']:.2f}s")

    # Compare results
    print("\nComparison:")
    if result_enhanced["text"] != result_raw["text"]:
        print("✓ Enhancement changed the transcription result")
    else:
        print("→ Enhancement did not change the transcription result")

    time_diff = result_enhanced["processing_time"] - result_raw["processing_time"]
    print(f"Processing time difference: {time_diff:+.2f}s")


def demonstrate_real_time_recording(pipeline: SpeechRecognitionPipeline):
    """Demonstrate real-time recording and transcription."""
    print("\nReal-time Speech Recognition")
    print("This feature requires a working microphone.")

    try:
        # Check if microphone is available
        print("Checking microphone availability...")
        devices = sd.query_devices()
        input_devices = [d for d in devices if d["max_input_channels"] > 0]

        if not input_devices:
            print("No microphone detected. Skipping real-time demonstration.")
            return

        print(f"Found {len(input_devices)} input device(s).")

        # Create real-time recorder
        recorder = RealTimeRecorder(pipeline, chunk_duration=3.0)

        print("\nStarting real-time recording...")
        print("Speak clearly into your microphone for 15 seconds.")
        print("Press Ctrl+C to stop early.")

        results = recorder.record_and_transcribe(duration=15.0)

        print("\nReal-time transcription completed!")
        print(f"Processed {len(results)} audio chunks.")

        if results:
            print("\nComplete transcription:")
            full_text = " ".join([r["text"] for r in results if r["text"]])
            print(f">> {full_text}")
        else:
            print("No speech detected during recording.")

    except Exception as e:
        print(f"Real-time recording failed: {e}")
        print(
            "Make sure you have a working microphone and the 'sounddevice' package installed."
        )


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        demonstrate_features()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"Program error: {e}")
        import traceback

        traceback.print_exc()

    print("\nThank you for trying OpenVINO-Easy speech recognition!")
