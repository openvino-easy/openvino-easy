#!/usr/bin/env python3
"""
Audio Speech Recognition Example with OpenVINO-Easy

This example demonstrates how to use OpenVINO-Easy for speech-to-text
using Whisper models.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import oe
import numpy as np
import tempfile
import wave
from pathlib import Path


def create_sample_audio(duration=3, sample_rate=16000, frequency=440):
    """Create a simple sine wave audio file for testing."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(frequency * 2 * np.pi * t) * 0.5

    # Convert to 16-bit integers
    audio_int16 = (audio_data * 32767).astype(np.int16)

    return audio_int16, sample_rate


def save_audio_to_wav(audio_data, sample_rate, filename):
    """Save audio data to a WAV file."""
    with wave.open(str(filename), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())


def main():
    """Execute comprehensive audio speech recognition demonstration."""
    print("OpenVINO-Easy Audio Speech Recognition Example")
    print("=" * 50)

    # Enumerate available compute devices
    compute_devices = oe.devices()
    print(f"Available compute devices: {compute_devices}")

    # Create a temporary audio file for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_audio = Path(temp_dir) / "test_audio.wav"

        print("\nGenerating synthetic audio sample...")
        audio_data, sample_rate = create_sample_audio()
        save_audio_to_wav(audio_data, sample_rate, temp_audio)
        print(f"Audio file created: {temp_audio}")

        # Initialize speech recognition model for evaluation
        try:
            print("\nInitializing Whisper speech recognition model...")
            # Deploy lightweight model variant for demonstration
            model_id = "openai/whisper-tiny"

            print(f"Loading model: {model_id}...")
            # Initialize model with optimal configuration
            oe.load(
                model_id,
                device_preference=["CPU"],  # CPU deployment for demonstration
                dtype="fp16",
            )

            print("✅ Model loaded successfully!")

            # Get model info (NEW API)
            info = oe.get_info()
            print("📊 Model info:")
            print(f"  - Device: {info['device']}")
            print(f"  - Precision: {info.get('dtype', 'unknown')}")
            print(f"  - Has tokenizer: {info.get('has_tokenizer', False)}")

            # Run inference on the audio file (NEW API)
            print(f"\n🎯 Running speech recognition on {temp_audio}...")
            result = oe.infer(str(temp_audio))

            print("✅ Speech recognition complete!")
            print(f"📝 Transcription: {result}")

            # Run benchmark (NEW API)
            print("\n⏱️ Running performance benchmark...")
            stats = oe.benchmark(warmup_runs=2, benchmark_runs=5)

            print("📊 Performance Results:")
            print(f"  - Average latency: {stats['avg_latency_ms']:.2f}ms")
            print(f"  - Throughput: {stats['throughput_fps']:.2f} inferences/sec")
            print(f"  - Device: {stats['device']}")

            # Clean up (NEW API)
            oe.unload()

        except Exception as e:
            print(f"❌ Error: {e}")
            print("\n💡 Tips:")
            print(
                "  - Make sure you have audio dependencies: pip install 'openvino-easy[audio]'"
            )
            print("  - Try a different model if this one isn't available")
            print("  - Check your internet connection for model download")

            # Show what the expected API would look like
            print("\n📖 Expected usage (NEW API):")
            print("  oe.load('openai/whisper-tiny')")
            print("  result = oe.infer('path/to/audio.wav')")
            print("  print(result)  # 'Hello, this is the transcribed text'")
            print("  oe.unload()  # Optional cleanup")


if __name__ == "__main__":
    main()
