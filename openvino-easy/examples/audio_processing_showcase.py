#!/usr/bin/env python3
"""
Comprehensive Audio Processing Example with OpenVINO-Easy

This example demonstrates:
- Speech-to-text with Whisper models
- Text-to-speech synthesis
- Audio classification and analysis
- Audio preprocessing and postprocessing
- Real-time audio processing simulation
- Multi-language support
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import oe
import numpy as np
import time
import wave
import tempfile
from pathlib import Path


def create_test_audio_file(
    filepath: str,
    duration: float = 3.0,
    frequency: float = 440.0,
    sample_rate: int = 16000,
) -> str:
    """Create a test audio file with a sine wave."""
    print(f"🎵 Creating test audio: {duration}s at {frequency}Hz")

    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Create a more interesting audio pattern
    # Mix of frequencies to simulate speech-like content
    audio_data = (
        0.3 * np.sin(frequency * 2 * np.pi * t)  # Fundamental
        + 0.2 * np.sin(frequency * 3 * 2 * np.pi * t)  # Harmonic
        + 0.1 * np.sin(frequency * 5 * 2 * np.pi * t)  # Higher harmonic
        + 0.05 * np.random.randn(len(t))  # Noise
    )

    # Add envelope to make it more natural
    envelope = np.exp(-t * 0.5)  # Decay envelope
    audio_data *= envelope

    # Normalize and convert to int16
    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
    audio_int16 = (audio_data * 32767).astype(np.int16)

    # Save as WAV file
    with wave.open(filepath, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    return filepath


def create_speech_like_audio(filepath: str, duration: float = 5.0) -> str:
    """Create more speech-like test audio with varying frequencies."""
    print(f"🗣️ Creating speech-like audio: {duration}s")

    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Create speech-like formants (simplified)
    formant1 = 800  # Typical formant frequencies for vowels
    formant2 = 1200
    formant3 = 2500

    # Modulate frequencies over time to simulate speech
    frequency_modulation = 1 + 0.3 * np.sin(2 * np.pi * t)

    audio_data = (
        0.4 * np.sin(formant1 * frequency_modulation * 2 * np.pi * t)
        + 0.3 * np.sin(formant2 * frequency_modulation * 2 * np.pi * t)
        + 0.2 * np.sin(formant3 * frequency_modulation * 2 * np.pi * t)
        + 0.1 * np.random.randn(len(t))  # Background noise
    )

    # Add amplitude modulation to simulate syllables
    syllable_rate = 3  # syllables per second
    amplitude_modulation = 0.5 + 0.5 * np.abs(np.sin(syllable_rate * 2 * np.pi * t))
    audio_data *= amplitude_modulation

    # Apply envelope
    envelope = np.exp(-t * 0.2)
    audio_data *= envelope

    # Normalize and save
    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
    audio_int16 = (audio_data * 32767).astype(np.int16)

    with wave.open(filepath, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    return filepath


def demo_speech_to_text():
    """Demonstrate speech-to-text with different Whisper models."""
    print("\n🎤 === Speech-to-Text Demo ===")

    # Test different Whisper model sizes
    whisper_models = [
        "openai/whisper-tiny",
        "openai/whisper-base",
        "openai/whisper-small",
    ]

    # Create test audio files
    test_files = []
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create different types of test audio
        speech_file = temp_path / "speech_test.wav"
        create_speech_like_audio(str(speech_file), duration=3.0)
        test_files.append(str(speech_file))

        tone_file = temp_path / "tone_test.wav"
        create_test_audio_file(str(tone_file), duration=2.0, frequency=220)
        test_files.append(str(tone_file))

        for model_id in whisper_models:
            print(f"\n🤖 Testing model: {model_id}")

            try:
                # Load Whisper model (NEW API)
                start_time = time.time()
                oe.load(model_id, device_preference=["NPU", "GPU", "CPU"], dtype="int8")
                load_time = time.time() - start_time

                # Get model info (NEW API)
                info = oe.get_info()
                print(f"✅ Loaded in {load_time:.2f}s on {info['device']}")

                # Process each test file
                for audio_file in test_files:
                    file_name = Path(audio_file).name
                    print(f"\n🎵 Processing: {file_name}")

                    try:
                        # Get audio file info
                        with wave.open(audio_file, "rb") as wav:
                            duration = wav.getnframes() / wav.getframerate()
                            print(
                                f"  Duration: {duration:.1f}s, "
                                f"Sample Rate: {wav.getframerate()}Hz"
                            )

                        # Run speech recognition (NEW API)
                        start_time = time.time()
                        result = oe.infer(audio_file)
                        inference_time = time.time() - start_time

                        # Parse result
                        transcription = ""
                        if isinstance(result, dict):
                            transcription = result.get("text", str(result))
                        elif isinstance(result, str):
                            transcription = result
                        else:
                            transcription = str(result)

                        print(f"  🎯 Transcription: '{transcription}'")
                        print(f"  ⏱️ Processing time: {inference_time:.2f}s")

                        # Calculate real-time factor
                        rtf = inference_time / duration
                        print(
                            f"  📊 Real-time factor: {rtf:.2f}x "
                            f"({'faster' if rtf < 1 else 'slower'} than real-time)"
                        )

                    except Exception as e:
                        print(f"  ❌ Failed to process {file_name}: {e}")

                # Run benchmark (NEW API)
                print(f"\n📊 Benchmarking {model_id}...")
                stats = oe.benchmark(
                    input_data=test_files[0], warmup_runs=2, benchmark_runs=5
                )

                print(f"  Average latency: {stats.get('avg_latency_ms', 0):.0f}ms")
                print(
                    f"  Throughput: {stats.get('throughput_fps', 0):.2f} inferences/sec"
                )

                # Clean up (NEW API)
                oe.unload()

                break  # Use first successful model

            except Exception as e:
                print(f"❌ Failed to load {model_id}: {e}")
                continue


def demo_text_to_speech():
    """Demonstrate text-to-speech synthesis."""
    print("\n🔊 === Text-to-Speech Demo ===")

    # Test texts in different languages and styles
    test_texts = [
        "Hello world, this is a test of text to speech synthesis.",
        "The quick brown fox jumps over the lazy dog.",
        "OpenVINO makes AI inference fast and efficient.",
        "Artificial intelligence is transforming technology.",
    ]

    tts_models = [
        "microsoft/speecht5_tts",
        # Add more TTS models as they become available
    ]

    for model_id in tts_models:
        print(f"\n🎙️ Testing TTS model: {model_id}")

        try:
            # Load TTS model (NEW API)
            start_time = time.time()
            oe.load(
                model_id,
                device_preference=["CPU"],  # TTS often works best on CPU
                dtype="fp16",
            )
            load_time = time.time() - start_time

            # Get model info (NEW API)
            info = oe.get_info()
            print(f"✅ Loaded in {load_time:.2f}s on {info['device']}")

            # Process each test text
            for i, text in enumerate(test_texts):
                print(f"\n📝 Text {i + 1}: '{text[:50]}...'")

                try:
                    # Generate speech (NEW API)
                    start_time = time.time()
                    result = oe.infer(text)
                    synthesis_time = time.time() - start_time

                    # Parse result (usually audio array)
                    if isinstance(result, dict) and "audio" in result:
                        audio_data = result["audio"]
                    elif isinstance(result, np.ndarray):
                        audio_data = result
                    else:
                        print(f"  ⚠️ Unexpected result format: {type(result)}")
                        continue

                    # Calculate audio duration
                    if isinstance(audio_data, np.ndarray):
                        # Assume 16kHz sample rate for duration calculation
                        audio_duration = len(audio_data) / 16000
                        print(f"  🎵 Generated {audio_duration:.1f}s of audio")
                        print(f"  ⏱️ Synthesis time: {synthesis_time:.2f}s")

                        # Calculate real-time factor
                        rtf = (
                            synthesis_time / audio_duration
                            if audio_duration > 0
                            else float("inf")
                        )
                        print(f"  📊 Real-time factor: {rtf:.2f}x")

                        # Save audio sample (optional)
                        if i == 0:  # Save first sample
                            output_file = (
                                f"/tmp/tts_sample_{model_id.split('/')[-1]}.wav"
                            )
                            try:
                                # Normalize and convert to int16
                                audio_normalized = (
                                    audio_data / np.max(np.abs(audio_data)) * 0.8
                                )
                                audio_int16 = (audio_normalized * 32767).astype(
                                    np.int16
                                )

                                with wave.open(output_file, "wb") as wav_file:
                                    wav_file.setnchannels(1)
                                    wav_file.setsampwidth(2)
                                    wav_file.setframerate(16000)
                                    wav_file.writeframes(audio_int16.tobytes())

                                print(f"  💾 Saved sample: {output_file}")
                            except Exception as e:
                                print(f"  ⚠️ Failed to save audio: {e}")

                except Exception as e:
                    print(f"  ❌ Failed to synthesize: {e}")

            # Clean up (NEW API)
            oe.unload()

            break  # Use first successful model

        except Exception as e:
            print(f"❌ Failed to load {model_id}: {e}")
            continue


def demo_audio_classification():
    """Demonstrate audio classification and analysis."""
    print("\n🎵 === Audio Classification Demo ===")

    # Audio classification models to test
    classification_models = [
        "facebook/hubert-base-ls960",
        # Add more audio classification models
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create different types of audio for classification
        test_files = []

        # Music-like audio (higher frequency)
        music_file = temp_path / "music_test.wav"
        create_test_audio_file(str(music_file), duration=2.0, frequency=880)
        test_files.append(("Music-like", str(music_file)))

        # Speech-like audio
        speech_file = temp_path / "speech_test.wav"
        create_speech_like_audio(str(speech_file), duration=2.0)
        test_files.append(("Speech-like", str(speech_file)))

        # Low frequency audio
        bass_file = temp_path / "bass_test.wav"
        create_test_audio_file(str(bass_file), duration=2.0, frequency=110)
        test_files.append(("Bass-like", str(bass_file)))

        for model_id in classification_models:
            print(f"\n🔍 Testing classification model: {model_id}")

            try:
                # Load classification model (NEW API)
                oe.load(model_id, device_preference=["NPU", "GPU", "CPU"], dtype="int8")

                # Get model info (NEW API)
                info = oe.get_info()
                print(f"✅ Loaded on {info['device']}")

                # Classify each audio file
                for audio_type, audio_file in test_files:
                    print(f"\n🎵 Classifying: {audio_type}")

                    try:
                        start_time = time.time()
                        result = oe.infer(audio_file)  # NEW API
                        classification_time = time.time() - start_time

                        # Parse classification result
                        if isinstance(result, dict):
                            if "logits" in result:
                                logits = result["logits"]
                                predicted_class = np.argmax(logits)
                                confidence = float(np.max(logits))
                            else:
                                predicted_class = 0
                                confidence = 1.0
                        elif isinstance(result, np.ndarray):
                            predicted_class = np.argmax(result)
                            confidence = float(np.max(result))
                        else:
                            predicted_class = 0
                            confidence = 1.0

                        print(f"  🎯 Predicted class: {predicted_class}")
                        print(f"  📊 Confidence: {confidence:.3f}")
                        print(f"  ⏱️ Classification time: {classification_time:.3f}s")

                    except Exception as e:
                        print(f"  ❌ Classification failed: {e}")

                # Clean up (NEW API)
                oe.unload()

                break  # Use first successful model

            except Exception as e:
                print(f"❌ Failed to load {model_id}: {e}")
                continue


def demo_real_time_audio_processing():
    """Simulate real-time audio processing."""
    print("\n⚡ === Real-time Audio Processing Simulation ===")

    try:
        # Load a fast speech recognition model (NEW API)
        oe.load(
            "openai/whisper-tiny",  # Fastest Whisper model
            device_preference=["NPU", "GPU", "CPU"],
            dtype="int8",
        )

        # Get model info (NEW API)
        info = oe.get_info()
        print(f"🎥 Simulating real-time audio processing on {info['device']}")

        # Simulate audio streaming (1-second chunks)
        chunk_duration = 1.0  # seconds
        num_chunks = 10
        processing_times = []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            print(
                f"🔄 Processing {num_chunks} audio chunks of {chunk_duration}s each..."
            )

            for chunk_idx in range(num_chunks):
                # Create audio chunk
                chunk_file = temp_path / f"chunk_{chunk_idx}.wav"

                # Vary the audio content
                frequency = 200 + (chunk_idx * 50)  # Varying frequency
                create_test_audio_file(
                    str(chunk_file), duration=chunk_duration, frequency=frequency
                )

                # Process chunk (NEW API)
                start_time = time.time()
                result = oe.infer(str(chunk_file))
                processing_time = time.time() - start_time
                processing_times.append(processing_time)

                # Parse transcription
                transcription = ""
                if isinstance(result, dict):
                    transcription = result.get("text", "")
                elif isinstance(result, str):
                    transcription = result

                print(
                    f"  Chunk {chunk_idx + 1}: {processing_time:.3f}s -> '{transcription[:30]}...'"
                )

                # Check if processing is fast enough for real-time
                if processing_time > chunk_duration:
                    print(
                        f"    ⚠️ Slower than real-time ({processing_time:.2f}s > {chunk_duration}s)"
                    )

        # Calculate real-time performance metrics
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        min_processing_time = np.min(processing_times)

        real_time_factor = avg_processing_time / chunk_duration

        print("\n📊 Real-time Performance Summary:")
        print(f"  Average processing time: {avg_processing_time:.3f}s")
        print(
            f"  Min/Max processing time: {min_processing_time:.3f}s / {max_processing_time:.3f}s"
        )
        print(f"  Real-time factor: {real_time_factor:.2f}x")

        if real_time_factor < 1.0:
            print(
                f"  ✅ Real-time capable: {1 / real_time_factor:.1f}x faster than real-time"
            )
        else:
            print(f"  ⚠️ Not real-time: {real_time_factor:.1f}x slower than real-time")

        # Throughput calculation
        total_audio_duration = num_chunks * chunk_duration
        total_processing_time = sum(processing_times)
        throughput = total_audio_duration / total_processing_time

        print(f"  Audio throughput: {throughput:.1f}x real-time")

        # Clean up (NEW API)
        oe.unload()

    except Exception as e:
        print(f"❌ Real-time audio processing failed: {e}")


def main():
    """Run all audio processing demonstrations."""
    print("🎵 OpenVINO-Easy Audio Processing Demo")
    print("=====================================")

    # Check available devices
    devices = oe.devices()
    print(f"🖥️ Available devices: {devices}")

    # Check audio dependencies
    try:
        import wave

        print("✅ Audio processing dependencies available")
    except ImportError as e:
        print(f"⚠️ Audio dependencies missing: {e}")
        print("Install with: pip install 'openvino-easy[audio]'")
        return

    # Run demonstrations
    try:
        demo_speech_to_text()
        demo_text_to_speech()
        demo_audio_classification()
        demo_real_time_audio_processing()

        print("\n✅ All audio processing demos completed!")
        print("\n💡 Production Tips:")
        print("  - Use Whisper-tiny for real-time speech recognition")
        print("  - Process audio in chunks for streaming applications")
        print("  - NPU/GPU provide better performance for large audio models")
        print("  - Consider model quantization for deployment")
        print("  - Cache models locally for production use")
        print("  - Monitor real-time factor for streaming applications")

    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    main()
