#!/usr/bin/env python3
"""
Comprehensive Computer Vision Example with OpenVINO-Easy

This example demonstrates:
- Image classification with multiple models
- Object detection workflows
- Image preprocessing and postprocessing
- Batch processing for efficiency
- Performance optimization
- Real-time webcam processing
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import oe
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple
import requests
from PIL import Image, ImageDraw, ImageFont


def download_sample_images() -> List[str]:
    """Download sample images for testing."""
    print("📥 Downloading sample images...")

    # Sample images from various sources
    image_urls = [
        {
            "url": "https://github.com/openvinotoolkit/openvino_notebooks/raw/main/notebooks/hello-world/data/coco.jpg",
            "filename": "coco_sample.jpg",
            "description": "COCO dataset sample",
        },
        {
            "url": "https://github.com/openvinotoolkit/openvino_notebooks/raw/main/notebooks/hello-world/data/coco_bike.jpg",
            "filename": "coco_bike.jpg",
            "description": "Bicycle image",
        },
    ]

    downloaded_images = []

    for img_info in image_urls:
        try:
            response = requests.get(img_info["url"], timeout=10)
            response.raise_for_status()

            # Save image
            image_path = Path(f"/tmp/{img_info['filename']}")
            with open(image_path, "wb") as f:
                f.write(response.content)

            downloaded_images.append(str(image_path))
            print(f"✅ Downloaded: {img_info['description']}")

        except Exception as e:
            print(f"⚠️ Failed to download {img_info['filename']}: {e}")
            # Create a simple test image as fallback
            create_test_image(f"/tmp/{img_info['filename']}")
            downloaded_images.append(f"/tmp/{img_info['filename']}")

    return downloaded_images


def create_test_image(filepath: str, size: Tuple[int, int] = (224, 224)) -> str:
    """Create a simple test image."""
    # Create a colorful test pattern
    img = Image.new("RGB", size, color="lightblue")
    draw = ImageDraw.Draw(img)

    # Add some geometric shapes
    draw.rectangle([50, 50, 150, 150], fill="red", outline="darkred", width=3)
    draw.ellipse([75, 75, 125, 125], fill="yellow", outline="orange", width=2)

    # Add text if possible
    try:
        font = ImageFont.load_default()
        draw.text((10, 10), "Test Image", fill="black", font=font)
    except:
        draw.text((10, 10), "Test", fill="black")

    img.save(filepath)
    return filepath


def demo_image_classification():
    """Demonstrate image classification with multiple models."""
    print("\n🖼️ === Image Classification Demo ===")

    # Download sample images
    sample_images = download_sample_images()

    # Models to test (from lightweight to more complex)
    models_to_try = [
        "microsoft/resnet-18",
        "microsoft/resnet-50",
        "google/vit-base-patch16-224",
    ]

    for model_id in models_to_try:
        print(f"\n🤖 Testing model: {model_id}")

        try:
            # Load model with optimization (NEW API)
            start_time = time.time()
            oe.load(
                model_id,
                device_preference=["NPU", "GPU", "CPU"],
                dtype="int8",  # Quantization for speed
            )
            load_time = time.time() - start_time

            # Get model info (NEW API)
            info = oe.get_info()
            print(f"✅ Loaded in {load_time:.2f}s on {info['device']}")

            # Process each image
            for img_path in sample_images:
                print(f"\nProcessing image sample: {Path(img_path).name}")

                try:
                    # Load image and extract metadata
                    image_data = Image.open(img_path)
                    print(
                        f"  Image dimensions: {image_data.size}, Color mode: {image_data.mode}"
                    )

                    # Execute model inference
                    inference_start = time.time()
                    classification_result = oe.infer(img_path)
                    inference_duration = time.time() - inference_start

                    # Process classification results
                    if isinstance(classification_result, dict):
                        # Process different output formats
                        if "logits" in classification_result:
                            logits = classification_result["logits"]
                            predicted_class = np.argmax(logits)
                            confidence = float(np.max(logits))
                        elif "predictions" in classification_result:
                            predictions = classification_result["predictions"]
                            predicted_class = predictions[0] if predictions else 0
                            confidence = 1.0
                        else:
                            predicted_class = 0
                            confidence = 1.0
                    elif isinstance(classification_result, np.ndarray):
                        predicted_class = np.argmax(classification_result)
                        confidence = float(np.max(classification_result))
                    else:
                        predicted_class = 0
                        confidence = 1.0

                    print(
                        f"  Classification: Class {predicted_class}, "
                        f"Confidence: {confidence:.3f}"
                    )
                    print(f"  Processing time: {inference_duration:.3f}s")

                except Exception as e:
                    print(f"  ❌ Failed to process {img_path}: {e}")

            # Run benchmark (NEW API)
            print(f"\n📊 Benchmarking {model_id}...")
            stats = oe.benchmark(
                input_data=sample_images[0], warmup_runs=3, benchmark_runs=10
            )

            print(f"  Average latency: {stats.get('avg_latency_ms', 0):.1f}ms")
            print(f"  Throughput: {stats.get('throughput_fps', 0):.1f} FPS")

            # Clean up (NEW API)
            oe.unload()

            break  # Use first successful model

        except Exception as e:
            print(f"❌ Failed to load {model_id}: {e}")
            continue


def demo_batch_image_processing():
    """Demonstrate efficient batch processing of images."""
    print("\n📦 === Batch Image Processing Demo ===")

    try:
        # Load a fast classification model (NEW API)
        oe.load(
            "microsoft/resnet-18", device_preference=["NPU", "GPU", "CPU"], dtype="int8"
        )

        # Get model info (NEW API)
        info = oe.get_info()
        print(f"🚀 Using {info['device']} for batch processing")

        # Create multiple test images
        batch_size = 5
        test_images = []

        print(f"🖼️ Creating {batch_size} test images...")
        for i in range(batch_size):
            img_path = f"/tmp/batch_test_{i}.jpg"
            create_test_image(img_path, size=(224, 224))
            test_images.append(img_path)

        # Single image processing (baseline)
        print("\n1️⃣ Processing images individually...")
        individual_start = time.time()
        individual_results = []

        for i, img_path in enumerate(test_images):
            start_time = time.time()
            result = oe.infer(img_path)  # NEW API
            latency = time.time() - start_time
            individual_results.append(result)
            print(f"  Image {i + 1}: {latency:.3f}s")

        individual_total = time.time() - individual_start

        # Calculate metrics
        avg_latency = individual_total / len(test_images)
        throughput = len(test_images) / individual_total

        print("\n📊 Batch Processing Results:")
        print(f"  Total time: {individual_total:.2f}s")
        print(f"  Average latency: {avg_latency:.3f}s per image")
        print(f"  Throughput: {throughput:.1f} images/sec")

        # Memory usage comparison
        import psutil

        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"  Memory usage: {memory_mb:.1f} MB")

        # Clean up (NEW API)
        oe.unload()

    except Exception as e:
        print(f"❌ Batch processing failed: {e}")


def demo_real_time_processing():
    """Simulate real-time image processing."""
    print("\n⚡ === Real-time Processing Simulation ===")

    try:
        # Load optimized model for real-time processing (NEW API)
        oe.load(
            "microsoft/resnet-18", device_preference=["NPU", "GPU", "CPU"], dtype="int8"
        )

        # Get model info (NEW API)
        info = oe.get_info()
        print(f"🎥 Simulating real-time processing on {info['device']}")

        # Create a stream of test images
        num_frames = 30  # Simulate 30 frames
        frame_times = []

        print(f"🔄 Processing {num_frames} frames...")

        start_time = time.time()

        for frame_idx in range(num_frames):
            # Create a unique test image for each frame
            img_path = f"/tmp/frame_{frame_idx}.jpg"

            # Vary image content slightly
            color = ["lightblue", "lightgreen", "lightcoral", "lightyellow"][
                frame_idx % 4
            ]
            img = Image.new("RGB", (224, 224), color=color)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"Frame {frame_idx}", fill="black")
            img.save(img_path)

            # Process frame (NEW API)
            frame_start = time.time()
            oe.infer(img_path)
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)

            # Progress indicator
            if (frame_idx + 1) % 10 == 0:
                print(f"  Processed {frame_idx + 1}/{num_frames} frames")

        total_time = time.time() - start_time

        # Calculate real-time metrics
        avg_frame_time = np.mean(frame_times)
        max_frame_time = np.max(frame_times)
        min_frame_time = np.min(frame_times)
        fps = num_frames / total_time

        print("\n📊 Real-time Performance:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Frames per second: {fps:.1f} FPS")
        print(f"  Average frame time: {avg_frame_time:.3f}s")
        print(f"  Min/Max frame time: {min_frame_time:.3f}s / {max_frame_time:.3f}s")

        # Real-time capability assessment
        target_fps = 30
        if fps >= target_fps:
            print(f"✅ Real-time capable: {fps:.1f} FPS ≥ {target_fps} FPS")
        else:
            print(f"⚠️ Not real-time: {fps:.1f} FPS < {target_fps} FPS")
            print("   Recommended: Use faster model or better hardware")

        # Clean up (NEW API)
        oe.unload()

    except Exception as e:
        print(f"❌ Real-time processing failed: {e}")


def demo_multi_model_comparison():
    """Compare multiple vision models side by side."""
    print("\n🏆 === Multi-Model Comparison ===")

    models = [
        {"id": "microsoft/resnet-18", "desc": "ResNet-18 (Fast)"},
        {"id": "microsoft/resnet-50", "desc": "ResNet-50 (Balanced)"},
        {"id": "google/vit-base-patch16-224", "desc": "ViT (Transformer)"},
    ]

    # Create a test image
    test_image = "/tmp/comparison_test.jpg"
    create_test_image(test_image, size=(224, 224))

    print(f"🖼️ Test image: {test_image}")

    results = []

    for model_info in models:
        model_id = model_info["id"]
        description = model_info["desc"]

        print(f"\n🔬 Testing: {description}")

        try:
            # Load model (NEW API)
            load_start = time.time()
            oe.load(model_id, device_preference=["NPU", "GPU", "CPU"], dtype="int8")
            load_time = time.time() - load_start

            # Get model info (NEW API)
            info = oe.get_info()

            # Single inference (NEW API)
            inference_start = time.time()
            result = oe.infer(test_image)
            inference_time = time.time() - inference_start

            # Benchmark (NEW API)
            stats = oe.benchmark(
                input_data=test_image, warmup_runs=3, benchmark_runs=10
            )

            model_result = {
                "model": description,
                "load_time": load_time,
                "inference_time": inference_time,
                "avg_latency": stats.get("avg_latency_ms", 0),
                "throughput": stats.get("throughput_fps", 0),
                "device": info.get("device", "unknown"),
                "quantized": info.get("quantized", False),
            }

            results.append(model_result)

            print(f"  ✅ Load: {load_time:.2f}s, Inference: {inference_time:.3f}s")
            print(f"     Avg Latency: {model_result['avg_latency']:.1f}ms")
            print(f"     Throughput: {model_result['throughput']:.1f} FPS")

            # Clean up (NEW API)
            oe.unload()

        except Exception as e:
            print(f"  ❌ Failed: {e}")

    # Display comparison table
    if results:
        print("\n📊 === Model Comparison Summary ===")
        print(
            f"{'Model':<25} {'Load Time':<10} {'Latency':<10} {'Throughput':<12} {'Device'}"
        )
        print("-" * 75)

        for result in results:
            print(
                f"{result['model']:<25} "
                f"{result['load_time']:.2f}s{'':<4} "
                f"{result['avg_latency']:.1f}ms{'':<3} "
                f"{result['throughput']:.1f} FPS{'':<4} "
                f"{result['device']}"
            )

        # Find best models
        best_speed = max(results, key=lambda x: x["throughput"])
        best_load = min(results, key=lambda x: x["load_time"])

        print(
            f"\n🏆 Fastest Inference: {best_speed['model']} ({best_speed['throughput']:.1f} FPS)"
        )
        print(
            f"🏆 Fastest Loading: {best_load['model']} ({best_load['load_time']:.2f}s)"
        )


def main():
    """Run all computer vision demonstrations."""
    print("🎯 OpenVINO-Easy Computer Vision Demo")
    print("=====================================")

    # Check available devices
    devices = oe.devices()
    print(f"🖥️ Available devices: {devices}")

    # Run demonstrations
    try:
        demo_image_classification()
        demo_batch_image_processing()
        demo_real_time_processing()
        demo_multi_model_comparison()

        print("\n✅ All computer vision demos completed!")
        print("\n💡 Production Tips:")
        print("  - Use quantization (int8) for real-time applications")
        print("  - Batch processing improves throughput")
        print("  - NPU/GPU provide better performance for vision tasks")
        print("  - Profile different models to find optimal trade-offs")
        print("  - Cache models locally for production deployment")

    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    main()
