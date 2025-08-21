#!/usr/bin/env python3
"""
Comprehensive Text Generation Example with OpenVINO-Easy

This example demonstrates:
- Loading various text generation models
- Interactive conversation with streaming
- Quantization for faster inference
- Device optimization
- Benchmarking and performance tuning
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import oe
import time


def demo_basic_text_generation():
    """Basic text generation with different models."""
    print("Basic Text Generation Demo")
    print("=" * 40)

    # List of models to try (from smallest to largest)
    models_to_try = [
        "microsoft/DialoGPT-small",  # Conversational model
        "gpt2",  # Classic GPT-2
        "distilgpt2",  # Faster, smaller variant
    ]

    for model_id in models_to_try:
        print(f"\nTesting model: {model_id}")
        try:
            # Load model with automatic device selection (NEW API)
            start_time = time.time()
            oe.load(
                model_id,
                dtype="int8",  # Use quantization for speed
                device_preference=["NPU", "GPU", "CPU"],
            )
            load_time = time.time() - start_time

            # Get model info (NEW API)
            info = oe.get_info()
            print(f"Loaded in {load_time:.2f}s on {info['device']}")

            # Test different prompts
            prompts = [
                "The future of artificial intelligence is",
                "In a world where technology advances rapidly,",
                "The most important skill for the 21st century is",
            ]

            for prompt in prompts:
                print(f"\nPrompt: '{prompt}'")

                start_time = time.time()
                result = oe.infer(prompt)  # NEW API
                inference_time = time.time() - start_time

                # Process inference results
                if isinstance(result, dict) and "generated_text" in result:
                    generated_text = result["generated_text"]
                elif isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", str(result[0]))
                else:
                    generated_text = str(result)

                print(
                    f"Generated text ({inference_time:.2f}s): {generated_text[:200]}..."
                )

            # Execute performance benchmarking
            print("\nPerformance analysis in progress...")
            benchmark_results = oe.benchmark()
            print(
                f"Average latency: {benchmark_results.get('avg_latency_ms', 'N/A')}ms"
            )
            print(
                f"Throughput: {benchmark_results.get('throughput_fps', 'N/A')} inferences/sec"
            )

            # Release model resources
            oe.unload()

        except Exception as e:
            print(f"Failed to load {model_id}: {e}")
            continue

        break  # Use first successful model


def demo_conversation_with_context():
    """Demonstrate interactive conversation capabilities with contextual memory management."""
    print("\nInteractive Conversation Demo")
    print("=" * 40)

    try:
        # Initialize conversational model with optimal configuration
        oe.load(
            "microsoft/DialoGPT-small",
            device_preference=["NPU", "GPU", "CPU"],
            dtype="int8",
        )

        deployment_info = oe.get_info()
        print(f"Conversational interface active on {deployment_info['device']}")
        print("Commands: 'quit' to exit, 'benchmark' to run performance analysis")

        conversation_context = []

        while True:
            user_input = input("\nUser: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                break
            elif user_input.lower() == "benchmark":
                print("\nExecuting performance benchmark...")
                performance_stats = oe.benchmark(warmup_runs=3, benchmark_runs=10)
                for metric, value in performance_stats.items():
                    print(f"  {metric}: {value:.3f}")
                continue

            if not user_input:
                continue

            # Construct contextual prompt from recent conversation history
            context_window = ""
            for turn in conversation_context[-3:]:  # Maintain last 3 conversation turns
                context_window += f"Human: {turn['human']}\nAI: {turn['ai']}\n"

            contextual_prompt = f"{context_window}Human: {user_input}\nAI:"

            try:
                start_time = time.time()
                inference_result = oe.infer(contextual_prompt)
                processing_time = time.time() - start_time

                # Parse model response
                if (
                    isinstance(inference_result, dict)
                    and "generated_text" in inference_result
                ):
                    ai_response = inference_result["generated_text"]
                elif isinstance(inference_result, list) and len(inference_result) > 0:
                    ai_response = inference_result[0].get(
                        "generated_text", str(inference_result[0])
                    )
                else:
                    ai_response = str(inference_result)

                # Extract response content excluding prompt
                if contextual_prompt in ai_response:
                    ai_response = ai_response.replace(contextual_prompt, "").strip()

                print(f"Assistant ({processing_time:.2f}s): {ai_response}")

                # Update conversation context
                conversation_context.append({"human": user_input, "ai": ai_response})

            except Exception as e:
                print(f"Response generation error: {e}")

        # Release conversational model resources
        oe.unload()

    except Exception as e:
        print(f"Conversation initialization failed: {e}")


def demo_performance_comparison():
    """Conduct comprehensive performance analysis across multiple model configurations."""
    print("\nPerformance Configuration Analysis")
    print("=" * 40)

    benchmark_model = "distilgpt2"  # Lightweight model for evaluation

    precision_configurations = [
        {"dtype": "fp32", "desc": "Full Precision (FP32)"},
        {"dtype": "fp16", "desc": "Half Precision (FP16)"},
        {"dtype": "int8", "desc": "INT8 Quantized"},
    ]

    available_devices = oe.devices()
    print(f"Available compute devices: {available_devices}")

    benchmark_results = []

    for config in precision_configurations:
        for device in available_devices:
            configuration_id = f"{config['desc']} on {device}"
            print(f"\nEvaluating configuration: {configuration_id}")

            try:
                # Load model with configuration (NEW API)
                start_time = time.time()
                oe.load(
                    benchmark_model, device_preference=[device], dtype=config["dtype"]
                )
                load_time = time.time() - start_time

                # Get model info (NEW API)
                info = oe.get_info()

                # Run benchmark (NEW API)
                stats = oe.benchmark(warmup_runs=3, benchmark_runs=10)

                result = {
                    "config": configuration_id,
                    "load_time": load_time,
                    "avg_latency": stats.get("avg_latency_ms", 0),
                    "throughput": stats.get("throughput_token_per_sec", 0),
                    "device": info.get("device", "unknown"),
                    "quantized": info.get("quantized", False),
                }
                benchmark_results.append(result)

                print(
                    f"Load: {load_time:.2f}s, Latency: {result['avg_latency']:.1f}ms, "
                    f"Throughput: {result['throughput']:.1f} tokens/sec"
                )

                # Clean up (NEW API)
                oe.unload()

            except Exception as e:
                print(f"Failed: {e}")
                continue

    # Display comparison table
    print("\nPerformance Summary")
    print("=" * 70)
    print(f"{'Configuration':<25} {'Load Time':<10} {'Latency':<12} {'Throughput':<15}")
    print("-" * 70)

    for result in benchmark_results:
        print(
            f"{result['config']:<25} "
            f"{result['load_time']:.2f}s{'':<4} "
            f"{result['avg_latency']:.1f}ms{'':<4} "
            f"{result['throughput']:.1f} tok/s"
        )

    # Find best configuration
    if benchmark_results:
        best_throughput = max(benchmark_results, key=lambda x: x["throughput"])
        best_latency = min(benchmark_results, key=lambda x: x["avg_latency"])

        print(
            f"\nBest Throughput: {best_throughput['config']} "
            f"({best_throughput['throughput']:.1f} tokens/sec)"
        )
        print(
            f"Best Latency: {best_latency['config']} "
            f"({best_latency['avg_latency']:.1f}ms)"
        )


def demo_batch_processing():
    """Demonstrate batch processing for efficiency."""
    print("\nBatch Processing Demo")
    print("=" * 40)

    try:
        # Load model (NEW API)
        oe.load("distilgpt2", device_preference=["NPU", "GPU", "CPU"], dtype="int8")

        # Prepare batch of prompts
        prompts = [
            "The benefits of renewable energy include",
            "Machine learning algorithms can help",
            "The future of space exploration is",
            "Climate change solutions require",
            "Artificial intelligence will transform",
        ]

        print(f"Processing {len(prompts)} prompts...")

        # Process individually (baseline)
        print("\nIndividual Processing:")
        individual_start = time.time()
        individual_results = []

        for i, prompt in enumerate(prompts):
            start_time = time.time()
            result = oe.infer(prompt)  # NEW API
            latency = time.time() - start_time
            individual_results.append(result)
            print(f"  Prompt {i + 1}: {latency:.3f}s")

        individual_total = time.time() - individual_start

        # Calculate efficiency metrics
        avg_individual_latency = individual_total / len(prompts)

        print("\nResults:")
        print(f"  Total time: {individual_total:.2f}s")
        print(f"  Average per prompt: {avg_individual_latency:.2f}s")
        print(f"  Throughput: {len(prompts) / individual_total:.1f} prompts/sec")

        # Show sample results
        print("\nSample generations:")
        for i, (prompt, result) in enumerate(zip(prompts[:2], individual_results[:2])):
            if isinstance(result, dict) and "generated_text" in result:
                text = result["generated_text"]
            elif isinstance(result, list) and len(result) > 0:
                text = result[0].get("generated_text", str(result[0]))
            else:
                text = str(result)

            print(f"  {i + 1}. '{prompt}' → '{text[:100]}...'")

        # Clean up (NEW API)
        oe.unload()

    except Exception as e:
        print(f"Batch processing failed: {e}")


def main():
    """Run all text generation demonstrations."""
    print("OpenVINO-Easy Text Generation Demo")
    print("=" * 40)

    # Check available devices
    devices = oe.devices()
    print(f"Available devices: {devices}")

    # Run demonstrations
    try:
        demo_basic_text_generation()
        demo_conversation_with_context()
        demo_performance_comparison()
        demo_batch_processing()

        print("\nAll demos completed successfully!")
        print("\nTips for production use:")
        print("  - Use quantization (int8) for faster inference")
        print("  - Choose NPU/GPU for better performance when available")
        print("  - Cache models locally for faster startup")
        print("  - Monitor performance with benchmark() method")

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")


if __name__ == "__main__":
    main()
