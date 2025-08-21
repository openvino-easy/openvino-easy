#!/usr/bin/env python3
"""
Example demonstrating memory management with OpenVINO-Easy.

This example shows different approaches to managing model memory:
1. Explicit unload() method
2. Context manager (automatic cleanup)
3. Multiple model switching
4. Error handling after unload
"""

import sys
import os

# Add the parent directory to path to import oe
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Now import oe (OpenVINO-Easy)
# import oe  # This would be used in real scenarios


def demonstrate_explicit_unload():
    """Demonstrate explicit model unloading."""
    print("=== Explicit Unload Example ===")

    # This would normally load a real model
    # oe.load("microsoft/DialoGPT-medium")
    print("1. Model loaded")

    # Run inference
    # result = oe.infer("Hello, how are you?")
    print("2. Inference completed")

    # Explicitly unload to free memory
    # oe.unload()
    print("3. Model unloaded - memory freed")

    # Try to use after unload (would raise error)
    # try:
    #     oe.infer("This will fail")
    # except RuntimeError as e:
    #     print(f"4. Expected error: {e}")


def demonstrate_context_manager():
    """Demonstrate context manager for automatic cleanup."""
    print("\n=== Context Manager Example ===")

    # Use context manager for automatic cleanup
    # with oe.load("microsoft/DialoGPT-medium") as pipe:
    print("1. Model loaded in context manager")

    #     result = pipe.infer("Hello!")
    print("2. Inference completed")

    # Model is automatically unloaded when exiting the 'with' block
    print("3. Model automatically unloaded when exiting context")


def demonstrate_multiple_models():
    """Demonstrate switching between multiple models."""
    print("\n=== Multiple Models Example ===")

    # Load first model
    # oe.load("text-model")
    print("1. Text model loaded")

    # Use it
    # result1 = oe.infer("Process this text")
    print("2. Text inference completed")

    # Unload to free memory before loading large image model
    # oe.unload()
    print("3. Text model unloaded")

    # Load second model
    # oe.load("image-model")
    print("4. Image model loaded")

    # Use it
    # result2 = oe.infer(image_data)
    print("5. Image inference completed")

    # Clean up
    # oe.unload()
    print("6. Image model unloaded")


def demonstrate_safety_checks():
    """Demonstrate safety checks and error handling."""
    print("\n=== Safety Checks Example ===")

    # This would normally load a real model
    # oe.load("some-model")
    print("1. Model loaded")

    # Check if loaded
    # if oe.is_loaded():
    print("2. Model is loaded - safe to use")
    #     result = oe.infer(data)

    # Unload the model
    # oe.unload()
    print("3. Model unloaded")

    # Check again
    # if not oe.is_loaded():
    print("4. Model is no longer loaded")

    # Attempting inference now would raise a helpful error
    # try:
    #     oe.infer("This will fail")
    # except RuntimeError as e:
    #     print(f"5. Helpful error message: {e}")


if __name__ == "__main__":
    print("OpenVINO-Easy Memory Management Examples")
    print("=" * 50)

    demonstrate_explicit_unload()
    demonstrate_context_manager()
    demonstrate_multiple_models()
    demonstrate_safety_checks()

    print("\n" + "=" * 50)
    print("Summary of Memory Management Options:")
    print("1. oe.unload() - Explicit memory cleanup")
    print("2. with oe.load(...) as pipe: - Automatic cleanup")
    print("3. oe.is_loaded() - Check model status")
    print("4. Error handling - Safe after unload")
    print("\nRecommendation: Use context manager for most cases!")
    print("\nThe NEW TRUE 3-Function API:")
    print("- oe.load('model')   # Load model")
    print("- oe.infer(data)     # Run inference")
    print("- oe.benchmark()     # Measure performance")
