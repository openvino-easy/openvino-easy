#!/usr/bin/env python3
"""
Example Runner for OpenVINO-Easy

Interactive runner to easily execute different examples.
"""

import subprocess
import sys
from pathlib import Path
import argparse


def main():
    """Interactive example runner."""
    print("OpenVINO-Easy Examples Runner")
    print("=" * 40)

    examples = [
        {
            "name": "Text Generation & LLMs",
            "file": "text_generation_llm.py",
            "description": "Large language models, conversation, and text generation",
            "time": "~3-5 minutes",
        },
        {
            "name": "Computer Vision Pipeline",
            "file": "computer_vision_pipeline.py",
            "description": "Image classification, batch processing, real-time analysis",
            "time": "~2-4 minutes",
        },
        {
            "name": "Audio Processing Showcase",
            "file": "audio_processing_showcase.py",
            "description": "Speech recognition, text-to-speech, audio classification",
            "time": "~3-5 minutes",
        },
        {
            "name": "Production Deployment",
            "file": "production_deployment.py",
            "description": "Enterprise patterns, monitoring, error handling",
            "time": "~2-3 minutes",
        },
        {
            "name": "Multimodal AI Showcase",
            "file": "multimodal_ai_showcase.py",
            "description": "Vision-language models, creative AI, cross-modal search",
            "time": "~4-6 minutes",
        },
        {
            "name": "Basic Audio Recognition",
            "file": "audio_speech_recognition.py",
            "description": "Simple speech-to-text example",
            "time": "~1-2 minutes",
        },
    ]

    parser = argparse.ArgumentParser(description="Run OpenVINO-Easy examples")
    parser.add_argument("--example", type=int, help="Example number to run directly")
    parser.add_argument("--all", action="store_true", help="Run all examples")
    parser.add_argument("--list", action="store_true", help="List all examples")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable Examples:")
        for i, example in enumerate(examples, 1):
            print(f"\n{i}. {example['name']}")
            print(f"   File: {example['file']}")
            print(f"   Description: {example['description']}")
            print(f"   Estimated time: {example['time']}")
        return

    if args.example:
        if 1 <= args.example <= len(examples):
            example = examples[args.example - 1]
            run_example(example)
        else:
            print(f"Invalid example number. Use 1-{len(examples)}")
        return

    if args.all:
        print("\nRunning all examples...")
        for i, example in enumerate(examples, 1):
            print(f"\n{'=' * 50}")
            print(f"Running Example {i}/{len(examples)}: {example['name']}")
            print(f"{'=' * 50}")

            try:
                run_example(example)
                print(f"Example {i} completed successfully")
            except KeyboardInterrupt:
                print(f"\nExample {i} interrupted by user")
                break
            except Exception as e:
                print(f"Example {i} failed: {e}")
                continue

        print("\nAll examples completed!")
        return

    # Interactive mode
    while True:
        print("\nAvailable Examples:")
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example['name']} ({example['time']})")
            print(f"   {example['description']}")

        print("\n0. Exit")

        try:
            choice = input(f"\nSelect an example (0-{len(examples)}): ").strip()

            if choice == "0":
                print("Goodbye!")
                break

            choice_num = int(choice)
            if 1 <= choice_num <= len(examples):
                example = examples[choice_num - 1]
                print(f"\nRunning: {example['name']}")
                print(f"File: {example['file']}")
                print(f"Estimated time: {example['time']}")

                confirm = input("\nProceed? (y/N): ").strip().lower()
                if confirm in ["y", "yes"]:
                    run_example(example)
                    print("\nExample completed!")
                else:
                    print("Skipped")
            else:
                print(f"Invalid choice. Please enter 0-{len(examples)}")

        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_example(example):
    """Run a specific example."""
    example_path = Path(__file__).parent / example["file"]

    if not example_path.exists():
        raise FileNotFoundError(f"Example file not found: {example_path}")

    print(f"\nStarting {example['name']}...")
    print(f"Running: python {example_path}")
    print("-" * 50)

    # Run the example
    result = subprocess.run([sys.executable, str(example_path)], capture_output=False)

    if result.returncode != 0:
        raise RuntimeError(f"Example failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()
