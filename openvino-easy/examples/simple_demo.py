#!/usr/bin/env python3
"""
OpenVINO-Easy: TRUE 3-Function API Demo

This demonstrates the revolutionary simplicity of the new API.
"""


def main():
    print("OPENVINO-EASY: TRUE 3-FUNCTION API")
    print("Transform any AI model into 3 simple functions")
    print("=" * 50)

    print("\nBEFORE (Complex Object Management):")
    print("  pipe = oe.load('microsoft/DialoGPT-medium')")
    print("  result = pipe.infer('Hello, how are you?')")
    print("  stats = oe.benchmark(pipe)")
    print("  pipe.unload()")

    print("\nAFTER (Pure Function Calls):")
    print("  oe.load('microsoft/DialoGPT-medium')")
    print("  result = oe.infer('Hello, how are you?')")
    print("  stats = oe.benchmark()")
    print("  oe.unload()")

    print("\n" + "=" * 50)
    print("BENEFITS:")
    print("- No objects to manage")
    print("- No confusing namespaces")
    print("- True 3-function simplicity")
    print("- Matches original vision")
    print("- Everything through 'oe'")

    print("\nTHE API:")
    print("1. oe.load(model)    # Load any AI model")
    print("2. oe.infer(data)    # Run inference")
    print("3. oe.benchmark()    # Measure performance")
    print("4. oe.unload()       # Free memory (optional)")

    print("\nFROM 80+ LINES OF OPENVINO CODE TO 3 FUNCTIONS!")
    print("OpenVINO has never been easier.")


if __name__ == "__main__":
    main()
