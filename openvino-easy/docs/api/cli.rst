Command Line Interface
======================

OpenVINO-Easy provides a comprehensive command-line interface for common inference tasks, benchmarking, and model management.

Overview
--------

The CLI offers three main commands:

- ``oe run`` - Run inference on text, images, or audio
- ``oe bench`` - Benchmark model performance  
- ``oe devices`` - List available hardware devices

All commands support device selection, performance tuning, and output formatting options.

Global Options
--------------

These options are available for all commands:

.. option:: --device DEVICE

   Target device for inference (CPU, GPU, NPU, or auto).
   
   **Default:** auto
   
   **Example:**
   
   .. code-block:: bash
   
      oe run --device GPU "microsoft/DialoGPT-medium" "Hello"

.. option:: --precision PRECISION

   Model precision (FP32, FP16, INT8).
   
   **Default:** FP32
   
   **Example:**
   
   .. code-block:: bash
   
      oe run --precision FP16 "model.onnx" "input text"

.. option:: --verbose, -v

   Enable verbose logging.
   
   **Example:**
   
   .. code-block:: bash
   
      oe run -v "model_name" "input"

.. option:: --quiet, -q

   Suppress non-essential output.

.. option:: --help, -h

   Show help message and exit.

oe run
------

Run inference with a model on provided inputs.

**Syntax:**

.. code-block:: bash

   oe run [OPTIONS] MODEL INPUT [INPUT ...]

**Arguments:**

.. option:: MODEL

   Model identifier or path. Can be:
   
   - HuggingFace model name: ``microsoft/DialoGPT-medium``
   - Local ONNX file: ``./models/model.onnx``
   - OpenVINO IR files: ``./models/model.xml``
   - Local directory: ``./my_model/``

.. option:: INPUT

   Input data for inference. Format depends on model type:
   
   - **Text models:** String input
   - **Image models:** Path to image file
   - **Audio models:** Path to audio file
   - **Multiple inputs:** Space-separated list

**Options:**

.. option:: --output FILE, -o FILE

   Save output to file instead of printing to stdout.
   
   **Example:**
   
   .. code-block:: bash
   
      oe run -o results.txt "model_name" "input text"

.. option:: --format FORMAT

   Output format (text, json, csv).
   
   **Default:** text
   
   **Example:**
   
   .. code-block:: bash
   
      oe run --format json "model_name" "input"

.. option:: --batch-size SIZE

   Process inputs in batches of specified size.
   
   **Default:** 1
   
   **Example:**
   
   .. code-block:: bash
   
      oe run --batch-size 4 "model" input1 input2 input3 input4

.. option:: --max-length LENGTH

   Maximum output length for text generation models.
   
   **Default:** 50
   
   **Example:**
   
   .. code-block:: bash
   
      oe run --max-length 100 "gpt2" "The future of AI"

.. option:: --temperature TEMP

   Sampling temperature for text generation (0.0-2.0).
   
   **Default:** 1.0
   
   **Example:**
   
   .. code-block:: bash
   
      oe run --temperature 0.7 "gpt2" "Once upon a time"

.. option:: --top-p PROB

   Top-p (nucleus) sampling parameter (0.0-1.0).
   
   **Example:**
   
   .. code-block:: bash
   
      oe run --top-p 0.9 "gpt2" "The weather today"

.. option:: --top-k K

   Top-k sampling parameter.
   
   **Example:**
   
   .. code-block:: bash
   
      oe run --top-k 50 "gpt2" "In the future"

.. option:: --no-sample

   Disable sampling (use greedy decoding).
   
   **Example:**
   
   .. code-block:: bash
   
      oe run --no-sample "gpt2" "Explain quantum computing"

**Examples:**

**Text Generation:**

.. code-block:: bash

   # Basic text generation
   oe run "microsoft/DialoGPT-medium" "Hello, how are you?"
   
   # With custom parameters
   oe run --max-length 100 --temperature 0.8 "gpt2" "The future of technology"
   
   # Multiple inputs
   oe run "model_name" "Input 1" "Input 2" "Input 3"
   
   # Save to file
   oe run -o responses.txt "chatbot_model" "What is AI?"

**Image Classification:**

.. code-block:: bash

   # Single image
   oe run "resnet50" "image.jpg"
   
   # Multiple images
   oe run "image_classifier" img1.jpg img2.jpg img3.jpg
   
   # JSON output
   oe run --format json "vision_model" "photo.png"

**Audio Processing:**

.. code-block:: bash

   # Speech recognition
   oe run "speech_recognition_model" "audio.wav"
   
   # Multiple audio files
   oe run "audio_model" file1.wav file2.mp3 file3.flac

oe bench
--------

Benchmark model performance on specified hardware.

**Syntax:**

.. code-block:: bash

   oe bench [OPTIONS] MODEL [INPUT]

**Arguments:**

.. option:: MODEL

   Model to benchmark (same format as ``oe run``).

.. option:: INPUT

   Sample input for benchmarking (optional, dummy data used if not provided).

**Options:**

.. option:: --iterations N, -n N

   Number of inference iterations.
   
   **Default:** 100
   
   **Example:**
   
   .. code-block:: bash
   
      oe bench -n 500 "microsoft/DialoGPT-medium"

.. option:: --warmup N

   Number of warmup iterations before measurement.
   
   **Default:** 10
   
   **Example:**
   
   .. code-block:: bash
   
      oe bench --warmup 20 "model_name"

.. option:: --batch-size SIZE

   Batch size for benchmarking.
   
   **Default:** 1
   
   **Example:**
   
   .. code-block:: bash
   
      oe bench --batch-size 8 "image_model"

.. option:: --compare-devices

   Benchmark on all available devices.
   
   **Example:**
   
   .. code-block:: bash
   
      oe bench --compare-devices "model_name"

.. option:: --compare-precision

   Benchmark different precision modes (FP32, FP16, INT8).
   
   **Example:**
   
   .. code-block:: bash
   
      oe bench --compare-precision --device GPU "model"

.. option:: --output FILE, -o FILE

   Save benchmark results to JSON file.
   
   **Example:**
   
   .. code-block:: bash
   
      oe bench -o benchmark_results.json "model_name"

.. option:: --profile

   Enable detailed performance profiling.
   
   **Example:**
   
   .. code-block:: bash
   
      oe bench --profile "model_name"

**Examples:**

**Basic Benchmarking:**

.. code-block:: bash

   # Basic benchmark
   oe bench "microsoft/DialoGPT-medium"
   
   # Extended benchmark
   oe bench -n 1000 --warmup 50 "model_name"
   
   # GPU benchmark
   oe bench --device GPU "image_classifier"

**Comparative Benchmarking:**

.. code-block:: bash

   # Compare all devices
   oe bench --compare-devices "universal_model"
   
   # Compare precisions on GPU
   oe bench --device GPU --compare-precision "large_model"
   
   # Full comparison with profiling
   oe bench --compare-devices --compare-precision --profile "model"

**Sample Output:**

.. code-block:: text

   Model: microsoft/DialoGPT-medium
   Device: GPU.0 (Intel Arc A770)
   Precision: FP16
   
   Performance Metrics:
   ==================
   Average Latency: 45.2ms
   Throughput: 22.1 FPS
   Memory Usage: 1,245MB
   
   Percentiles (ms):
   P50: 43.1
   P90: 48.7  
   P95: 52.3
   P99: 67.8
   
   Device Utilization:
   GPU: 78%
   Memory: 45% (1,245MB / 2,800MB)

oe devices
----------

List and inspect available inference devices.

**Syntax:**

.. code-block:: bash

   oe devices [OPTIONS]

**Options:**

.. option:: --detailed, -d

   Show detailed device information.
   
   **Example:**
   
   .. code-block:: bash
   
      oe devices --detailed

.. option:: --test

   Test inference capability on each device.
   
   **Example:**
   
   .. code-block:: bash
   
      oe devices --test

.. option:: --memory

   Show memory information for each device.
   
   **Example:**
   
   .. code-block:: bash
   
      oe devices --memory

.. option:: --capabilities

   Show device capabilities and supported features.
   
   **Example:**
   
   .. code-block:: bash
   
      oe devices --capabilities

**Examples:**

**Basic Device List:**

.. code-block:: bash

   oe devices

**Output:**

.. code-block:: text

   Available Devices:
   ==================
   CPU    - Intel Core i7-12700K (16 cores)
   GPU.0  - Intel Arc A770 (16GB)
   NPU    - Intel NPU (integrated)

**Detailed Information:**

.. code-block:: bash

   oe devices --detailed --memory

**Output:**

.. code-block:: text

   Device Details:
   ===============
   
   CPU (Intel Core i7-12700K):
     Cores: 16 (8P + 8E)
     Memory: 32GB DDR4-3200
     Capabilities: VNNI, AVX-512
     Status: Ready
   
   GPU.0 (Intel Arc A770):
     Memory: 16GB GDDR6
     Compute Units: 32 Xe-Cores
     Memory Bandwidth: 560GB/s
     Driver: 31.0.101.4502
     Status: Ready
     Memory Usage: 245MB / 16,384MB (1.5%)
   
   NPU (Intel NPU):
     Architecture: Intel AI Boost
     Memory: Shared system memory
     Power: Ultra-low power mode
     Status: Ready

**Device Testing:**

.. code-block:: bash

   oe devices --test

**Output:**

.. code-block:: text

   Testing Devices:
   ================
   CPU    ✓ Working (45ms latency)
   GPU.0  ✓ Working (12ms latency) 
   NPU    ✓ Working (8ms latency, 0.5W power)
   
   Recommended device for:
   - Text generation: GPU.0
   - Image processing: GPU.0  
   - Low-power inference: NPU

Configuration and Environment
-----------------------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~

.. envvar:: OE_CACHE_DIR

   Directory for model cache.
   
   **Default:** ``~/.cache/openvino_easy``
   
   **Example:**
   
   .. code-block:: bash
   
      export OE_CACHE_DIR="/tmp/oe_cache"
      oe run "model_name" "input"

.. envvar:: OE_DEFAULT_DEVICE

   Default device for inference.
   
   **Example:**
   
   .. code-block:: bash
   
      export OE_DEFAULT_DEVICE="GPU"
      oe run "model_name" "input"  # Uses GPU automatically

.. envvar:: OE_LOG_LEVEL

   Logging level (DEBUG, INFO, WARNING, ERROR).
   
   **Example:**
   
   .. code-block:: bash
   
      export OE_LOG_LEVEL="DEBUG"
      oe run "model_name" "input"

.. envvar:: OE_CPU_ONLY

   Force CPU-only inference.
   
   **Example:**
   
   .. code-block:: bash
   
      export OE_CPU_ONLY="1"
      oe devices  # Shows only CPU

Configuration Files
~~~~~~~~~~~~~~~~~~

OpenVINO-Easy looks for configuration in:

1. ``./openvino_easy.json`` (project-specific)
2. ``~/.config/openvino_easy/config.json`` (user-specific)
3. Environment variables (highest priority)

**Example config.json:**

.. code-block:: json

   {
     "default_device": "GPU",
     "default_precision": "FP16",
     "cache_dir": "/path/to/cache",
     "device_preferences": ["NPU", "GPU", "CPU"],
     "performance_mode": "high",
     "enable_memory_optimization": true
   }

Error Handling and Debugging
----------------------------

Common Exit Codes
~~~~~~~~~~~~~~~~~

- **0**: Success
- **1**: General error (model loading, inference failure)
- **2**: Invalid arguments or configuration
- **3**: Device not available
- **4**: File not found (model or input files)
- **5**: Memory error (insufficient memory)

Debug Mode
~~~~~~~~~~

Enable detailed debug output:

.. code-block:: bash

   # Environment variable
   export OE_LOG_LEVEL="DEBUG"
   oe run "model" "input"
   
   # Command flag
   oe run --verbose "model" "input"
   
   # Both for maximum detail
   OE_LOG_LEVEL="DEBUG" oe run -v "model" "input"

Error Examples
~~~~~~~~~~~~~

**Model not found:**

.. code-block:: bash

   $ oe run "nonexistent/model" "input"
   Error: Model 'nonexistent/model' not found
   Suggestion: Check model name or verify network connectivity

**Device not available:**

.. code-block:: bash

   $ oe run --device NPU "model" "input"  
   Error: Device 'NPU' not available
   Available devices: CPU, GPU.0
   Suggestion: Use --device auto or install NPU drivers

**Memory error:**

.. code-block:: bash

   $ oe run "very-large-model" "input"
   Error: Insufficient memory for model loading
   Suggestion: Use --precision FP16 or --device CPU

Integration Examples
-------------------

Shell Scripts
~~~~~~~~~~~~

.. code-block:: bash

   #!/bin/bash
   # Batch processing script
   
   MODEL="microsoft/DialoGPT-medium"
   INPUT_DIR="./prompts"
   OUTPUT_DIR="./responses"
   
   mkdir -p "$OUTPUT_DIR"
   
   for file in "$INPUT_DIR"/*.txt; do
       basename=$(basename "$file" .txt)
       echo "Processing $basename..."
       
       oe run "$MODEL" "$(cat "$file")" > "$OUTPUT_DIR/${basename}_response.txt"
       
       if [ $? -eq 0 ]; then
           echo "✓ Success: $basename"
       else
           echo "✗ Failed: $basename"
       fi
   done

Python Integration
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import subprocess
   import json
   
   def run_inference(model, input_text, device="auto"):
       """Run inference using CLI."""
       cmd = [
           "oe", "run",
           "--device", device,
           "--format", "json",
           model, input_text
       ]
       
       result = subprocess.run(cmd, capture_output=True, text=True)
       
       if result.returncode == 0:
           return json.loads(result.stdout)
       else:
           raise RuntimeError(f"Inference failed: {result.stderr}")
   
   # Usage
   response = run_inference("microsoft/DialoGPT-medium", "Hello!")

Docker Integration
~~~~~~~~~~~~~~~~~

.. code-block:: dockerfile

   FROM python:3.11-slim
   
   RUN pip install openvino-easy
   
   # Copy models and scripts
   COPY models/ /app/models/
   COPY process.sh /app/
   
   WORKDIR /app
   
   # Run inference on startup
   CMD ["oe", "run", "./models/my_model.onnx", "default input"]

**Docker usage:**

.. code-block:: bash

   # Build image
   docker build -t my-inference-app .
   
   # Run with custom input
   docker run my-inference-app oe run "model" "custom input"
   
   # Run with GPU support
   docker run --device /dev/dri my-inference-app oe run --device GPU "model" "input"

See Also
--------

- :doc:`pipeline` - Python API for programmatic access
- :doc:`../getting_started` - Basic usage examples
- :doc:`../performance_tuning` - Performance optimization
- :doc:`../production_deployment` - Production deployment patterns