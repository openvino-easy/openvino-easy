Core API Reference
==================

This section documents the core functions and classes of OpenVINO-Easy.

Main Functions
--------------

.. autofunction:: oe.load

   **Example Usage:**

   .. code-block:: python

      import oe
      
      # Basic loading (3-function API)
      oe.load("microsoft/DialoGPT-medium")
      result = oe.infer("Hello!")
      oe.unload()
      
      # With specific device preference
      oe.load(
          "runwayml/stable-diffusion-v1-5",
          device_preference=["NPU", "GPU", "CPU"],
          dtype="int8"
      )
      result = oe.infer("a beautiful landscape")
      stats = oe.benchmark()
      oe.unload()
      
      # With custom cache directory
      oe.load(
          "bert-base-uncased",
          cache_dir="./my_models",
          dtype="fp16"
      )
      result = oe.infer("classify this text")
      oe.unload()

.. autofunction:: oe.devices

   **Example Usage:**

   .. code-block:: python

      import oe
      
      # Get available devices
      available = oe.devices()
      print(f"Available devices: {available}")
      # Output: ['NPU', 'GPU', 'CPU']

.. autofunction:: oe.detect_best_device

   **Example Usage:**

   .. code-block:: python

      import oe
      
      # Get best available device
      best = oe.detect_best_device()
      print(f"Best device: {best}")
      # Output: 'NPU'

Pipeline Class (Legacy)
-----------------------

.. note::
   The Pipeline class is maintained for backward compatibility. 
   **New code should use the 3-function API**: ``oe.load()``, ``oe.infer()``, ``oe.benchmark()``, ``oe.unload()``.

.. autoclass:: oe.Pipeline
   :members:
   :undoc-members:
   :show-inheritance:

   **Key Methods:**

   .. automethod:: oe.Pipeline.infer

      **Parameters:**
      
      * **input_data** (*str, np.ndarray, Dict[str, Any]*): Input data for inference
      * **\\*\\*kwargs**: Additional inference parameters
      
      **Returns:**
      
      * **Any**: Model output (format depends on model type)
      
      **Examples:**
      
      .. code-block:: python
      
         # Load model first
         oe.load("microsoft/DialoGPT-medium")
         
         # Text input
         result = oe.infer("Hello, how are you?")
         
         # Numpy array input
         import numpy as np
         img_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
         result = oe.infer(img_data)
         
         # Dictionary input (multi-input models)
         inputs = {
             "input_ids": np.array([[101, 7592, 102]]),
             "attention_mask": np.array([[1, 1, 1]])
         }
         result = oe.infer(inputs)
         
         # Clean up
         oe.unload()

   .. automethod:: oe.Pipeline.benchmark

      **Parameters:**
      
      * **warmup_runs** (*int, optional*): Number of warmup runs (default: 5)
      * **benchmark_runs** (*int, optional*): Number of benchmark runs (default: 20)
      * **\\*\\*kwargs**: Additional benchmark parameters
      
      **Returns:**
      
      * **Dict[str, Any]**: Benchmark results with timing statistics
      
      **Examples:**
      
      .. code-block:: python
      
         # Load model first
         oe.load("microsoft/DialoGPT-medium")
         
         # Basic benchmarking
         stats = oe.benchmark()
         print(f"FPS: {stats['fps']}")
         print(f"Mean latency: {stats['mean_ms']}ms")
         
         # Extended benchmarking
         stats = oe.benchmark(
             warmup_runs=10,
             benchmark_runs=100
         )
         
         # Clean up
         oe.unload()

   .. automethod:: oe.Pipeline.get_info

      **Returns:**
      
      * **Dict[str, Any]**: Model and runtime information
      
      **Examples:**
      
      .. code-block:: python
      
         # Load model first
         oe.load("microsoft/DialoGPT-medium")
         
         info = oe.get_info()
         print(f"Device: {info['device']}")
         print(f"Model path: {info['model_path']}")
         print(f"Input shapes: {info['runtime_info']['input_info']}")
         
         # Clean up
         oe.unload()

Version Information
-------------------

.. autodata:: oe.__version__

   Current version of OpenVINO-Easy.

   **Example:**

   .. code-block:: python

      import oe
      print(f"OpenVINO-Easy version: {oe.__version__}")

Device Detection
----------------

The device detection system automatically selects the best available hardware:

**Priority Order:**
1. **NPU** (Intel AI Boost) - Highest performance for AI workloads
2. **GPU** (Intel Arc/Xe Graphics) - Good performance, broad compatibility  
3. **CPU** (Intel Core) - Universal fallback, always available

**Device Validation:**

OpenVINO-Easy validates that devices are not just detected but actually functional:

* **NPU**: Checks driver status, device name, and supported properties
* **GPU**: Verifies OpenCL support and Intel GPU drivers
* **CPU**: Always functional (used as final fallback)

**Example:**

.. code-block:: python

   import oe
   
   # Check device status
   devices = oe.devices()
   print(f"Functional devices: {devices}")
   
   # Get recommended device
   best = oe.detect_best_device()
   print(f"Recommended: {best}")
   
   # Load with specific preference
   oe.load(
       "model-name",
       device_preference=["NPU", "GPU", "CPU"]
   )
   info = oe.get_info()
   print(f"Actually using: {info['device']}")
   oe.unload()

Supported Model Formats
-----------------------

OpenVINO-Easy supports a wide range of model formats with automatic detection:

**Hugging Face Models:**
* Transformers (BERT, GPT, T5, etc.)
* Diffusers (Stable Diffusion, etc.)
* Vision models (ViT, ResNet, etc.)

**ONNX Models:**
* Text models with tokenizer metadata
* Vision models with preprocessing info
* Generic ONNX models

**OpenVINO IR:**
* Native .xml/.bin model files
* Optimized for Intel hardware

**TensorFlow:**
* SavedModel format
* Frozen graphs (.pb files)

**PyTorch:**
* Models with transformers config
* Native PyTorch checkpoints

**Model Type Detection:**

.. code-block:: python

   # OpenVINO-Easy automatically detects model type:
   oe.load("microsoft/DialoGPT-medium")      # → transformers_optimum
   oe.load("runwayml/stable-diffusion-v1-5") # → diffusers  
   oe.load("model.onnx")                     # → onnx
   oe.load("model.xml")                      # → openvino_ir
   
   # Use oe.infer() and oe.benchmark() after loading
   # Remember to call oe.unload() when done 