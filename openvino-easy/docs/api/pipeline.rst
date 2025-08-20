Pipeline API Reference
======================

.. warning::
   **This documentation covers the legacy Pipeline class.** 
   
   **New code should use the 3-function API**: ``oe.load()``, ``oe.infer()``, ``oe.benchmark()``, ``oe.unload()``
   
   See :doc:`core` for the modern 3-function API documentation.

The Pipeline class is the legacy interface for running inference with OpenVINO-Easy. It provides a unified API for all model types and handles device management, preprocessing, and postprocessing automatically.

.. currentmodule:: oe

.. autoclass:: Pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The Pipeline class wraps OpenVINO models and provides a simplified interface for inference. It automatically handles:

- Device selection and optimization
- Input preprocessing based on model requirements
- Batch processing
- Output formatting
- Memory management
- Error handling with fallback strategies

Basic Usage
-----------

**Modern 3-Function API (Recommended):**

.. code-block:: python

   import oe
   
   # Modern approach - 3 functions
   oe.load("microsoft/DialoGPT-medium")
   result = oe.infer("Hello!")
   stats = oe.benchmark()
   oe.unload()

**Legacy Pipeline API (Still Supported):**

.. code-block:: python

   import oe
   
   # Load from HuggingFace
   pipeline = oe.load("microsoft/DialoGPT-medium")
   
   # Load local model
   pipeline = oe.load("path/to/model.onnx")
   
   # Specify device
   pipeline = oe.load("model_name", device="GPU")

**Modern Inference (Recommended):**

.. code-block:: python

   # Load model first
   oe.load("microsoft/DialoGPT-medium")
   
   # Single input
   result = oe.infer("Hello, how are you?")
   
   # Clean up
   oe.unload()

**Legacy Pipeline Inference:**

.. code-block:: python

   # Single input
   result = pipeline("Hello, how are you?")
   
   # Batch processing
   results = pipeline(["Hello", "How are you?", "What's new?"])
   
   # With parameters
   result = pipeline("Hello", max_length=50, temperature=0.8)

Class Reference
---------------

Constructor
~~~~~~~~~~~

.. py:class:: Pipeline(model_path, device="auto", **kwargs)

   Create a new inference pipeline.

   :param model_path: Path to model file or HuggingFace model identifier
   :type model_path: str
   :param device: Target device ("CPU", "GPU", "NPU", or "auto")
   :type device: str
   :param precision: Model precision ("FP32", "FP16", or "INT8")
   :type precision: str, optional
   :param optimize_memory: Enable memory optimization
   :type optimize_memory: bool, optional
   :param performance_mode: Performance setting ("low", "medium", "high")
   :type performance_mode: str, optional
   :param cache_compiled: Cache compiled model for faster loading
   :type cache_compiled: bool, optional
   
   **Example:**
   
   .. code-block:: python
   
      pipeline = oe.Pipeline(
          "microsoft/DialoGPT-medium",
          device="GPU",
          precision="FP16",
          optimize_memory=True,
          performance_mode="high"
      )

Methods
~~~~~~~

.. py:method:: Pipeline.__call__(inputs, **kwargs)

   Run inference on the provided inputs.

   :param inputs: Input data (text, image, audio, or list of inputs)
   :type inputs: str, numpy.ndarray, PIL.Image, list, or dict
   :param max_length: Maximum output length for text generation
   :type max_length: int, optional
   :param temperature: Sampling temperature (0.0 to 2.0)
   :type temperature: float, optional
   :param top_p: Nucleus sampling parameter
   :type top_p: float, optional
   :param top_k: Top-k sampling parameter
   :type top_k: int, optional
   :param do_sample: Enable sampling for text generation
   :type do_sample: bool, optional
   :returns: Model output (format depends on model type)
   :rtype: str, numpy.ndarray, dict, or list

   **Text Generation Example:**
   
   .. code-block:: python
   
      result = pipeline(
          "The future of AI is",
          max_length=100,
          temperature=0.7,
          top_p=0.9,
          do_sample=True
      )

   **Computer Vision Example:**
   
   .. code-block:: python
   
      import PIL.Image
      
      image = PIL.Image.open("image.jpg")
      result = pipeline(image)  # Returns classification or detection results

   **Audio Processing Example:**
   
   .. code-block:: python
   
      result = pipeline("audio.wav")  # Returns transcription or audio features

.. py:method:: Pipeline.preprocess(inputs)

   Preprocess inputs for the model.

   :param inputs: Raw input data
   :returns: Preprocessed inputs ready for model
   :rtype: dict

   This method is called automatically during inference but can be used manually for debugging or custom workflows.

.. py:method:: Pipeline.postprocess(outputs, inputs=None)

   Postprocess model outputs.

   :param outputs: Raw model outputs
   :param inputs: Original inputs (optional, used for context)
   :returns: Formatted outputs
   :rtype: str, numpy.ndarray, dict, or list

.. py:method:: Pipeline.benchmark(num_runs=100, warmup_runs=10)

   Benchmark the pipeline performance.

   :param num_runs: Number of inference runs for benchmarking
   :type num_runs: int
   :param warmup_runs: Number of warmup runs before benchmarking
   :type warmup_runs: int
   :returns: Benchmark results with timing statistics
   :rtype: dict

   **Example:**
   
   .. code-block:: python
   
      results = pipeline.benchmark(num_runs=50)
      print(f"Average latency: {results['avg_latency']:.2f}ms")
      print(f"Throughput: {results['throughput']:.1f} FPS")

.. py:method:: Pipeline.get_memory_usage()

   Get current memory usage information.

   :returns: Memory usage statistics
   :rtype: dict

   **Example:**
   
   .. code-block:: python
   
      memory = pipeline.get_memory_usage()
      print(f"GPU memory: {memory['gpu_used']:.1f}MB / {memory['gpu_total']:.1f}MB")

Properties
~~~~~~~~~~

.. py:attribute:: Pipeline.device

   The device currently used by the pipeline.
   
   :type: str

.. py:attribute:: Pipeline.model_type

   The detected model type (e.g., "text-generation", "image-classification").
   
   :type: str

.. py:attribute:: Pipeline.input_shapes

   Dictionary of input tensor shapes.
   
   :type: dict

.. py:attribute:: Pipeline.output_shapes

   Dictionary of output tensor shapes.
   
   :type: dict

.. py:attribute:: Pipeline.model_info

   Comprehensive model information including format, precision, and capabilities.
   
   :type: dict

   **Example:**
   
   .. code-block:: python
   
      info = pipeline.model_info
      print(f"Model format: {info['format']}")
      print(f"Precision: {info['precision']}")
      print(f"Parameters: {info['parameters']:,}")

Context Manager Support
~~~~~~~~~~~~~~~~~~~~~~

Pipeline supports context manager protocol for automatic resource cleanup:

.. code-block:: python

   # Modern approach (recommended)
   oe.load("microsoft/DialoGPT-medium")
   result = oe.infer("Hello!")
   oe.unload()  # Explicit cleanup
   
   # Legacy context manager (still works)
   with oe.load("microsoft/DialoGPT-medium") as pipeline:
       result = pipeline.infer("Hello!")
   # Resources automatically cleaned up

Error Handling
--------------

The Pipeline class provides comprehensive error handling with informative messages:

.. py:exception:: ModelLoadError

   Raised when model loading fails.

.. py:exception:: DeviceNotFoundError

   Raised when specified device is not available.

.. py:exception:: InferenceError

   Raised when inference fails.

.. py:exception:: PreprocessingError

   Raised when input preprocessing fails.

**Example Error Handling:**

.. code-block:: python

   import oe
   
   try:
       pipeline = oe.load("invalid/model", device="NPU")
   except oe.ModelLoadError as e:
       print(f"Failed to load model: {e}")
       print("Suggestion: Check model name or try a different device")
   except oe.DeviceNotFoundError as e:
       print(f"Device not available: {e}")
       # Fallback to CPU
       pipeline = oe.load("invalid/model", device="CPU")

Advanced Usage
--------------

Custom Preprocessing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   # Load pipeline
   pipeline = oe.load("image-classification-model")
   
   # Custom preprocessing
   image = load_image("image.jpg")
   preprocessed = pipeline.preprocess(image)
   
   # Modify preprocessed data if needed
   preprocessed['input'] = np.clip(preprocessed['input'], 0, 1)
   
   # Run inference with custom preprocessed data
   outputs = pipeline.model(preprocessed)
   results = pipeline.postprocess(outputs)

Batch Processing Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Efficient batch processing
   inputs = ["text1", "text2", "text3", "text4"]
   
   # Process in batches of 2
   batch_size = 2
   results = []
   
   for i in range(0, len(inputs), batch_size):
       batch = inputs[i:i + batch_size]
       batch_results = pipeline(batch)
       results.extend(batch_results)

Model Switching
~~~~~~~~~~~~~~

.. code-block:: python

   # Modern approach: load models sequentially
   # Text model
   oe.load("microsoft/DialoGPT-medium", device_preference=["CPU"])
   text_result = oe.infer("Hello!")
   oe.unload()
   
   # Vision model  
   oe.load("microsoft/resnet-50", device_preference=["GPU", "CPU"])
   image_result = oe.infer(image_data)
   oe.unload()
   
   # Legacy approach (still works)
   text_model = oe.load("microsoft/DialoGPT-medium", device="CPU")
   vision_model = oe.load("microsoft/resnet-50", device="GPU")
   
   text_result = text_model("Hello!")
   image_result = vision_model(image_data)

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # High-performance configuration
   pipeline = oe.load(
       "model_name",
       device="GPU",
       precision="FP16",
       optimize_memory=True,
       performance_mode="high",
       cache_compiled=True
   )
   
   # Warm up the model
   pipeline("warmup input")
   
   # Run benchmark
   perf = pipeline.benchmark()
   print(f"Optimized throughput: {perf['throughput']:.1f} FPS")

Integration Examples
-------------------

Flask Web Service
~~~~~~~~~~~~~~~~

.. code-block:: python

   from flask import Flask, request, jsonify
   import oe
   
   app = Flask(__name__)
   
   # Modern approach: Load model once at startup
   oe.load("microsoft/DialoGPT-medium")
   
   @app.route('/generate', methods=['POST'])
   def generate():
       try:
           prompt = request.json['prompt']
           result = oe.infer(prompt)
           return jsonify({'response': result})
       except Exception as e:
           return jsonify({'error': str(e)}), 500
   
   # Legacy approach (still works)
   model = oe.load("microsoft/DialoGPT-medium")
   
   @app.route('/generate_legacy', methods=['POST']) 
   def generate_legacy():
       try:
           prompt = request.json['prompt']
           result = model(prompt, max_length=50)
           return jsonify({'response': result})
       except Exception as e:
           return jsonify({'error': str(e)}), 500

Async Processing
~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   import concurrent.futures
   import oe
   
   # Thread-safe pipeline usage
   pipeline = oe.load("model_name")
   executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
   
   async def process_async(text):
       loop = asyncio.get_event_loop()
       result = await loop.run_in_executor(executor, pipeline, text)
       return result
   
   # Usage
   results = await asyncio.gather(
       process_async("text1"),
       process_async("text2"),
       process_async("text3")
   )

See Also
--------

- :doc:`../getting_started` - Basic usage examples
- :doc:`../performance_tuning` - Optimization guidelines  
- :doc:`runtime` - Runtime configuration
- :doc:`../examples/index` - Complete examples