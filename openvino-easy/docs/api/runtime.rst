Runtime API Reference
=====================

The Runtime module provides low-level access to OpenVINO functionality and configuration options. It handles device management, model compilation, and execution optimization.

.. currentmodule:: oe.runtime

Overview
--------

The Runtime module contains classes and functions for:

- Device detection and management
- Model compilation and optimization
- Memory management
- Performance profiling
- Custom inference workflows

This module is primarily for advanced users who need fine-grained control over the inference process.

Classes
-------

RuntimeWrapper
~~~~~~~~~~~~~~

.. py:class:: RuntimeWrapper(core=None)

   Low-level wrapper around OpenVINO Core functionality.

   :param core: OpenVINO Core instance (created automatically if None)
   :type core: openvino.runtime.Core, optional

   **Example:**
   
   .. code-block:: python
   
      from oe.runtime import RuntimeWrapper
      
      runtime = RuntimeWrapper()
      devices = runtime.get_available_devices()
      print(f"Available devices: {devices}")

   .. py:method:: get_available_devices()

      Get list of available inference devices.

      :returns: List of device names
      :rtype: list[str]

      **Example:**
      
      .. code-block:: python
      
         devices = runtime.get_available_devices()
         # Returns: ['CPU', 'GPU.0', 'NPU']

   .. py:method:: load_model(model_path, device="CPU", config=None)

      Load and compile a model for inference.

      :param model_path: Path to model file
      :type model_path: str
      :param device: Target device name
      :type device: str
      :param config: Device-specific configuration
      :type config: dict, optional
      :returns: Compiled model instance
      :rtype: openvino.runtime.CompiledModel

      **Example:**
      
      .. code-block:: python
      
         # Basic loading
         model = runtime.load_model("model.xml", device="GPU")
         
         # With device configuration
         config = {"PERFORMANCE_HINT": "THROUGHPUT"}
         model = runtime.load_model("model.xml", device="GPU", config=config)

   .. py:method:: create_infer_request(compiled_model)

      Create inference request from compiled model.

      :param compiled_model: Compiled OpenVINO model
      :type compiled_model: openvino.runtime.CompiledModel
      :returns: Inference request object
      :rtype: openvino.runtime.InferRequest

   .. py:method:: preprocess_inputs(inputs, input_info)

      Preprocess inputs according to model requirements.

      :param inputs: Raw input data
      :type inputs: dict, numpy.ndarray, str, or PIL.Image
      :param input_info: Model input information
      :type input_info: dict
      :returns: Preprocessed inputs ready for inference
      :rtype: dict

      **Text Input Example:**
      
      .. code-block:: python
      
         # Text preprocessing
         input_info = {
             'input_ids': {'shape': [1, -1], 'dtype': 'int64'}
         }
         
         processed = runtime.preprocess_inputs(
             "Hello world", 
             input_info
         )

      **Image Input Example:**
      
      .. code-block:: python
      
         import PIL.Image
         
         # Image preprocessing  
         input_info = {
             'image': {'shape': [1, 3, 224, 224], 'dtype': 'float32'}
         }
         
         image = PIL.Image.open("image.jpg")
         processed = runtime.preprocess_inputs(image, input_info)

   .. py:method:: run_inference(infer_request, inputs)

      Execute inference with preprocessed inputs.

      :param infer_request: OpenVINO inference request
      :type infer_request: openvino.runtime.InferRequest
      :param inputs: Preprocessed input tensors
      :type inputs: dict
      :returns: Raw model outputs
      :rtype: dict

   .. py:method:: get_device_info(device_name)

      Get detailed information about a specific device.

      :param device_name: Name of the device
      :type device_name: str
      :returns: Device capabilities and properties
      :rtype: dict

      **Example:**
      
      .. code-block:: python
      
         gpu_info = runtime.get_device_info("GPU.0")
         print(f"GPU name: {gpu_info['FULL_DEVICE_NAME']}")
         print(f"Memory: {gpu_info['GPU_DEVICE_TOTAL_MEM_SIZE']}")

   .. py:method:: optimize_for_device(model, device, performance_hint="THROUGHPUT")

      Apply device-specific optimizations to model.

      :param model: OpenVINO model
      :type model: openvino.runtime.Model
      :param device: Target device name
      :type device: str
      :param performance_hint: Optimization hint
      :type performance_hint: str
      :returns: Optimized model
      :rtype: openvino.runtime.Model

Device Management
-----------------

.. py:function:: detect_best_device(preference=None, model_type=None)

   Automatically detect the best available device for inference.

   :param preference: Preferred device types in order
   :type preference: list[str], optional
   :param model_type: Type of model for device optimization
   :type model_type: str, optional
   :returns: Best available device name
   :rtype: str

   **Example:**
   
   .. code-block:: python
   
      from oe.runtime import detect_best_device
      
      # Auto-detect best device
      device = detect_best_device()
      
      # With preference order
      device = detect_best_device(preference=["NPU", "GPU", "CPU"])
      
      # For specific model type
      device = detect_best_device(model_type="text-generation")

.. py:function:: list_devices()

   List all available inference devices with details.

   :returns: Dictionary of device names and capabilities
   :rtype: dict

   **Example:**
   
   .. code-block:: python
   
      devices = list_devices()
      for name, info in devices.items():
          print(f"{name}: {info['type']} - {info['capabilities']}")

.. py:function:: get_device_memory(device_name)

   Get memory information for a specific device.

   :param device_name: Name of the device
   :type device_name: str
   :returns: Memory usage statistics
   :rtype: dict

   **Example:**
   
   .. code-block:: python
   
      memory = get_device_memory("GPU.0")
      print(f"Used: {memory['used_mb']:.1f}MB")
      print(f"Free: {memory['free_mb']:.1f}MB")
      print(f"Total: {memory['total_mb']:.1f}MB")

Model Compilation
-----------------

.. py:function:: compile_model(model_path, device, config=None, cache_dir=None)

   Compile a model for optimal performance on target device.

   :param model_path: Path to model file
   :type model_path: str
   :param device: Target device name  
   :type device: str
   :param config: Compilation configuration
   :type config: dict, optional
   :param cache_dir: Directory for caching compiled models
   :type cache_dir: str, optional
   :returns: Compiled model ready for inference
   :rtype: openvino.runtime.CompiledModel

   **Configuration Options:**
   
   .. code-block:: python
   
      config = {
          # Performance hints
          "PERFORMANCE_HINT": "THROUGHPUT",  # or "LATENCY"
          "PERFORMANCE_HINT_NUM_REQUESTS": 4,
          
          # Memory optimization
          "OPTIMIZE_FOR_MEMORY": True,
          
          # Precision settings
          "INFERENCE_PRECISION_HINT": "f16",  # or "f32"
          
          # Device-specific options
          "GPU_ENABLE_LOOP_UNROLLING": False,
          "CPU_THREADS_NUM": 4,
      }
      
      model = compile_model("model.xml", "GPU", config=config)

.. py:function:: get_model_info(model_path)

   Extract detailed information from a model file.

   :param model_path: Path to model file
   :type model_path: str
   :returns: Model metadata and structure information
   :rtype: dict

   **Example:**
   
   .. code-block:: python
   
      info = get_model_info("model.onnx")
      print(f"Input shapes: {info['inputs']}")
      print(f"Output shapes: {info['outputs']}")
      print(f"Model type: {info['model_type']}")
      print(f"Parameter count: {info['parameters']:,}")

Performance Optimization
------------------------

.. py:class:: PerformanceProfiler(compiled_model, device)

   Performance profiling and optimization utilities.

   :param compiled_model: Compiled OpenVINO model
   :type compiled_model: openvino.runtime.CompiledModel
   :param device: Target device name
   :type device: str

   .. py:method:: benchmark(num_iterations=100, warmup_iterations=10)

      Run comprehensive performance benchmark.

      :param num_iterations: Number of inference iterations
      :type num_iterations: int
      :param warmup_iterations: Warmup iterations before measurement
      :type warmup_iterations: int
      :returns: Detailed performance metrics
      :rtype: dict

      **Example:**
      
      .. code-block:: python
      
         profiler = PerformanceProfiler(compiled_model, "GPU")
         
         results = profiler.benchmark(num_iterations=50)
         print(f"Average latency: {results['latency_ms']:.2f}ms")
         print(f"Throughput: {results['fps']:.1f} FPS")
         print(f"Memory usage: {results['memory_mb']:.1f}MB")

   .. py:method:: profile_inference(inputs, iterations=10)

      Profile individual inference steps.

      :param inputs: Input data for profiling
      :param iterations: Number of profiling iterations
      :type iterations: int
      :returns: Detailed timing breakdown
      :rtype: dict

   .. py:method:: optimize_batch_size(inputs, max_batch_size=32)

      Find optimal batch size for throughput.

      :param inputs: Sample input data
      :param max_batch_size: Maximum batch size to test
      :type max_batch_size: int
      :returns: Optimal batch size and performance metrics
      :rtype: dict

.. py:function:: optimize_threading(device, model_type=None)

   Get optimal threading configuration for device and model.

   :param device: Target device name
   :type device: str
   :param model_type: Type of model being optimized
   :type model_type: str, optional
   :returns: Recommended threading settings
   :rtype: dict

   **Example:**
   
   .. code-block:: python
   
      threading = optimize_threading("CPU", model_type="text-generation")
      print(f"Recommended CPU threads: {threading['cpu_threads']}")
      print(f"Streams: {threading['streams']}")

Memory Management
-----------------

.. py:class:: MemoryManager()

   Memory optimization and monitoring utilities.

   .. py:method:: get_memory_usage()

      Get current memory usage across all devices.

      :returns: Memory usage statistics
      :rtype: dict

   .. py:method:: clear_cache(device=None)

      Clear model compilation cache.

      :param device: Specific device to clear (all devices if None)
      :type device: str, optional

   .. py:method:: optimize_memory_usage(models, device)

      Optimize memory usage for multiple models.

      :param models: List of compiled models
      :type models: list
      :param device: Target device
      :type device: str
      :returns: Memory optimization report
      :rtype: dict

.. py:function:: monitor_memory(callback=None, interval=1.0)

   Monitor memory usage in real-time.

   :param callback: Function called with memory stats
   :type callback: callable, optional
   :param interval: Monitoring interval in seconds
   :type interval: float
   :returns: Memory monitoring context manager
   :rtype: contextmanager

   **Example:**
   
   .. code-block:: python
   
      def memory_callback(stats):
          if stats['gpu_usage_percent'] > 90:
              print("Warning: High GPU memory usage!")
      
      with monitor_memory(callback=memory_callback, interval=0.5):
          # Run inference
          results = model(inputs)

Error Handling
--------------

.. py:exception:: RuntimeError

   Base class for runtime-related errors.

.. py:exception:: DeviceError

   Raised when device operations fail.

.. py:exception:: CompilationError

   Raised when model compilation fails.

.. py:exception:: InferenceError

   Raised when inference execution fails.

Advanced Usage Examples
-----------------------

Custom Inference Loop
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from oe.runtime import RuntimeWrapper
   
   # Create runtime
   runtime = RuntimeWrapper()
   
   # Load and compile model
   compiled_model = runtime.load_model("model.xml", device="GPU")
   
   # Create inference request
   infer_request = runtime.create_infer_request(compiled_model)
   
   # Get model input info
   input_info = {
       name: {"shape": input.shape, "dtype": str(input.dtype)}
       for name, input in compiled_model.inputs.items()
   }
   
   # Custom inference loop
   for batch in data_batches:
       # Preprocess
       inputs = runtime.preprocess_inputs(batch, input_info)
       
       # Run inference
       outputs = runtime.run_inference(infer_request, inputs)
       
       # Process outputs
       results = process_outputs(outputs)

Multi-Model Pipeline
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from oe.runtime import RuntimeWrapper, PerformanceProfiler
   
   runtime = RuntimeWrapper()
   
   # Load multiple models
   detector = runtime.load_model("detector.xml", "GPU")
   classifier = runtime.load_model("classifier.xml", "GPU")
   
   # Create inference requests
   detector_req = runtime.create_infer_request(detector)
   classifier_req = runtime.create_infer_request(classifier)
   
   def multi_stage_inference(image):
       # Stage 1: Object detection
       detection_inputs = runtime.preprocess_inputs(image, detector_info)
       detections = runtime.run_inference(detector_req, detection_inputs)
       
       # Stage 2: Classification of detected objects
       results = []
       for bbox in extract_bboxes(detections):
           crop = crop_image(image, bbox)
           class_inputs = runtime.preprocess_inputs(crop, classifier_info)
           classification = runtime.run_inference(classifier_req, class_inputs)
           results.append(classification)
       
       return results

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from oe.runtime import PerformanceProfiler, monitor_memory
   import time
   
   # Setup profiler
   profiler = PerformanceProfiler(compiled_model, "GPU")
   
   # Monitor performance
   def performance_callback(stats):
       print(f"GPU Usage: {stats['gpu_usage_percent']:.1f}%")
       print(f"Memory: {stats['memory_used_mb']:.1f}MB")
   
   with monitor_memory(callback=performance_callback):
       # Benchmark model
       benchmark_results = profiler.benchmark(num_iterations=100)
       
       # Find optimal batch size
       optimal_batch = profiler.optimize_batch_size(sample_inputs)
       
       print(f"Best batch size: {optimal_batch['batch_size']}")
       print(f"Throughput: {optimal_batch['throughput']:.1f} FPS")

Device-Specific Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from oe.runtime import detect_best_device, optimize_threading
   
   # Detect best device for specific model type
   device = detect_best_device(
       preference=["NPU", "GPU", "CPU"],
       model_type="image-classification"
   )
   
   # Get optimal threading configuration
   threading_config = optimize_threading(device, "image-classification")
   
   # Apply device-specific optimizations
   config = {
       "PERFORMANCE_HINT": "THROUGHPUT",
       "CPU_THREADS_NUM": threading_config['cpu_threads'],
       "GPU_THROUGHPUT_STREAMS": threading_config['streams'],
   }
   
   # Compile with optimized configuration
   optimized_model = runtime.load_model(
       "model.xml", 
       device=device, 
       config=config
   )

See Also
--------

- :doc:`pipeline` - High-level Pipeline API
- :doc:`../performance_tuning` - Performance optimization guide
- :doc:`../production_deployment` - Production deployment patterns