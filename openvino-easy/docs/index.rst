OpenVINO-Easy Documentation
==========================

**Framework-agnostic Python wrapper for OpenVINO 2025**

OpenVINO-Easy transforms any AI model into a **TRUE 3-function experience**:

.. code-block:: python

   import oe

   oe.load("runwayml/stable-diffusion-v1-5")    # 1. Load (auto NPU>GPU>CPU)
   img = oe.infer("a neon cyber-city at night")   # 2. Infer
   stats = oe.benchmark()                        # 3. Benchmark
   oe.unload()                                   # Clean up

🚀 **Key Features**
------------------

* **TRUE 3-Function API**: ``oe.load()`` → ``oe.infer()`` → ``oe.benchmark()`` → ``oe.unload()``
* **Universal Model Support**: Hugging Face, ONNX, OpenVINO IR, TensorFlow  
* **Smart Device Selection**: Automatic NPU → GPU → CPU fallback with validation
* **FP16-NF4 Precision**: Arrow Lake/Lunar Lake NPU support for 4-bit inference
* **Stateful Module Design**: Global model state, no pipeline objects to manage
* **Professional CLI**: Rich colored output + JSON for CI/CD integration

📚 **Table of Contents**
------------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   installation
   model_compatibility
   production_deployment
   performance_tuning
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials

   examples/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   CONTRIBUTING
   CHANGELOG

🎯 **Quick Start**
-----------------

**Installation**

.. code-block:: bash

   # CPU-only (40MB, fastest install)
   pip install "openvino-easy[cpu]"
   
   # Intel GPU support
   pip install "openvino-easy[gpu]"
   
   # Intel NPU support (Arrow Lake/Lunar Lake with FP16-NF4)
   pip install "openvino-easy[npu]"
   
   # With quantization & datasets
   pip install "openvino-easy[quant,datasets]"

**3-Function API**

.. code-block:: python

   import oe
   
   # 1. Load any model (auto device selection)
   oe.load("microsoft/DialoGPT-medium")
   
   # 2. Run inference
   result = oe.infer("Hello, how are you?")
   
   # 3. Benchmark performance  
   stats = oe.benchmark()
   print(f"FPS: {stats['fps']}, Latency: {stats['mean_ms']}ms")
   
   # Clean up
   oe.unload()

**Advanced Usage**

.. code-block:: python

   # Smart device preference
   oe.load("model", device_preference=["NPU", "GPU", "CPU"])
   
   # Quantization for speed
   oe.load("model", dtype="int8")  # or "fp16", "fp16-nf4"
   
   # Get model information
   info = oe.get_info()
   print(f"Running on: {info['device']}")
   
   # Extended benchmarking
   stats = oe.benchmark(warmup_runs=10, benchmark_runs=50)

📊 **Model Compatibility Matrix**
---------------------------------

+-------------------+------------------+------------------+------------------+
| Model Type        | Format Support   | Conversion       | Performance      |
+===================+==================+==================+==================+
| **Text Models**   |                  |                  |                  |
+-------------------+------------------+------------------+------------------+
| BERT/RoBERTa      | ✅ HF, ONNX      | ✅ Optimum       | ⚡ Excellent     |
+-------------------+------------------+------------------+------------------+
| GPT/DialoGPT      | ✅ HF, ONNX      | ✅ Optimum       | ⚡ Excellent     |
+-------------------+------------------+------------------+------------------+
| T5/BART           | ✅ HF, ONNX      | ✅ Optimum       | ⚡ Good          |
+-------------------+------------------+------------------+------------------+
| **Vision Models** |                  |                  |                  |
+-------------------+------------------+------------------+------------------+
| ResNet/EfficientNet| ✅ HF, ONNX, TF | ✅ Direct        | ⚡ Excellent     |
+-------------------+------------------+------------------+------------------+
| Vision Transformer| ✅ HF, ONNX      | ✅ Optimum       | ⚡ Good          |
+-------------------+------------------+------------------+------------------+
| **Generative**    |                  |                  |                  |
+-------------------+------------------+------------------+------------------+
| Stable Diffusion  | ✅ HF Diffusers  | ✅ Optimum       | ⚡ Excellent     |
+-------------------+------------------+------------------+------------------+
| **Multimodal**    |                  |                  |                  |
+-------------------+------------------+------------------+------------------+
| CLIP              | ✅ HF, ONNX      | ✅ Optimum       | ⚡ Good          |
+-------------------+------------------+------------------+------------------+

**Legend**: ✅ Fully Supported, ⚠️ Partial Support, ❌ Not Supported

🔗 **External Resources**
-------------------------

* `OpenVINO Documentation <https://docs.openvino.ai/>`_
* `Hugging Face Hub <https://huggingface.co/models>`_
* `Intel Neural Compressor <https://github.com/intel/neural-compressor>`_
* `GitHub Repository <https://github.com/example/openvino-easy>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 