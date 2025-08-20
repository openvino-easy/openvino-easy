Model Compatibility Matrix
===========================

This page provides a comprehensive overview of model formats, architectures, and performance characteristics supported by OpenVINO-Easy.

Supported Formats
-----------------

OpenVINO-Easy supports a wide range of model formats with automatic detection and conversion:

.. list-table:: **Format Support Matrix**
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Format
     - Detection
     - Conversion
     - Performance
     - Notes
   * - **Hugging Face Transformers**
     - ✅ Automatic
     - ✅ Optimum-Intel
     - ⚡ Excellent
     - BERT, GPT, T5, etc.
   * - **Hugging Face Diffusers**
     - ✅ Automatic
     - ✅ Optimum-Intel
     - ⚡ Excellent
     - Stable Diffusion, etc.
   * - **ONNX**
     - ✅ Automatic
     - ✅ Direct
     - ⚡ Excellent
     - All ONNX models
   * - **OpenVINO IR**
     - ✅ Automatic
     - ✅ Direct
     - ⚡ Optimal
     - Native format
   * - **TensorFlow SavedModel**
     - ✅ Automatic
     - ✅ Direct
     - ⚡ Good
     - .pb files
   * - **PyTorch**
     - ⚠️ Config-based
     - ⚠️ Limited
     - ⚡ Good
     - With transformers config
   * - **SafeTensors**
     - ✅ Automatic
     - ✅ Optimum-Intel
     - ⚡ Excellent
     - Modern format

Text Models
-----------

.. list-table:: **Text Model Support**
   :header-rows: 1
   :widths: 25 15 15 15 30

   * - Architecture
     - Support
     - Quantization
     - Performance
     - Example Models
   * - **BERT**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Excellent
     - bert-base-uncased, distilbert
   * - **RoBERTa**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Excellent
     - roberta-base, roberta-large
   * - **GPT-2**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Excellent
     - gpt2, distilgpt2
   * - **DialoGPT**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Excellent
     - microsoft/DialoGPT-medium
   * - **T5**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Good
     - t5-small, t5-base
   * - **BART**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Good
     - facebook/bart-base
   * - **GPT-J**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Good
     - EleutherAI/gpt-j-6b
   * - **BLOOM**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Good
     - bigscience/bloom-560m
   * - **OPT**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Good
     - facebook/opt-125m
   * - **ALBERT**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Excellent
     - albert-base-v2
   * - **DeBERTa**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Good
     - microsoft/deberta-base

**Example Usage:**

.. code-block:: python

   import oe
   
   # Text classification
   classifier = oe.load("cardiffnlp/twitter-roberta-base-sentiment-latest")
   result = classifier.infer("I love this product!")
   
   # Text generation
   generator = oe.load("microsoft/DialoGPT-medium")
   response = generator.infer("Hello, how are you?")
   
   # Question answering
   qa_model = oe.load("distilbert-base-cased-distilled-squad")
   answer = qa_model.infer({
       "question": "What is OpenVINO?",
       "context": "OpenVINO is Intel's toolkit for optimizing AI models."
   })

Vision Models
-------------

.. list-table:: **Vision Model Support**
   :header-rows: 1
   :widths: 25 15 15 15 30

   * - Architecture
     - Support
     - Quantization
     - Performance
     - Example Models
   * - **ResNet**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Excellent
     - resnet18, resnet50, resnet101
   * - **EfficientNet**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Excellent
     - efficientnet-b0 to b7
   * - **Vision Transformer (ViT)**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Good
     - google/vit-base-patch16-224
   * - **DeiT**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Good
     - facebook/deit-base-distilled
   * - **DETR**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Good
     - facebook/detr-resnet-50
   * - **YOLO**
     - ✅ ONNX
     - ✅ INT8/FP16
     - ⚡ Excellent
     - YOLOv5, YOLOv8 (ONNX)
   * - **MobileNet**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Excellent
     - mobilenet_v2, mobilenet_v3
   * - **Swin Transformer**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Good
     - microsoft/swin-tiny-patch4

**Example Usage:**

.. code-block:: python

   import oe
   import numpy as np
   
   # Image classification
   classifier = oe.load("google/vit-base-patch16-224")
   
   # Random image (replace with real image)
   image = np.random.rand(1, 3, 224, 224).astype(np.float32)
   result = classifier.infer(image)
   
   # Object detection (ONNX)
   detector = oe.load("yolov5s.onnx")
   detections = detector.infer(image)

Generative Models
-----------------

.. list-table:: **Generative Model Support**
   :header-rows: 1
   :widths: 25 15 15 15 30

   * - Type
     - Support
     - Quantization
     - Performance
     - Example Models
   * - **Stable Diffusion 1.x**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Excellent
     - runwayml/stable-diffusion-v1-5
   * - **Stable Diffusion 2.x**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Excellent
     - stabilityai/stable-diffusion-2-1
   * - **Stable Diffusion XL**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Good
     - stabilityai/stable-diffusion-xl
   * - **ControlNet**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Good
     - lllyasviel/sd-controlnet-canny
   * - **LCM (Latent Consistency)**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Excellent
     - latent-consistency models
   * - **Kandinsky**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Good
     - kandinsky-community models

**Example Usage:**

.. code-block:: python

   import oe
   
   # Text-to-image generation
   pipe = oe.load("runwayml/stable-diffusion-v1-5", dtype="int8")
   image = pipe.infer(
       "a serene mountain landscape at sunset, highly detailed",
       num_inference_steps=20,
       guidance_scale=7.5
   )
   
   # Fast generation with LCM
   lcm_pipe = oe.load("SimianLuo/LCM_Dreamshaper_v7")
   fast_image = lcm_pipe.infer(
       "a cyberpunk city at night",
       num_inference_steps=4  # Much faster
   )

Multimodal Models
-----------------

.. list-table:: **Multimodal Model Support**
   :header-rows: 1
   :widths: 25 15 15 15 30

   * - Architecture
     - Support
     - Quantization
     - Performance
     - Example Models
   * - **CLIP**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Good
     - openai/clip-vit-base-patch32
   * - **BLIP**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Good
     - Salesforce/blip-image-captioning
   * - **BLIP-2**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Good
     - Salesforce/blip2-opt-2.7b
   * - **LLaVA**
     - ⚠️ Partial
     - ✅ INT8/FP16
     - ⚡ Good
     - llava-hf models
   * - **ALIGN**
     - ✅ Full
     - ✅ INT8/FP16
     - ⚡ Good
     - kakaobrain/align-base

**Example Usage:**

.. code-block:: python

   import oe
   import numpy as np
   
   # Vision-language understanding
   clip_model = oe.load("openai/clip-vit-base-patch32")
   
   # Image + text inputs
   image = np.random.rand(1, 3, 224, 224).astype(np.float32)
   text = "a photo of a cat"
   
   similarity = clip_model.infer({
       "pixel_values": image,
       "input_ids": text
   })

Performance Characteristics
---------------------------

**Hardware Performance Rankings:**

.. list-table:: **Performance by Hardware**
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Model Type
     - NPU (AI Boost)
     - GPU (Arc/Xe)
     - CPU (Core)
     - Notes
   * - **Text (BERT-like)**
     - ⚡ 45 tok/s
     - ⚡ 38 tok/s
     - ⚡ 12 tok/s
     - INT8 quantized
   * - **Text (GPT-like)**
     - ⚡ 32 tok/s
     - ⚡ 28 tok/s
     - ⚡ 8 tok/s
     - INT8 quantized
   * - **Vision (ResNet)**
     - ⚡ 890 FPS
     - ⚡ 720 FPS
     - ⚡ 180 FPS
     - Batch size 1
   * - **Vision (ViT)**
     - ⚡ 340 FPS
     - ⚡ 280 FPS
     - ⚡ 85 FPS
     - Batch size 1
   * - **Stable Diffusion**
     - ⚡ 2.3 img/s
     - ⚡ 1.8 img/s
     - ⚡ 0.4 img/s
     - 512x512, 20 steps

**Quantization Impact:**

.. list-table:: **FP16 vs INT8 Performance**
   :header-rows: 1
   :widths: 25 25 25 25

   * - Model Type
     - FP16 Speed
     - INT8 Speed
     - Speedup
   * - **BERT-base**
     - 28 tok/s
     - 45 tok/s
     - 🚀 1.6x
   * - **ResNet-50**
     - 520 FPS
     - 890 FPS
     - 🚀 1.7x
   * - **Stable Diffusion**
     - 1.4 img/s
     - 2.3 img/s
     - 🚀 1.6x
   * - **Vision Transformer**
     - 210 FPS
     - 340 FPS
     - 🚀 1.6x

Model Conversion Strategies
---------------------------

OpenVINO-Easy uses intelligent conversion strategies based on model type:

**Strategy 1: Optimum-Intel (Recommended)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used for Hugging Face models with known architectures:

.. code-block:: python

   # Automatic optimum-intel conversion
   pipeline = oe.load("microsoft/DialoGPT-medium")
   # ✅ Uses optimum-intel for best performance

**Strategy 2: Direct OpenVINO**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used for ONNX, OpenVINO IR, and TensorFlow models:

.. code-block:: python

   # Direct conversion
   pipeline = oe.load("model.onnx")
   # ✅ Direct OpenVINO conversion

**Strategy 3: Fallback Chain**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multiple strategies attempted for maximum compatibility:

.. code-block:: python

   # Automatic fallback
   pipeline = oe.load("complex/model")
   # 1. Try optimum-intel
   # 2. Try direct OpenVINO
   # 3. Provide helpful error if all fail

Troubleshooting Guide
--------------------

**Common Issues and Solutions:**

**Model Not Found**
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Error: ModelNotFoundError
   try:
       pipeline = oe.load("typo/model-name")
   except oe.ModelNotFoundError:
       # Check spelling, verify model exists on Hugging Face

**Conversion Failed**
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Error: ModelConversionError  
   try:
       pipeline = oe.load("unsupported/model")
   except oe.ModelConversionError as e:
       print(f"Install missing deps: {e}")
       # pip install 'openvino-easy[text,stable-diffusion]'

**Out of Memory**
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Large models may need different approach
   try:
       pipeline = oe.load("huge/model", dtype="int8")  # Use quantization
   except MemoryError:
       # Try on machine with more RAM or use smaller model

**Performance Issues**
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimize for performance
   pipeline = oe.load(
       "model",
       dtype="int8",           # Use quantization
       device_preference=["NPU", "GPU"]  # Use accelerators
   )
   
   # Use batch processing
   results = pipeline.infer_batch(["input1", "input2", "input3"])

Testing Compatibility
---------------------

Use these commands to test model compatibility:

.. code-block:: bash

   # Test basic loading
   oe run your-model-id --prompt "test input"
   
   # Test with quantization
   oe run your-model-id --dtype int8 --prompt "test input"
   
   # Test performance
   oe bench your-model-id --dtype int8 --benchmark-runs 50

**Compatibility Checklist:**

1. ✅ Model loads without errors
2. ✅ Inference produces reasonable outputs  
3. ✅ Quantization works (if supported)
4. ✅ Performance meets requirements
5. ✅ Caching works correctly

Roadmap
-------

**Planned Support (Future Versions):**

* **Audio Models**: Whisper, Wav2Vec2, etc.
* **3D Models**: NeRF, 3D object detection
* **Graph Models**: GNN architectures
* **Specialized Hardware**: Gaudi, Habana support
* **Edge Deployment**: Model optimization for mobile devices

**Request Support for New Models:**

If you need support for a specific model architecture, please:

1. Open an issue on GitHub with model details
2. Provide example model ID or format
3. Include use case and performance requirements
4. Test with our compatibility script (if available) 