Examples and Tutorials
======================

.. note::
   **All examples use the new 3-function API** (``oe.load()``, ``oe.infer()``, ``oe.benchmark()``, ``oe.unload()``).
   For legacy Pipeline API patterns, see the git history or :doc:`../api/pipeline`.

Comprehensive examples demonstrating OpenVINO-Easy capabilities across different domains and use cases.

.. toctree::
   :maxdepth: 2
   :caption: Basic Examples

   text_generation_llm
   computer_vision_pipeline
   audio_speech_recognition
   multimodal_ai_showcase
   production_deployment

.. toctree::
   :maxdepth: 2
   :caption: Advanced Examples

   audio_speech_recognition_enhanced
   cross_platform_deployment
   performance_optimization
   custom_model_integration

Quick Start Examples
--------------------

**Text Generation (3-function API):**

.. code-block:: python

   import oe
   oe.load("microsoft/DialoGPT-medium")
   response = oe.infer("Hello, how are you today?")
   oe.unload()

**Image Classification:**

.. code-block:: python

   import oe
   from PIL import Image
   
   oe.load("microsoft/resnet-50")
   image = Image.open("photo.jpg")
   result = oe.infer(image)
   oe.unload()

**Speech Recognition:**

.. code-block:: python

   import oe
   
   oe.load("openai/whisper-base")
   transcription = oe.infer("audio_file.wav")
   oe.unload()

Example Categories
------------------

Basic Examples
~~~~~~~~~~~~~~

Perfect for getting started with OpenVINO-Easy:

- **Text Generation**: Chat models, story generation, Q&A systems
- **Computer Vision**: Image classification, object detection, image processing
- **Audio Processing**: Speech recognition, audio classification, TTS
- **Multimodal AI**: Image captioning, visual question answering
- **Production Deployment**: Flask/FastAPI services, monitoring, scaling

Advanced Examples
~~~~~~~~~~~~~~~~~

For users who need specialized functionality:

- **Enhanced Audio**: Real-time processing, batch transcription, audio enhancement
- **Cross-Platform**: Docker, Kubernetes, CI/CD, cloud deployment
- **Performance Optimization**: Benchmarking, profiling, memory management
- **Custom Integration**: Custom models, preprocessing, postprocessing

Running Examples
----------------

**Prerequisites:**

.. code-block:: bash

   # Install OpenVINO-Easy with examples dependencies
   pip install "openvino-easy[full]"
   
   # Additional packages for specific examples
   pip install matplotlib seaborn jupyter sounddevice

**Download Examples:**

.. code-block:: bash

   git clone https://github.com/your-org/openvino-easy.git
   cd openvino-easy/examples

**Run Individual Examples:**

.. code-block:: bash

   # Basic text generation
   python text_generation_llm.py
   
   # Computer vision pipeline
   python computer_vision_pipeline.py
   
   # Audio processing
   python audio_speech_recognition.py

**Interactive Jupyter Notebooks:**

.. code-block:: bash

   jupyter notebook stable_diffusion_showcase.ipynb

Example Highlights
------------------

Text Generation LLM
~~~~~~~~~~~~~~~~~~~

**File**: ``text_generation_llm.py``

Comprehensive text generation example featuring:

- Multiple model architectures (GPT, T5, BERT)
- Interactive chat interface
- Batch text processing
- Performance comparison across devices
- Custom prompt templates

**Key Features:**
- Device optimization (CPU/GPU/NPU)
- Temperature and sampling controls
- Streaming text generation
- Conversation memory

.. code-block:: python

   # Snippet from the example
   oe.load("microsoft/DialoGPT-large", device_preference=["NPU", "GPU", "CPU"])
   
   # Interactive chat
   while True:
       user_input = input("You: ")
       response = oe.infer(user_input)
       print(f"AI: {response}")
   
   # Clean up when done
   oe.unload()

Computer Vision Pipeline  
~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``computer_vision_pipeline.py``

End-to-end computer vision workflows:

- Image classification with confidence scores
- Object detection and tracking
- Image segmentation
- Batch image processing
- Real-time webcam inference

**Key Features:**
- Multiple vision model types
- Preprocessing and postprocessing
- Visualization tools
- Performance metrics

.. code-block:: python

   # Multi-model vision pipeline (sequential loading)
   from PIL import Image
   
   for image_path in image_files:
       image = Image.open(image_path)
       
       # Classification
       oe.load("resnet50", device_preference=["GPU", "CPU"])
       classification = oe.infer(image)
       oe.unload()
       
       # Object detection
       oe.load("yolov8n", device_preference=["GPU", "CPU"])
       detections = oe.infer(image)
       oe.unload()

Audio Speech Recognition Enhanced
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``audio_speech_recognition_enhanced.py``

Professional audio processing capabilities:

- Real-time speech recognition
- Multiple audio format support
- Audio enhancement and noise reduction
- Batch audio file processing
- Language detection and multilingual support

**Key Features:**
- Streaming audio processing
- Audio quality optimization
- Confidence scoring
- Performance benchmarking

.. code-block:: python

   # Real-time audio processing
   recognizer = SpeechRecognitionPipeline("openai/whisper-base", device="NPU")
   
   # Process audio stream
   with microphone_stream() as stream:
       for chunk in stream:
           transcription = recognizer.transcribe(chunk)
           print(f">> {transcription['text']}")

Cross-Platform Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``cross_platform_deployment.py``

Production deployment across platforms:

- Platform detection and optimization
- Docker containerization
- Kubernetes manifests
- CI/CD pipeline generation
- Cloud deployment patterns

**Key Features:**
- Automatic platform optimization
- Container orchestration
- Infrastructure as Code
- Monitoring and scaling

.. code-block:: python

   # Auto-optimize for current platform
   manager = DeploymentManager()
   pipeline = manager.create_optimized_pipeline("model_name")
   
   # Generate deployment configs
   dockerfile = manager.generate_dockerfile("model_name")
   k8s_manifests = manager.generate_kubernetes_manifests("model_name")

Production Deployment
~~~~~~~~~~~~~~~~~~~~

**File**: ``production_deployment.py``

Enterprise-ready service deployment:

- REST API with FastAPI/Flask
- Authentication and authorization
- Rate limiting and input validation
- Monitoring and observability
- Error handling and recovery

**Key Features:**
- Scalable API architecture
- Prometheus metrics
- Structured logging  
- Health checks
- Auto-scaling support

.. code-block:: python

   # Production API service
   app = FastAPI()
   model = oe.load("production_model", device="auto")
   
   @app.post("/predict")
   async def predict(request: PredictionRequest):
       result = await model(request.input)
       return {"prediction": result, "model_version": "1.0.0"}

Interactive Examples
--------------------

Jupyter Notebooks
~~~~~~~~~~~~~~~~~

**Stable Diffusion Showcase**: ``stable_diffusion_showcase.ipynb``

Interactive image generation with:
- Text-to-image generation
- Image-to-image transformation
- Parameter tuning interface
- Gallery of generated images
- Performance comparison

**Multimodal AI Demo**: ``multimodal_demo.ipynb``

Cross-modal AI capabilities:
- Image captioning
- Visual question answering
- Text-to-speech synthesis
- Audio analysis

Command Line Examples
~~~~~~~~~~~~~~~~~~~~

**Quick Inference:**

.. code-block:: bash

   # Text generation
   oe run "gpt2" "The future of AI is"
   
   # Image classification
   oe run "resnet50" "image.jpg" --format json
   
   # Speech recognition
   oe run "whisper-base" "audio.wav"

**Benchmarking:**

.. code-block:: bash

   # Compare devices
   oe bench "microsoft/DialoGPT-medium" --compare-devices
   
   # Compare precisions
   oe bench "resnet50" --compare-precision --device GPU
   
   # Extended benchmark
   oe bench "model" -n 1000 --profile

Use Case Scenarios
------------------

Customer Support Chatbot
~~~~~~~~~~~~~~~~~~~~~~~~

Combine text generation with sentiment analysis:

.. code-block:: python

   # Multi-model customer support
   def handle_customer_query(message):
       # Get sentiment
       oe.load("cardiffnlp/twitter-roberta-base-sentiment")
       sentiment = oe.infer(message)
       oe.unload()
       
       # Generate response
       oe.load("microsoft/DialoGPT-large")
       response = oe.infer(message)
       oe.unload()
       
       return {"response": response, "sentiment": sentiment}

Content Moderation Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-modal content analysis:

.. code-block:: python

   # Content moderation pipeline
   def moderate_content(text, image):
       # Text moderation
       oe.load("unitary/toxic-bert")
       text_score = oe.infer(text)
       oe.unload()
       
       # Image moderation
       oe.load("Falconsai/nsfw_image_detection")
       image_score = oe.infer(image)
       oe.unload()
       
       return {"safe": text_score < 0.5 and image_score < 0.5}

Real-Time Analytics Dashboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Live data processing with visualization:

.. code-block:: python

   # Real-time analytics
   def process_social_media_stream(posts):
       oe.load("cardiffnlp/twitter-roberta-base-sentiment")
       
       results = []
       for post in posts:
           result = oe.infer(post)
           results.append(result)
       
       oe.unload()
       return analyze_sentiment_trends(results)

Performance Benchmarks
----------------------

Expected performance on common hardware:

**Intel i7-12700K + Arc A770:**

+------------------+----------+----------+----------+
| Model            | CPU      | GPU      | NPU      |
+==================+==========+==========+==========+
| BERT-base        | 45ms     | 12ms     | 8ms      |
+------------------+----------+----------+----------+
| GPT-2 small      | 120ms    | 25ms     | 15ms     |
+------------------+----------+----------+----------+
| ResNet-50        | 35ms     | 8ms      | 12ms     |
+------------------+----------+----------+----------+
| Whisper-base     | 180ms    | 45ms     | 30ms     |
+------------------+----------+----------+----------+

**Memory Usage:**

- Small models (<1B params): 1-2GB
- Medium models (1-7B params): 3-8GB  
- Large models (7B+ params): 12GB+

Troubleshooting Examples
-----------------------

**Model Loading Issues:**

.. code-block:: python

   try:
       oe.load("model_name")
   except Exception as e:
       print(f"Model loading failed: {e}")
       print("Suggestion: Check model name or network connection")

**Device Compatibility:**

.. code-block:: python

   # Robust device selection (built into 3-function API)
   oe.load("model", device_preference=["NPU", "GPU", "CPU"])
   info = oe.get_info()
   print(f"Using device: {info['device']}")

**Memory Optimization:**

.. code-block:: python

   # Memory-efficient loading
   oe.load(
       "large_model",
       device_preference=["NPU", "GPU", "CPU"],  # Auto-select
       dtype="fp16"                                # Reduce memory usage
   )
   
   # Use model efficiently
   result = oe.infer(input_data)
   
   # Clean up immediately when done
   oe.unload()

Contributing Examples
--------------------

We welcome community contributions to our examples collection!

**Guidelines:**
- Follow the existing code style and structure
- Include comprehensive documentation and comments
- Add error handling and edge case management
- Test on multiple platforms when possible
- Include performance benchmarks if relevant

**Example Template:**

.. code-block:: python

   #!/usr/bin/env python3
   """
   Example Title
   
   Brief description of what this example demonstrates.
   
   Features:
   - Feature 1
   - Feature 2
   - Feature 3
   
   Requirements:
       pip install "openvino-easy[extras]" additional-packages
   """
   
   import oe
   # ... implementation

**Submission Process:**
1. Create example following the template
2. Test on multiple platforms
3. Add to appropriate category in this index
4. Submit pull request with detailed description

See Also
--------

- :doc:`../getting_started` - Basic tutorials and concepts
- :doc:`../api/index` - Complete API reference
- :doc:`../performance_tuning` - Optimization guidelines
- :doc:`../production_deployment` - Production deployment patterns
- :doc:`../troubleshooting` - Common issues and solutions