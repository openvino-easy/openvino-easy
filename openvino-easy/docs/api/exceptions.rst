Exception Handling
==================

OpenVINO-Easy provides a comprehensive exception hierarchy for better error handling and debugging.

Exception Hierarchy
-------------------

.. autoexception:: oe.ModelLoadError
   :show-inheritance:

   Base exception for all model loading errors. All other model-related exceptions inherit from this.

   **Example:**

   .. code-block:: python

      import oe
      
      try:
          pipeline = oe.load("some/model")
      except oe.ModelLoadError as e:
          print(f"Model loading failed: {e}")

.. autoexception:: oe.ModelNotFoundError
   :show-inheritance:

   Raised when a model cannot be found or accessed.

   **Common Causes:**
   * Invalid Hugging Face model ID
   * Private model without authentication
   * Network connectivity issues
   * Typos in model name

   **Example:**

   .. code-block:: python

      import oe
      
      try:
          pipeline = oe.load("nonexistent/model")
      except oe.ModelNotFoundError as e:
          print(f"Model not found: {e}")
          # Suggest alternatives or check spelling

.. autoexception:: oe.ModelConversionError
   :show-inheritance:

   Raised when model conversion from source format to OpenVINO fails.

   **Common Causes:**
   * Unsupported model architecture
   * Missing dependencies (optimum-intel, transformers)
   * Invalid model configuration
   * Insufficient memory

   **Example:**

   .. code-block:: python

      import oe
      
      try:
          pipeline = oe.load("complex/model", dtype="int8")
      except oe.ModelConversionError as e:
          print(f"Conversion failed: {e}")
          # Try with different dtype or install missing dependencies

.. autoexception:: oe.NetworkError
   :show-inheritance:

   Raised when network-related errors occur during model download.

   **Common Causes:**
   * Internet connectivity issues
   * Hugging Face Hub downtime
   * Download timeouts
   * Rate limiting

   **Example:**

   .. code-block:: python

      import oe
      
      try:
          pipeline = oe.load("large/model")
      except oe.NetworkError as e:
          print(f"Network error: {e}")
          # Retry later or check internet connection

.. autoexception:: oe.UnsupportedModelError
   :show-inheritance:

   Raised when a model format is not supported by OpenVINO-Easy.

   **Common Causes:**
   * Exotic model formats (CoreML, TensorFlow Lite)
   * Custom architectures not supported by OpenVINO
   * Deprecated model formats

   **Example:**

   .. code-block:: python

      import oe
      
      try:
          pipeline = oe.load("model.tflite")
      except oe.UnsupportedModelError as e:
          print(f"Unsupported format: {e}")
          # Convert to ONNX or use supported format

.. autoexception:: oe.CorruptedModelError
   :show-inheritance:

   Raised when a model file is corrupted or invalid.

   **Common Causes:**
   * Interrupted downloads
   * Corrupted cache files
   * Invalid model files
   * Storage issues

   **Example:**

   .. code-block:: python

      import oe
      
      try:
          pipeline = oe.load("model.xml")
      except oe.CorruptedModelError as e:
          print(f"Model corrupted: {e}")
          # Clear cache and re-download

Error Recovery Strategies
-------------------------

OpenVINO-Easy implements several error recovery mechanisms:

**Automatic Retry**
~~~~~~~~~~~~~~~~~~

Network operations automatically retry with exponential backoff:

.. code-block:: python

   # Automatic retry for downloads
   pipeline = oe.load("model-id")  # Retries 3 times on network errors

**Graceful Fallbacks**
~~~~~~~~~~~~~~~~~~~~~

Multiple conversion strategies are attempted:

.. code-block:: python

   # Tries multiple conversion methods automatically
   pipeline = oe.load("complex/model")
   # 1. Try optimum-intel conversion
   # 2. Fall back to direct OpenVINO conversion
   # 3. Provide detailed error if all fail

**Cache Integrity**
~~~~~~~~~~~~~~~~~~

Corrupted cache files are automatically detected and cleared:

.. code-block:: python

   # Cache verification prevents corrupted models
   pipeline = oe.load("model")  # Verifies integrity before loading

Best Practices
--------------

**Exception Handling**
~~~~~~~~~~~~~~~~~~~~~

Use specific exception types for targeted error handling:

.. code-block:: python

   import oe
   
   def load_model_safely(model_id):
       try:
           return oe.load(model_id)
       except oe.ModelNotFoundError:
           print(f"❌ Model '{model_id}' not found. Check the model ID.")
           return None
       except oe.NetworkError:
           print("🌐 Network error. Retrying in 30 seconds...")
           time.sleep(30)
           return load_model_safely(model_id)  # Retry
       except oe.ModelConversionError as e:
           print(f"🔧 Conversion failed: {e}")
           print("💡 Try installing additional dependencies:")
           print("   pip install 'openvino-easy[text,stable-diffusion]'")
           return None
       except oe.UnsupportedModelError:
           print(f"❌ Model format not supported. Convert to ONNX first.")
           return None
       except oe.ModelLoadError as e:
           print(f"❌ Unexpected error: {e}")
           return None

**Debugging**
~~~~~~~~~~~~

Enable verbose logging for troubleshooting:

.. code-block:: python

   import logging
   import oe
   
   # Enable debug logging
   logging.basicConfig(level=logging.DEBUG)
   
   try:
       pipeline = oe.load("problematic/model")
   except oe.ModelLoadError as e:
       print(f"Error details: {e}")
       print(f"Error type: {type(e).__name__}")

**Resource Management**
~~~~~~~~~~~~~~~~~~~~~~

Properly handle resources in error scenarios:

.. code-block:: python

   import oe
   
   pipeline = None
   try:
       pipeline = oe.load("model")
       result = pipeline.infer("input")
   except oe.ModelLoadError as e:
       print(f"Model loading failed: {e}")
   finally:
       # Cleanup if needed
       if pipeline:
           del pipeline

Error Messages
--------------

OpenVINO-Easy provides detailed, actionable error messages:

**Network Errors:**

.. code-block:: text

   NetworkError: Failed to download 'microsoft/DialoGPT-medium' after 3 attempts. 
   Last error: HTTPSConnectionPool(host='huggingface.co', port=443): 
   Read timed out. (read timeout=300.0)
   
   💡 Suggestions:
   • Check your internet connection
   • Try again later (server may be busy)
   • Use a different network if possible

**Conversion Errors:**

.. code-block:: text

   ModelConversionError: Missing dependency 'optimum' for model conversion.
   Install with: pip install 'openvino-easy[text,stable-diffusion]'
   
   💡 Model type detected: transformers_optimum
   💡 Required for: BERT, GPT, T5 model conversion

**Model Not Found:**

.. code-block:: text

   ModelNotFoundError: Model 'microsoft/DialoGPT-mediuum' not found on Hugging Face Hub.
   
   💡 Did you mean: 'microsoft/DialoGPT-medium'?
   💡 Check model ID at: https://huggingface.co/models

**Unsupported Format:**

.. code-block:: text

   UnsupportedModelError: Model format 'tflite' is not directly supported.
   Please convert to ONNX, OpenVINO IR, or a supported format first.
   
   💡 Supported formats:
   • Hugging Face (transformers, diffusers)
   • ONNX (.onnx files)
   • OpenVINO IR (.xml/.bin files)
   • TensorFlow SavedModel (.pb files)

Logging
-------

OpenVINO-Easy uses Python's standard logging module:

.. code-block:: python

   import logging
   import oe
   
   # Configure logging
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   
   # Load model with logging
   pipeline = oe.load("model")
   
   # Log levels used:
   # DEBUG: Detailed conversion steps
   # INFO: Major operations (download, conversion)
   # WARNING: Non-fatal issues (fallbacks, deprecated features)
   # ERROR: Failed operations (before raising exceptions) 