Troubleshooting Guide
====================

This guide covers common issues and solutions for OpenVINO-Easy. Issues are organized by category for quick resolution.

Installation Issues
-------------------

ModuleNotFoundError
~~~~~~~~~~~~~~~~~~~

**Symptoms:**
.. code-block:: text

   ModuleNotFoundError: No module named 'openvino'
   ModuleNotFoundError: No module named 'oe'

**Solutions:**

1. **Basic installation issue:**
   .. code-block:: bash

      pip install --upgrade openvino-easy

2. **Virtual environment issue:**
   .. code-block:: bash

      # Verify you're in the correct environment
      which python
      pip list | grep openvino

3. **Python version incompatibility:**
   .. code-block:: bash

      python --version  # Should be 3.8-3.12
      pip install "openvino-easy" --force-reinstall

4. **System-wide vs user installation:**
   .. code-block:: bash

      # Try user installation
      pip install --user openvino-easy

      # Or use virtual environment
      python -m venv venv
      source venv/bin/activate  # Linux/Mac
      # venv\Scripts\activate   # Windows
      pip install openvino-easy

Import Errors After Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
.. code-block:: text

   ImportError: cannot import name 'Core' from 'openvino.runtime'

**Solutions:**

1. **Clear cache and reinstall:**
   .. code-block:: bash

      pip cache purge
      pip uninstall openvino openvino-dev openvino-easy
      pip install openvino-easy

2. **Check for conflicting installations:**
   .. code-block:: bash

      pip list | grep openvino
      # Should see consistent versions

3. **Verify OpenVINO installation:**
   .. code-block:: python

      try:
          from openvino.runtime import Core
          print("OpenVINO runtime OK")
      except ImportError as e:
          print(f"OpenVINO issue: {e}")

Hardware Detection Issues
-------------------------

GPU Not Detected
~~~~~~~~~~~~~~~~

**Symptoms:**
.. code-block:: python

   devices = oe.list_devices()
   print(devices)  # Only shows ['CPU']

**Diagnosis:**
.. code-block:: python

   import oe
   
   # Check detailed device info
   print("Available devices:", oe.list_devices())
   
   # Check OpenVINO core directly
   from openvino.runtime import Core
   core = Core()
   print("OpenVINO devices:", core.available_devices)

**Solutions:**

1. **Update graphics drivers:**
   
   **Windows:**
   - Download from Intel.com or device manufacturer
   - Install latest Intel Graphics driver
   - Restart system
   
   **Linux:**
   .. code-block:: bash

      # Ubuntu/Debian
      sudo apt update
      sudo apt install intel-opencl-icd
      
      # Check GPU is visible
      clinfo | grep "Device Name"

2. **Install OpenCL runtime:**
   
   **Windows:**
   - Often included with graphics drivers
   - Manually install Intel OpenCL Runtime if needed
   
   **Linux:**
   .. code-block:: bash

      # Install OpenCL
      sudo apt install opencl-headers ocl-icd-opencl-dev
      
      # Verify installation
      ls /usr/lib/x86_64-linux-gnu/*OpenCL*

3. **Check hardware compatibility:**
   .. code-block:: bash

      # Linux: Check GPU info
      lspci | grep -i vga
      lspci | grep -i intel
      
      # Windows: Check Device Manager
      # Look for Intel(R) graphics under Display adapters

4. **Verify OpenCL support:**
   .. code-block:: python

      try:
          import pyopencl as cl
          platforms = cl.get_platforms()
          for platform in platforms:
              print(f"Platform: {platform.name}")
              devices = platform.get_devices()
              for device in devices:
                  print(f"  Device: {device.name}")
      except ImportError:
          print("Install pyopencl: pip install pyopencl")
      except Exception as e:
          print(f"OpenCL error: {e}")

NPU Not Available
~~~~~~~~~~~~~~~~

**Symptoms:**
.. code-block:: text

   No NPU device found in oe.list_devices()

**Solutions:**

1. **Verify NPU hardware support:**
   - Intel Arc A-Series graphics
   - Intel Core Ultra processors
   - Check manufacturer specifications

2. **Update drivers:**
   - Intel Graphics driver 31.0.101.4502 or newer
   - Download from Intel support site
   - Restart after installation

3. **Check device manager (Windows):**
   - Look under "System devices" for Neural Processing Unit
   - If not visible, NPU may not be supported or enabled

4. **BIOS/UEFI settings:**
   - Some systems have NPU enable/disable option
   - Check "Advanced" or "Intel" settings in BIOS

5. **Verify with Intel tools:**
   .. code-block:: bash

      # Download and run Intel System Support Utility
      # Check for NPU in the report

Model Loading Issues
--------------------

Model Download Failures
~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
.. code-block:: text

   FileNotFoundError: Model not found
   ConnectionError: Failed to download model

**Solutions:**

1. **Check internet connection:**
   .. code-block:: python

      import requests
      try:
          response = requests.get("https://huggingface.co", timeout=10)
          print("Connection OK")
      except requests.RequestException as e:
          print(f"Connection issue: {e}")

2. **Clear model cache:**
   .. code-block:: python

      import oe
      oe.clear_cache()
      
      # Or manually clear cache directory
      import shutil
      from pathlib import Path
      
      cache_dir = Path.home() / '.cache' / 'openvino_easy'
      if cache_dir.exists():
          shutil.rmtree(cache_dir)

3. **Use local model files:**
   .. code-block:: python

      # If you have local ONNX/IR files
      model = oe.load("/path/to/local/model.onnx")

4. **Configure proxy (if behind corporate firewall):**
   .. code-block:: python

      import os
      os.environ['HTTP_PROXY'] = 'http://proxy.company.com:8080'
      os.environ['HTTPS_PROXY'] = 'https://proxy.company.com:8080'

5. **Alternative model sources:**
   .. code-block:: python

      # Try different model formats
      model = oe.load("microsoft/DialoGPT-medium")  # HuggingFace
      model = oe.load("path/to/model.onnx")         # Local ONNX
      model = oe.load("path/to/model.xml")          # OpenVINO IR

Unsupported Model Format
~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
.. code-block:: text

   ValueError: Unsupported model format
   RuntimeError: Failed to load model

**Solutions:**

1. **Check supported formats:**
   .. code-block:: python

      print(oe.supported_formats())
      # Should include: ONNX, OpenVINO IR, TensorFlow, PyTorch

2. **Convert unsupported models:**
   .. code-block:: bash

      # PyTorch to ONNX
      python -c "
      import torch
      import torch.onnx
      
      model = torch.load('model.pth')
      dummy_input = torch.randn(1, 3, 224, 224)
      torch.onnx.export(model, dummy_input, 'model.onnx')
      "

3. **Use model conversion tools:**
   .. code-block:: bash

      # OpenVINO Model Optimizer
      mo --input_model model.pb --output_dir converted/

Memory Issues
-------------

Out of Memory Errors
~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
.. code-block:: text

   RuntimeError: [GPU] out of memory
   MemoryError: Unable to allocate array

**Solutions:**

1. **Reduce batch size:**
   .. code-block:: python

      # Instead of processing large batches
      results = model(large_input_list)
      
      # Process in smaller chunks
      results = []
      batch_size = 4
      for i in range(0, len(input_list), batch_size):
          batch = input_list[i:i+batch_size]
          results.extend(model(batch))

2. **Use memory optimization:**
   .. code-block:: python

      model = oe.load("model_name", optimize_memory=True)

3. **Lower precision:**
   .. code-block:: python

      model = oe.load("model_name", precision="FP16")  # Half precision

4. **GPU memory management:**
   .. code-block:: python

      # Clear GPU cache between runs
      import gc
      gc.collect()
      
      # Monitor memory usage
      model = oe.load("model_name")
      print(f"GPU memory used: {model.get_memory_usage()}")

5. **Use CPU for large models:**
   .. code-block:: python

      model = oe.load("large_model", device="CPU")

Performance Issues
------------------

Slow Inference
~~~~~~~~~~~~~

**Symptoms:**
- Very slow first inference (> 30 seconds)
- Consistently slow subsequent inferences

**Diagnosis:**
.. code-block:: python

   import time
   import oe

   model = oe.load("microsoft/DialoGPT-medium")
   
   # Measure first inference (includes model compilation)
   start = time.time()
   result1 = model("Hello")
   first_time = time.time() - start
   
   # Measure subsequent inference
   start = time.time()
   result2 = model("How are you?")
   second_time = time.time() - start
   
   print(f"First inference: {first_time:.2f}s")
   print(f"Second inference: {second_time:.2f}s")
   print(f"Device: {model.device}")

**Solutions:**

1. **Enable model caching:**
   .. code-block:: python

      model = oe.load("model_name", cache_compiled=True)

2. **Use appropriate device:**
   .. code-block:: python

      # Check device performance
      devices = oe.list_devices()
      for device in devices:
          model = oe.load("model_name", device=device)
          # Run benchmark to compare

3. **Optimize model precision:**
   .. code-block:: python

      # FP16 for better GPU performance
      model = oe.load("model_name", precision="FP16", device="GPU")
      
      # INT8 for faster inference (may reduce quality)
      model = oe.load("model_name", precision="INT8")

4. **Batch processing:**
   .. code-block:: python

      # Process multiple inputs together
      inputs = ["input1", "input2", "input3"]
      results = model(inputs)  # Faster than individual calls

5. **Use performance mode:**
   .. code-block:: python

      model = oe.load("model_name", performance_mode="high")

Audio Processing Issues
-----------------------

Audio Dependencies Missing
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
.. code-block:: text

   ImportError: No module named 'librosa'
   AudioError: Failed to process audio file

**Solutions:**

1. **Install audio dependencies:**
   .. code-block:: bash

      pip install "openvino-easy[audio]"
      
      # Or install individually
      pip install librosa soundfile

2. **Verify audio support:**
   .. code-block:: python

      try:
          import librosa
          import soundfile as sf
          print("Audio support available")
      except ImportError:
          print("Install audio dependencies")

3. **Use fallback processing:**
   .. code-block:: python

      # If librosa not available, use basic processing
      model = oe.load("audio_model", audio_fallback=True)

Audio File Format Issues
~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
.. code-block:: text

   ValueError: Unsupported audio format
   RuntimeError: Failed to read audio file

**Solutions:**

1. **Check supported formats:**
   .. code-block:: python

      import oe
      print("Supported audio formats:", oe.audio.supported_formats())

2. **Convert audio format:**
   .. code-block:: python

      import librosa
      
      # Load and convert
      audio, sr = librosa.load("input.mp3", sr=16000)
      
      # Save as WAV
      import soundfile as sf
      sf.write("output.wav", audio, sr)

3. **Use compatible formats:**
   - WAV files are most reliable
   - Use 16kHz sample rate for speech models
   - Mono audio (single channel) preferred

Deployment Issues
-----------------

Docker Container Problems
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
.. code-block:: text

   GPU not available in container
   Permission denied accessing devices

**Solutions:**

1. **Enable GPU access:**
   .. code-block:: bash

      # Intel GPU access
      docker run --device /dev/dri -it your-image
      
      # Add user to video group
      docker run --group-add video -it your-image

2. **Install drivers in container:**
   .. code-block:: dockerfile

      FROM python:3.11-slim
      
      # Install Intel OpenCL
      RUN apt-get update && \
          apt-get install -y intel-opencl-icd && \
          rm -rf /var/lib/apt/lists/*
      
      RUN pip install "openvino-easy[gpu]"

3. **Check container capabilities:**
   .. code-block:: bash

      # Inside container
      ls -la /dev/dri/  # Should show render devices
      clinfo            # Should list OpenCL devices

Production Deployment Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
- Models work locally but fail in production
- Inconsistent performance across environments

**Solutions:**

1. **Environment consistency:**
   .. code-block:: python

      # Pin versions in requirements.txt
      openvino-easy==1.0.0
       openvino==2025.2.0

2. **Resource allocation:**
   .. code-block:: yaml

      # Kubernetes deployment
      resources:
        requests:
          memory: "2Gi"
          cpu: "1"
        limits:
          memory: "4Gi"
          cpu: "2"

3. **Model preloading:**
   .. code-block:: python

      # Preload models during container startup
      import oe
      
      # Warm up models
      model = oe.load("your_model")
      model("warmup input")  # Trigger compilation

Error Reporting and Debug
-------------------------

Enable Debug Logging
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import logging
   import oe

   # Enable debug logging
   logging.basicConfig(level=logging.DEBUG)
   oe.set_log_level("DEBUG")

   # Now run your code with detailed logging
   model = oe.load("model_name")

System Information
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import oe
   
   # Get comprehensive system info for bug reports
   info = oe.system_info()
   print("System Information:")
   print(f"  OS: {info['os']}")
   print(f"  Python: {info['python_version']}")
   print(f"  OpenVINO-Easy: {info['oe_version']}")
   print(f"  OpenVINO: {info['openvino_version']}")
   print(f"  Available devices: {info['devices']}")
   print(f"  GPU details: {info['gpu_info']}")

Performance Profiling
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import oe
   
   # Enable profiling
   model = oe.load("model_name", profile=True)
   
   # Run inference
   result = model("test input")
   
   # Get profiling report
   profile_report = model.get_profile_report()
   print(profile_report)

Getting Help
------------

If none of these solutions work:

1. **Check GitHub Issues:**
   - Search existing issues: https://github.com/your-org/openvino-easy/issues
   - Check closed issues for solutions

2. **Create a Bug Report:**
   Include this information:
   
   .. code-block:: python

      import oe
      
      print("Bug Report Information:")
      print("=" * 50)
      
      # System info
      info = oe.system_info()
      for key, value in info.items():
          print(f"{key}: {value}")
      
      # Error details
      print("\nError Details:")
      print("- What you were trying to do")
      print("- Full error message and traceback")
      print("- Steps to reproduce")
      print("- Expected vs actual behavior")

3. **Community Support:**
   - Discussions: https://github.com/your-org/openvino-easy/discussions
   - Stack Overflow: Tag with `openvino-easy`

4. **Intel OpenVINO Support:**
   - For underlying OpenVINO issues
   - Intel Community Forum
   - OpenVINO GitHub repository

Quick Diagnostic Script
-----------------------

Use this script to diagnose common issues:

.. code-block:: python

   #!/usr/bin/env python3
   """OpenVINO-Easy diagnostic script"""
   
   import sys
   import importlib
   
   def check_installation():
       """Check if OpenVINO-Easy is properly installed."""
       try:
           import oe
           print("✓ OpenVINO-Easy imported successfully")
           return True
       except ImportError as e:
           print(f"✗ OpenVINO-Easy import failed: {e}")
           return False
   
   def check_devices():
       """Check available devices."""
       try:
           import oe
           devices = oe.list_devices()
           print(f"✓ Available devices: {devices}")
           if len(devices) == 1 and devices[0] == 'CPU':
               print("! Only CPU available - check GPU setup")
           return True
       except Exception as e:
           print(f"✗ Device detection failed: {e}")
           return False
   
   def check_model_loading():
       """Test basic model loading."""
       try:
           import oe
           model = oe.load("microsoft/DialoGPT-small")
           print("✓ Model loading successful")
           return True
       except Exception as e:
           print(f"✗ Model loading failed: {e}")
           return False
   
   def check_inference():
       """Test basic inference."""
       try:
           import oe
           result = oe.run("microsoft/DialoGPT-small", "Hello")
           print(f"✓ Inference successful: {result[:50]}...")
           return True
       except Exception as e:
           print(f"✗ Inference failed: {e}")
           return False
   
   def main():
       print("OpenVINO-Easy Diagnostic Tool")
       print("=" * 40)
       
       checks = [
           ("Installation", check_installation),
           ("Device Detection", check_devices),
           ("Model Loading", check_model_loading),
           ("Inference", check_inference),
       ]
       
       results = []
       for name, check_func in checks:
           print(f"\n{name}:")
           success = check_func()
           results.append((name, success))
       
       print("\n" + "=" * 40)
       print("Summary:")
       for name, success in results:
           status = "PASS" if success else "FAIL"
           print(f"{name}: {status}")
       
       if all(result[1] for result in results):
           print("\n✓ All checks passed - OpenVINO-Easy is working correctly!")
       else:
           print("\n! Some checks failed - see troubleshooting guide above")
   
   if __name__ == "__main__":
       main()

Save this as ``diagnostic.py`` and run:

.. code-block:: bash

   python diagnostic.py

This will help identify where issues are occurring in your setup.