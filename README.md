# OpenVINO-Easy 🚀

**Framework-agnostic Python wrapper for OpenVINO 2025**

Load and run AI models with three functions:
```python
import oe

oe.load("runwayml/stable-diffusion-v1-5")   # auto-download & convert
img = oe.infer("a neon cyber-city at night")       # chooses NPU>GPU>CPU  
stats = oe.benchmark()                              # JSON perf report
```

## 🎯 Installation

**Pick the variant that matches your hardware:**

```bash
# CPU-only (40MB wheel, fastest install)
pip install "openvino-easy[cpu]"
# or
pip install "openvino-easy[runtime]"

# Intel® Arc/Xe GPU support
pip install "openvino-easy[gpu]"

# Intel® NPU support (Arrow Lake/Lunar Lake with FP16-NF4)
pip install "openvino-easy[npu]"

# With INT8 quantization support
pip install "openvino-easy[quant]"

# Audio model support (Whisper, TTS)
pip install "openvino-easy[audio]"

# Full development environment (OpenVINO, NNCF, optimum ~1GB)
pip install "openvino-easy[full]"

# Everything (for development)
pip install "openvino-easy[all]"
```

### 🩺 Installation Troubleshooting

**Something not working?** Run the doctor:

```bash
# Comprehensive diagnostics
oe doctor

# Get fix suggestions for specific hardware
oe doctor --fix gpu
oe doctor --fix npu

# JSON output for CI systems
oe doctor --json

# Check device status
oe devices
```

**Common issues:**

| Problem | Solution |
|---------|----------|
| `ImportError: OpenVINO runtime not found` | Install with hardware extras: `pip install "openvino-easy[cpu]"` |
| NPU detected but not functional | Install Intel NPU drivers from intel.com |
| GPU detected but not functional | Install Intel GPU drivers (`intel-opencl-icd` on Linux) |
| `NNCF not available` for INT8 quantization | Install quantization support: `pip install "openvino-easy[quant]"` |
| FP16-NF4 not supported | Requires Arrow Lake/Lunar Lake NPU with OpenVINO 2025.2+ |
| Version warnings | Upgrade OpenVINO: `pip install --upgrade "openvino>=2025.2,<2026.0"` |
| **PyTorch model (.pt/.pth) not loading** | **Convert to ONNX first:** `torch.onnx.export(model, dummy_input, "model.onnx")` then `oe.load("model.onnx")` |
| **"Native PyTorch model conversion failed"** | **Upload to Hugging Face Hub** with config.json or **use ONNX format** for best compatibility |

### 📦 What Each Variant Includes

| Variant | OpenVINO Package | Size | Best For |
|---------|------------------|------|----------|
| `[cpu]` / `[runtime]` | `openvino` runtime | ~40MB | Production deployments, CPU-only inference |
| `[gpu]` | `openvino` runtime | ~40MB | Intel GPU acceleration |
| `[npu]` | `openvino` runtime | ~40MB | Intel NPU acceleration |
| `[quant]` | `openvino` + NNCF | ~440MB | INT8 quantization support |
| `[audio]` | `openvino` + librosa | ~100MB | Audio models (Whisper, TTS) |
| `[full]` | `openvino` + NNCF + optimum | ~1GB | Development, model optimization, research |

## ⚡ Quick Start

### Basic Usage

```python
import oe

# Load any model (Hugging Face, ONNX, or OpenVINO IR)
oe.load("microsoft/DialoGPT-medium")

# Run inference (automatic tokenization for text models)
response = oe.infer("Hello, how are you?")
print(response)  # "I'm doing well, thank you for asking!"

# Benchmark performance
stats = oe.benchmark()
print(f"Average latency: {stats['avg_latency_ms']:.2f}ms")
print(f"Throughput: {stats['throughput_fps']:.1f} FPS")

# Explicitly free memory when done
oe.unload()
```

### Advanced Usage

```python
# Specify device preference and precision
oe.load(
    "runwayml/stable-diffusion-v1-5",
    device_preference=["NPU", "GPU", "CPU"],  # Try NPU first, fallback to GPU, then CPU
    dtype="fp16-nf4"  # New FP16-NF4 precision for Arrow Lake/Lunar Lake NPUs
)

# Generate image
image = oe.infer(
    "a serene mountain landscape at sunset",
    num_inference_steps=20,
    guidance_scale=7.5
)

# Get detailed model info
info = oe.get_info()
print(f"Running on: {info['device']}")
print(f"Model type: {info['dtype']}")
print(f"Quantized: {info['quantized']}")

# Context manager for automatic cleanup
with oe.load("runwayml/stable-diffusion-v1-5") as pipe:
    image = pipe.infer("a serene mountain landscape")
    # Model automatically unloaded when exiting context
```

### Audio Models

```python
# Speech-to-text with Whisper
oe.load("openai/whisper-base")
transcription = oe.infer("path/to/audio.wav")
print(transcription)  # "Hello, this is the transcribed audio"

# Text-to-speech (OpenVINO 2025.2+)
oe.load("microsoft/speecht5_tts")
audio = oe.infer("Hello world!")
# Save or play the generated audio
```

### Memory Management

OpenVINO-Easy provides flexible memory management for production applications:

```python
# Method 1: Explicit unload
oe.load("large-model")
result = oe.infer(data)
oe.unload()  # Free memory immediately

# Method 2: Context manager (recommended)
with oe.load("large-model") as pipe:
    result = pipe.infer(data)
    # Model automatically unloaded when exiting

# Method 3: Multiple model switching  
oe.load("text-model")
result1 = oe.infer("Hello world")
oe.unload()

oe.load("image-model")
result2 = oe.infer(image_data)
oe.unload()

# Check if model is still loaded
if oe.is_loaded():
    result = oe.infer(data)
else:
    print("Model has been unloaded")
```

### Model Management & Discovery

OpenVINO-Easy provides comprehensive model management capabilities:

```python
# Search for models on Hugging Face Hub
results = oe.models.search("stable diffusion", limit=5, model_type="image")
for model in results:
    print(f"{model['id']}: {model['downloads']:,} downloads")

# Get detailed model information
info = oe.models.info("microsoft/DialoGPT-medium")
print(f"Local: {info['local']}, Remote: {info['remote']}")
print(f"Requirements: {info['requirements']['min_memory_mb']} MB")

# Install models without loading them
result = oe.models.install("runwayml/stable-diffusion-v1-5", dtype="fp16")
print(f"Installed: {result['size_mb']:.1f} MB")

# Validate model integrity
results = oe.models.validate()
print(f"Validation: {results['passed']}/{results['validated']} models valid")

# Benchmark all installed models
results = oe.models.benchmark_all()
best = results['summary']['fastest_model']
print(f"Fastest model: {best['id']} ({best['fps']:.1f} FPS)")
```

### Model Storage & Cache Management

OpenVINO-Easy uses a clean, Ollama-style directory structure:

```python
# Check where models are stored
print("Models directory:", oe.models.dir())
# Windows: C:\Users\username\AppData\Local\openvino-easy\models\
# Linux/Mac: ~/.openvino-easy/models/

# List all cached models
models_list = oe.models.list()
for model in models_list:
    print(f"{model['name']}: {model['size_mb']:.1f} MB")

# Check cache usage
cache_info = oe.cache.size()
print(f"Total cache size: {cache_info['total_size_mb']:.1f} MB")
print(f"Models: {cache_info['model_count']}")

# Clean up temporary files only (keeps models)
oe.cache.clear()

# Remove a specific model (exact name required for safety)
result = oe.models.remove("microsoft--DialoGPT-medium--fp16--a1b2c3d4")
print(result)  # Shows what was removed

# Clear everything including models (requires confirmation)
result = oe.models.clear()  # Shows safety warning, requires confirm=False
result = oe.models.clear(confirm=False)  # Actually performs deletion

# Clear temp cache only (safe)
oe.cache.clear()

# Clear both temp cache and models (dangerous, requires confirmation)
oe.cache.clear(models=True)  # Shows safety warning
oe.cache.clear(models=True, confirm=False)  # Actually performs deletion
```

**Directory Structure:**
```
~/.openvino-easy/                    # Linux/Mac
C:\Users\user\AppData\Local\openvino-easy\  # Windows
├── models/                          # Downloaded/converted models (permanent)
│   ├── microsoft--DialoGPT-medium--fp16--a1b2c3d4/
│   └── openai--whisper-base--int8--e5f6g7h8/
├── cache/                           # Temporary conversion files
└── config/                          # User settings
```

**Environment Override:**
```bash
# Custom models directory
export OE_MODELS_DIR="/shared/ai-models"
# or
OE_MODELS_DIR="/shared/ai-models" python app.py
```

## 🔧 Command Line Interface

```bash
# Text inference
oe run "microsoft/DialoGPT-medium" --prompt "Hello there"

# Audio inference (speech-to-text)
oe run "openai/whisper-base" --input-file "audio.wav"

# Image generation
oe run "runwayml/stable-diffusion-v1-5" --prompt "a beautiful sunset"

# Benchmark with latest NPU precision
oe bench "runwayml/stable-diffusion-v1-5" --dtype fp16-nf4

# System diagnostics
oe doctor

# List available devices
oe devices

# Enhanced NPU diagnostics (Arrow Lake/Lunar Lake detection)
oe npu-doctor

# Cache management
oe cache list              # List cached models
oe cache size              # Show cache usage
oe cache remove <model>    # Remove specific model (with confirmation)
oe cache clear             # Clear temp cache only (safe)
oe cache clear --models    # Clear all models (DANGEROUS - requires confirmation)
oe cache clear --models --force  # Override safety (VERY DANGEROUS)

# Advanced model management
oe models search "stable diffusion" --limit 5  # Search HuggingFace Hub
oe models info microsoft/DialoGPT-medium       # Get model details
oe models install runwayml/stable-diffusion-v1-5 --dtype fp16  # Install model
oe models validate         # Validate all models
oe models benchmark        # Benchmark all installed models
```

## 🏗️ Architecture

OpenVINO-Easy wraps OpenVINO's API:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Your Code     │    │  OpenVINO-Easy   │    │   OpenVINO      │
│                 │    │                  │    │                 │
│ oe.load(...)    │───▶│ • Model Loading  │───▶│ • IR Conversion │
│ oe.infer(...)   │    │ • Device Select  │    │ • Compilation   │
│ oe.benchmark()  │    │ • Preprocessing  │    │ • Inference     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Features

- **Device Selection**: Chooses NPU → GPU → CPU based on availability
- **Model Loading**: Supports Hugging Face, ONNX, and OpenVINO IR formats
- **Conversion**: Converts models to OpenVINO IR format
- **INT8 Quantization**: Quantization with NNCF for faster inference
- **Benchmarking**: Performance metrics and timing
- **Caching**: SHA-256 based model caching for fast re-loading
- **Memory Management**: Explicit unload() and context manager support
- **Hardware Diagnostics**: Tools for troubleshooting device issues

## 🤖 Supported Models

### Text Models
- **Conversational**: DialoGPT, BlenderBot, ChatGLM
- **Text Generation**: GPT-2, GPT-J, OPT, BLOOM  
- **Question Answering**: BERT, RoBERTa, DeBERTa
- **Text Classification**: DistilBERT, ALBERT

### Vision Models
- **Image Generation**: Stable Diffusion, DALL-E 2
- **Object Detection**: YOLO, SSD, RetinaNet
- **Image Classification**: ResNet, EfficientNet, Vision Transformer
- **Segmentation**: U-Net, DeepLab, Mask R-CNN

### Audio Models
- **Speech Recognition**: Whisper, Wav2Vec2, WavLM
- **Text-to-Speech**: SpeechT5, Bark (coming soon)
- **Audio Classification**: Hubert, Audio Transformers

### Multimodal Models
- **Vision-Language**: CLIP, BLIP, LLaVA
- **Image Captioning**: BLIP-2, GIT, OFA

## 🚀 Performance

Performance benchmarks:

| Model | Hardware | Throughput | Latency |
|-------|----------|------------|---------|
| Stable Diffusion 1.5 | Intel Core Ultra 7 Lunar Lake (NPU) | **2.3+ img/s** | 420ms |
| Stable Diffusion 1.5 | Intel Core Ultra 7 Arrow Lake (NPU) | **2.2+ img/s** | 450ms |
| Stable Diffusion 1.5 | Intel Core Ultra 7 (1st gen NPU) | 1.8 img/s | 556ms |
| Stable Diffusion 1.5 | Intel Arc A770 (GPU) | 1.6 img/s | 625ms |
| Stable Diffusion 1.5 | Intel Core i7-13700K (CPU) | 0.4 img/s | 2.5s |
| DialoGPT-medium | Intel Core Ultra 7 Lunar Lake (NPU) | **50+ tok/s** | 20ms |
| DialoGPT-medium | Intel Core Ultra 7 Arrow Lake (NPU) | **48+ tok/s** | 21ms |
| DialoGPT-medium | Intel Core Ultra 7 (1st gen NPU) | 40 tok/s | 25ms |
| DialoGPT-medium | Intel Arc A770 (GPU) | 38 tok/s | 26ms |
| DialoGPT-medium | Intel Core i7-13700K (CPU) | 12 tok/s | 83ms |

*Benchmarks with FP16-NF4 precision on Arrow Lake/Lunar Lake NPUs (OpenVINO 2025.2+)*

## 🔬 Text Processing Details

OpenVINO-Easy handles text preprocessing automatically:

```python
# For text models, tokenization is automatic
pipe = oe.load("microsoft/DialoGPT-medium")

# Multiple input formats supported:
response = pipe.infer("Hello!")                    # String input
response = pipe.infer(["Hello!", "How are you?"])  # Batch input
response = pipe.infer({"text": "Hello!"})          # Dict input
```

**Tokenization Strategy:**
1. **HuggingFace Models**: Uses `transformers.AutoTokenizer` with model-specific settings
2. **ONNX Models**: Attempts to infer tokenizer from model metadata
3. **OpenVINO IR**: Falls back to basic text preprocessing
4. **Custom Models**: Provides hooks for custom tokenization

## 🧪 Development & Testing

### **Modern Python Packaging (Recommended)**

```bash
# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Or install specific extras for testing
pip install -e ".[full,dev]"  # Full OpenVINO + dev tools
```

### **Comprehensive Testing Framework**

OpenVINO-Easy includes a robust testing framework with multiple test categories:

```bash
# Quick tests (unit tests only, fast)
python test_runner.py --mode fast

# All tests except slow ones
python test_runner.py --mode full

# Integration tests with real OpenVINO models
python test_runner.py --mode integration

# End-to-end tests with real HuggingFace models (requires internet)
python test_runner.py --mode e2e

# Performance regression testing
python test_runner.py --mode performance

# Model compatibility validation
pytest tests/test_model_compatibility.py -v

# Cache management and safety tests
pytest tests/test_model_management.py -v

# CLI functionality tests
pytest tests/test_cli_models.py -v

# Run with coverage
python test_runner.py --mode coverage
```

### **Quality Assurance Features**

#### **Performance Regression Testing**
```python
# Automated performance baselines
from tests.test_performance_regression_enhanced import PerformanceRegression

tester = PerformanceRegression()
test = PerformanceTest(
    model_id="microsoft/DialoGPT-medium",
    tolerance_percent=15.0  # Allow 15% regression
)

results = tester.run_performance_test(test)
if results['regressions']:
    print("Performance regressions detected!")
```

#### **Model Compatibility Validation**
```python
# Automated compatibility testing across devices/precisions
from tests.test_model_compatibility import ModelCompatibilityValidator

validator = ModelCompatibilityValidator()
result = validator.validate_model_compatibility("runwayml/stable-diffusion-v1-5")

if not result['overall_compatible']:
    print(f"Compatibility issues: {result['issues']}")
```

#### **Enhanced Error Recovery**
```python
# Automatic device fallback and retry logic
oe.load(
    "microsoft/DialoGPT-medium",
    device_preference=["NPU", "GPU", "CPU"],
    retry_on_failure=True,
    fallback_device="CPU"
)
# Automatically tries NPU -> GPU -> CPU -> CPU with default config
```

### **Test Categories**

| Test Type | Command | Purpose |
|-----------|---------|----------|
| **Unit Tests** | `pytest tests/ -m "not slow and not integration"` | Core functionality |
| **Integration Tests** | `pytest tests/ -m "integration"` | Real model loading |
| **Performance Tests** | `pytest tests/ -m "performance"` | Regression detection |
| **Compatibility Tests** | `pytest tests/ -m "compatibility"` | Device/model validation |
| **End-to-End Tests** | `pytest tests/test_e2e_real_models.py` | Full workflows |
| **CLI Tests** | `pytest tests/test_cli*.py` | Command-line interface |
| **Safety Tests** | `pytest tests/test_model_management.py` | Security validation |

### **Development Workflow**

```bash
# Format code
black oe/ tests/
isort oe/ tests/

# Type checking
mypy oe/

# Run all quality checks
python test_runner.py --mode full
pytest tests/test_model_compatibility.py -x
pytest tests/test_performance_regression_enhanced.py -x
```

## 📚 Examples

Check out the `examples/` directory:

- **[Stable Diffusion Notebook](examples/stable_diffusion.ipynb)**: Image generation with automatic optimization
- **Text Generation**: Conversational AI with DialoGPT
- **ONNX Models**: Loading and running ONNX models
- **Custom Models**: Integrating your own models

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Intel OpenVINO Team for the inference engine
- Hugging Face for the transformers ecosystem  
- ONNX Community for the model format standards 