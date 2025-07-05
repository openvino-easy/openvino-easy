# OpenVINO-Easy 🚀

**Framework-agnostic Python wrapper for OpenVINO 2024**

Transform any AI model into a three-function experience:
```python
import openvino_easy as oe

pipe = oe.load("runwayml/stable-diffusion-v1-5")   # auto-download & convert
img  = pipe.infer("a neon cyber-city at night")     # chooses NPU>GPU>CPU
stats = oe.benchmark(pipe)                          # JSON perf report
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

# Intel® Core Ultra NPU support  
pip install "openvino-easy[npu]"

# With INT8 quantization support
pip install "openvino-easy[quant]"

# Full development environment (Model Optimizer, NNCF, samples ~1GB)
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
| Version warnings | Upgrade OpenVINO: `pip install --upgrade "openvino>=2024.0,<2026.0"` |

### 📦 What Each Variant Includes

| Variant | OpenVINO Package | Size | Best For |
|---------|------------------|------|----------|
| `[cpu]` / `[runtime]` | `openvino` runtime | ~40MB | Production deployments, CPU-only inference |
| `[gpu]` | `openvino` runtime | ~40MB | Intel GPU acceleration |
| `[npu]` | `openvino` runtime | ~40MB | Intel NPU acceleration |
| `[quant]` | `openvino` + NNCF | ~440MB | INT8 quantization support |
| `[full]` | `openvino-dev` + NNCF | ~1GB | Development, model optimization, research |

## ⚡ Quick Start

### Basic Usage

```python
import openvino_easy as oe

# Load any model (Hugging Face, ONNX, or OpenVINO IR)
pipe = oe.load("microsoft/DialoGPT-medium")

# Run inference (automatic tokenization for text models)
response = pipe.infer("Hello, how are you?")
print(response)  # "I'm doing well, thank you for asking!"

# Benchmark performance
stats = pipe.benchmark()
print(f"Average latency: {stats['avg_latency_ms']:.2f}ms")
print(f"Throughput: {stats['throughput_fps']:.1f} FPS")
```

### Advanced Usage

```python
# Specify device preference and precision
pipe = oe.load(
    "runwayml/stable-diffusion-v1-5",
    device_preference=["NPU", "GPU", "CPU"],  # Try NPU first, fallback to GPU, then CPU
    dtype="int8"  # Automatic INT8 quantization
)

# Generate image
image = pipe.infer(
    "a serene mountain landscape at sunset",
    num_inference_steps=20,
    guidance_scale=7.5
)

# Get detailed model info
info = pipe.get_info()
print(f"Running on: {info['device']}")
print(f"Model type: {info['dtype']}")
print(f"Quantized: {info['quantized']}")
```

## 🔧 Command Line Interface

```bash
# Run inference
oe run "microsoft/DialoGPT-medium" --prompt "Hello there"

# Benchmark any model
oe bench "runwayml/stable-diffusion-v1-5" --dtype int8

# System diagnostics
oe doctor

# List available devices
oe devices

# NPU-specific diagnostics
oe npu-doctor
```

## 🏗️ Architecture

OpenVINO-Easy provides a clean abstraction over OpenVINO's complexity:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Your Code     │    │  OpenVINO-Easy   │    │   OpenVINO      │
│                 │    │                  │    │                 │
│ oe.load(...)    │───▶│ • Model Loading  │───▶│ • IR Conversion │
│ pipe.infer(...) │    │ • Device Select  │    │ • Compilation   │
│ oe.benchmark()  │    │ • Preprocessing  │    │ • Inference     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Features

- **🎯 Smart Device Selection**: Automatically chooses NPU → GPU → CPU based on availability
- **📦 Universal Model Loading**: Supports Hugging Face, ONNX, and OpenVINO IR formats
- **🔧 Automatic Conversion**: Converts models to OpenVINO IR with optimal settings
- **⚡ INT8 Quantization**: Automatic quantization with NNCF for faster inference
- **📊 Built-in Benchmarking**: Comprehensive performance metrics
- **🛠️ Robust Caching**: SHA-256 based model caching for fast re-loading
- **🔍 Hardware Diagnostics**: Built-in tools for troubleshooting device issues

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

### Multimodal Models
- **Vision-Language**: CLIP, BLIP, LLaVA
- **Image Captioning**: BLIP-2, GIT, OFA

## 🚀 Performance

OpenVINO-Easy automatically optimizes models for your hardware:

| Model | Hardware | Throughput | Latency |
|-------|----------|------------|---------|
| Stable Diffusion 1.5 | Intel Core Ultra 7 (NPU) | 2.3 img/s | 435ms |
| Stable Diffusion 1.5 | Intel Arc A770 (GPU) | 1.8 img/s | 556ms |
| Stable Diffusion 1.5 | Intel Core i7-13700K (CPU) | 0.4 img/s | 2.5s |
| DialoGPT-medium | Intel Core Ultra 7 (NPU) | 45 tok/s | 22ms |
| DialoGPT-medium | Intel Arc A770 (GPU) | 38 tok/s | 26ms |
| DialoGPT-medium | Intel Core i7-13700K (CPU) | 12 tok/s | 83ms |

*Benchmarks run with INT8 quantization on Intel hardware*

## 🔬 Text Processing Details

OpenVINO-Easy handles text preprocessing automatically:

```python
# For text models, tokenization is automatic
pipe = oe.load("microsoft/DialoGPT-medium")

# These all work seamlessly:
response = pipe.infer("Hello!")                    # String input
response = pipe.infer(["Hello!", "How are you?"])  # Batch input
response = pipe.infer({"text": "Hello!"})          # Dict input
```

**Tokenization Strategy:**
1. **HuggingFace Models**: Uses `transformers.AutoTokenizer` with model-specific settings
2. **ONNX Models**: Attempts to infer tokenizer from model metadata
3. **OpenVINO IR**: Falls back to basic text preprocessing
4. **Custom Models**: Provides hooks for custom tokenization

## 🧪 Development

### **Modern Python Packaging (Recommended)**

```bash
# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Or install specific extras for testing
pip install -e ".[full,dev]"  # Full OpenVINO + dev tools
```

### **Legacy/Docker Support**

```bash
# For Docker builds or legacy CI systems
pip install -r requirements.txt

# For Docker with pre-cached layers
pip install -r requirements-docker.txt
```

### **Development Workflow**

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=oe tests/

# Format code
black oe/ tests/
isort oe/ tests/

# Type checking
mypy oe/
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

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Intel OpenVINO Team** for the amazing inference engine
- **Hugging Face** for the transformers ecosystem  
- **ONNX Community** for the model format standards

---

**Made with ❤️ by the OpenVINO-Easy team** 