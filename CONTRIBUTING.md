# Contributing to OpenVINO-Easy

Thank you for your interest in contributing to OpenVINO-Easy! This project simplifies Intel OpenVINO usage.

## Quick Start for Contributors

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/openvino-easy.git
   cd openvino-easy
   ```

2. **Set up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate   # Windows
   
   # Install in development mode
   pip install -e ".[full,dev]"
   ```

3. **Run Tests**
   ```bash
   # Quick test suite
   python tests/test_runner.py --mode fast
   
   # Full test suite
   python tests/test_runner.py --mode full
   ```

## Development Workflow

### Setting Up Your Environment

**Prerequisites:**
- Python 3.8-3.12
- Git
- Intel OpenVINO toolkit compatibility
- Optional: Intel GPU drivers for GPU testing

**Development Dependencies:**
```bash
pip install -e ".[dev]"  # Includes: pytest, black, flake8, mypy, pre-commit
```

**Pre-commit Hooks:**
```bash
pre-commit install
```

This automatically runs code formatting and linting on commit.

### Code Standards

**Code Style:**
- **Black** for formatting: `black .`
- **Flake8** for linting: `flake8 oe/`
- **Type hints** required for public APIs
- **Docstrings** for all public functions (Google style)

**Example:**
```python
def load_model(model_path: str, device: str = "auto") -> Pipeline:
    """Load a model for inference.
    
    Args:
        model_path: Path to model file or HuggingFace model name.
        device: Target device ('CPU', 'GPU', 'NPU', or 'auto').
        
    Returns:
        Configured inference pipeline.
        
    Raises:
        ModelLoadError: If model loading fails.
        DeviceNotFoundError: If specified device unavailable.
    """
    pass
```

**Architecture Principles:**
- **Simple public API** - hide OpenVINO complexity
- **Device abstraction** - seamless CPU/GPU/NPU switching
- **Error handling** - descriptive errors with suggestions
- **Performance first** - optimize for production use
- **Backward compatibility** - maintain API stability

### Testing Guidelines

**Test Structure:**
```
tests/
├── test_core.py          # Core functionality
├── test_loader.py        # Model loading
├── test_runtime.py       # Inference runtime
├── test_benchmark.py     # Performance testing
├── test_cli.py          # Command-line interface
├── test_audio.py        # Audio processing
├── test_cross_platform.py # Platform compatibility
└── test_e2e_real_models.py # End-to-end with real models
```

**Test Categories:**
- **Unit tests**: Fast, isolated component testing
- **Integration tests**: Multi-component interaction
- **End-to-end tests**: Real model workflows
- **Performance tests**: Benchmark regressions

**Test Markers:**
```python
import pytest

@pytest.mark.slow
def test_large_model_loading():
    """Tests that take >10 seconds"""
    pass

@pytest.mark.integration  
def test_gpu_cpu_consistency():
    """Tests requiring multiple devices"""
    pass

@pytest.mark.audio
def test_audio_preprocessing():
    """Audio-specific functionality"""
    pass
```

**Running Tests:**
```bash
# Fast tests only (CI-friendly)
python tests/test_runner.py --mode fast

# All tests except slow ones
python tests/test_runner.py --mode full

# Specific categories
python tests/test_runner.py --mode unit
python tests/test_runner.py --mode integration
python tests/test_runner.py --mode audio

# With coverage
python tests/test_runner.py --mode coverage
```

**Writing Tests:**
```python
def test_model_loading():
    """Test basic model loading functionality."""
    # Arrange
    model_name = "microsoft/DialoGPT-small"
    
    # Act
    model = oe.load(model_name)
    
    # Assert
    assert model is not None
    assert model.device in oe.list_devices()
    
    # Test inference
    result = model("Hello")
    assert isinstance(result, str)
    assert len(result) > 0
```

### Documentation

**Documentation Standards:**
- **API docs**: Inline docstrings for all public functions
- **User guides**: Step-by-step tutorials in `docs/`
- **Examples**: Working code in `examples/`
- **README**: Keep updated with new features

**Building Documentation:**
```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build Sphinx documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

**Documentation Style:**
- Clear, concise language
- Code examples that work
- Links to related functions/concepts
- Troubleshooting for common issues

### Submitting Changes

**Branch Naming:**
```bash
git checkout -b feature/model-caching
git checkout -b fix/gpu-memory-leak
git checkout -b docs/api-reference-update
```

**Commit Messages:**
Use [Conventional Commits](https://www.conventionalcommits.org/):
```
feat: add model caching for faster loading
fix: resolve GPU memory leak in batch processing
docs: update API reference for Pipeline class
test: add integration tests for NPU device
```

**Pull Request Process:**

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow code standards
   - Add/update tests
   - Update documentation

3. **Run Full Test Suite**
   ```bash
   python tests/test_runner.py --mode fast
   # Ensure all tests pass
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

**Pull Request Template:**
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] Added/updated unit tests
- [ ] Added/updated integration tests
- [ ] Manual testing completed
- [ ] All existing tests pass

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes without version bump
```

## Contributing Areas

### High Priority

**Core Functionality:**
- Model format support (ONNX, TensorFlow, PyTorch)
- Device optimization (CPU, GPU, NPU)
- Performance improvements
- Memory optimization

**User Experience:**
- Better error messages
- Installation improvements
- Documentation enhancements
- Example applications

**Platform Support:**
- Windows compatibility
- Linux optimization
- macOS support
- Docker images

### Medium Priority

**Advanced Features:**
- Model quantization improvements
- Batch processing optimization
- Streaming inference
- Model serving capabilities

**Developer Tools:**
- Debugging utilities
- Performance profiling
- Model conversion helpers
- Deployment tools

### Specialized Areas

**Audio Processing:**
- Audio model support
- Preprocessing pipelines
- Real-time audio processing

**Computer Vision:**
- Image preprocessing
- Video processing
- Multi-modal models

**Natural Language:**
- Text generation optimization
- Tokenization improvements
- Language model fine-tuning

## Issue Guidelines

### Reporting Bugs

**Bug Report Template:**
```markdown
**Describe the Bug**
Clear description of the issue.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g. Windows 11, Ubuntu 20.04]
- Python version: [e.g. 3.11.0]
- OpenVINO-Easy version: [e.g. 1.0.0]
- Hardware: [e.g. Intel i7-12700K, Intel Arc A770]

**Additional Context**
Add any other context about the problem.
```

**Include System Information:**
```python
import oe
print(oe.system_info())
```

### Feature Requests

**Feature Request Template:**
```markdown
**Feature Description**
Clear description of the desired feature.

**Use Case**
Explain why this feature would be useful.

**Proposed Solution**
If you have ideas for implementation.

**Alternatives Considered**
Any alternative solutions you've considered.

**Additional Context**
Any other context or screenshots.
```

### Performance Issues

For performance-related issues:
1. Include benchmark results
2. Specify hardware configuration
3. Compare with expected performance
4. Include profiling data if available

## Development Guidelines

### API Design Principles

**Simplicity:**
```python
# Good - simple, intuitive
result = oe.run("microsoft/DialoGPT-medium", "Hello")

# Avoid - complex, requires deep knowledge
core = ov.Core()
model = core.read_model("model.xml")
compiled_model = core.compile_model(model, "GPU")
# ... many more steps
```

**Consistency:**
```python
# Consistent device parameter across functions
model = oe.load("model", device="GPU")
result = oe.benchmark("model", device="GPU")
```

**Error Handling:**
```python
# Descriptive errors with solutions
try:
    model = oe.load("model", device="NPU")
except oe.DeviceNotFoundError as e:
    print(f"NPU not available: {e}")
    print("Suggestion: Install Intel NPU drivers or use device='auto'")
```

### Performance Considerations

**Memory Management:**
- Clean up resources in `__del__` methods
- Provide explicit cleanup methods
- Monitor memory usage in tests

**Compilation Caching:**
- Cache compiled models when possible
- Provide cache management utilities
- Clear documentation on cache behavior

**Device Optimization:**
- Optimize for target hardware
- Provide device-specific hints
- Fall back gracefully when devices unavailable

### Backward Compatibility

**API Stability:**
- Don't break existing function signatures
- Deprecate before removing features
- Use semantic versioning

**Migration Path:**
```python
# When changing APIs, provide migration path
def old_function():
    warnings.warn("old_function is deprecated, use new_function", 
                  DeprecationWarning)
    return new_function()
```

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):
- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

### Release Checklist

**Pre-release:**
- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in `pyproject.toml`
- [ ] Examples tested with new version

**Release:**
- [ ] Create release tag
- [ ] Build and test distribution packages
- [ ] Upload to PyPI
- [ ] Create GitHub release with notes

**Post-release:**
- [ ] Update documentation website
- [ ] Announce on relevant channels
- [ ] Monitor for issues

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- **Be respectful** in all interactions
- **Be constructive** when providing feedback
- **Be patient** with new contributors
- **Be inclusive** of different perspectives and experiences

### Communication Channels

**GitHub Issues:**
- Bug reports
- Feature requests
- Technical discussions

**GitHub Discussions:**
- General questions
- Usage help
- Community showcases

**Pull Request Reviews:**
- Focus on code quality and functionality
- Provide constructive feedback
- Acknowledge good work

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- GitHub contributor graphs
- Community showcases for notable applications

## Getting Help

**For Contributors:**
- Check existing issues and PRs
- Ask in GitHub Discussions
- Review documentation and examples
- Reach out to maintainers for guidance

**For Code Review:**
- Request review from relevant maintainers
- Address feedback promptly and respectfully
- Ask questions if feedback is unclear
- Make requested changes or explain why not

**For Technical Issues:**
- Use the diagnostic script in troubleshooting guide
- Provide complete system information
- Include minimal reproducible example
- Check if issue exists in latest version

## Resources

**Documentation:**
- [API Reference](docs/api/)
- [Getting Started Guide](docs/getting_started.md)
- [Performance Tuning](docs/performance_tuning.md)
- [Production Deployment](docs/production_deployment.md)

**Examples:**
- [Basic Usage](examples/)
- [Advanced Workflows](examples/)
- [Production Patterns](examples/)

**External Resources:**
- [Intel OpenVINO Documentation](https://docs.openvino.ai/)
- [OpenVINO Model Zoo](https://github.com/openvinotoolkit/open_model_zoo)
- [HuggingFace Model Hub](https://huggingface.co/models)

Thank you for contributing to OpenVINO-Easy!