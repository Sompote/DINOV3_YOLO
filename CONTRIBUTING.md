# Contributing to YOLOv13-DINO

Thank you for your interest in contributing to YOLOv13 with DINO Vision Transformer Backbones! 🎉

## 🚀 Quick Contributing Guide

### 🐛 Reporting Issues

1. **Search existing issues** first to avoid duplicates
2. **Use issue templates** when available
3. **Include detailed information**:
   - Environment details (Python version, PyTorch version, CUDA version)
   - Model configuration used
   - Steps to reproduce
   - Expected vs actual behavior
   - Error logs and stack traces

### 💡 Suggesting Enhancements

1. **Check existing feature requests** first
2. **Describe the enhancement** clearly
3. **Explain the use case** and motivation
4. **Consider implementation complexity**

### 🔧 Code Contributions

#### Getting Started

```bash
# Fork and clone the repository
git clone https://github.com/your-username/yolov13-dino.git
cd yolov13-dino

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

#### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow existing code style and patterns
   - Add docstrings to new functions/classes
   - Include type hints where appropriate

3. **Test your changes**:
   ```bash
   # Test architecture loading
   python -c "from ultralytics import YOLO; model = YOLO('ultralytics/cfg/models/v13/yolov13-dino3.yaml'); print('✅ Architecture test passed')"
   
   # Test specific functionality
   python -c "from ultralytics import YOLO; model = YOLO('your-config.yaml')"
   
   # Run inference test
   python dino_inference.py --weights test.pt --source test_image.jpg
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

### 📝 Code Style Guidelines

#### Python Code Style

- **PEP 8 compliance**: Use consistent formatting
- **Docstrings**: Google-style docstrings for functions and classes
- **Type hints**: Include type hints for function parameters and returns
- **Variable naming**: Use descriptive names

**Example**:
```python
def create_dino_backbone(
    model_name: str,
    freeze_weights: bool = True,
    output_channels: int = 512
) -> nn.Module:
    """
    Create a DINO backbone module.
    
    Args:
        model_name: Name of the DINO model variant
        freeze_weights: Whether to freeze backbone weights
        output_channels: Number of output channels
        
    Returns:
        Configured DINO backbone module
    """
    # Implementation here
    pass
```

#### YAML Configuration Style

```yaml
# Model configuration
backbone:
  # YOLOv13 layers
  - [-1, 1, Conv, [32, 3, 1]]
  
  # DINO3 enhancement
  - [-1, 1, DINO3Backbone, ['dinov3_vitb16', True, 512]]
```

### 🧪 Testing Guidelines

#### Required Tests

1. **Architecture Loading Tests**:
   ```bash
   python -c "from ultralytics import YOLO; model = YOLO('ultralytics/cfg/models/v13/yolov13-dino3.yaml'); print('✅ Architecture test passed')"
   ```

2. **Model Creation Tests**:
   ```python
   # Test in your code
   from ultralytics import YOLO
   model = YOLO('your-new-config.yaml')
   assert model is not None
   ```

3. **Inference Tests**:
   ```bash
   python dino_inference.py --weights model.pt --source test_image.jpg
   ```

#### Adding New Tests

When adding new functionality:

1. **Create test files** in the appropriate directory
2. **Use descriptive test names**: `test_dino3_multi_scale_loading()`
3. **Test edge cases** and error conditions
4. **Include performance benchmarks** for significant changes

### 📚 Documentation

#### Code Documentation

- **Docstrings**: Every public function/class needs docstrings
- **Comments**: Explain complex logic and design decisions
- **Type hints**: Use appropriate type annotations

#### README Updates

When adding new features:

1. **Update feature list** in README.md
2. **Add usage examples** in appropriate sections
3. **Update model zoo** if adding new architectures
4. **Include performance benchmarks** if available

### 🏗️ Architecture Contributions

#### Adding New DINO Variants

1. **Update DINO3Backbone class** in `ultralytics/nn/modules/block.py`:
   ```python
   self.dinov3_specs = {
       # Existing variants...
       'your_new_variant': {
           'params': 100,  # Million parameters
           'embed_dim': 768,
           'patch_size': 16,
           'type': 'vit'
       }
   }
   ```

2. **Create YAML configuration**:
   ```yaml
   # ultralytics/cfg/models/v13/yolov13-your-variant.yaml
   ```

3. **Update training scripts** to include new variant:
   ```python
   # In train_dino2.py
   choices=['dinov3_vitb16', 'your_new_variant', ...]
   ```

4. **Test the new architecture**:
   ```bash
   python -c "from ultralytics import YOLO; model = YOLO('ultralytics/cfg/models/v13/your-new-architecture.yaml'); print('✅ New architecture test passed')"
   ```

#### Adding New Multi-Scale Configurations

1. **Design the architecture** following existing patterns
2. **Create YAML configuration** with appropriate layer connections
3. **Update documentation** with performance characteristics
4. **Add to model zoo table** in README.md

### 🔄 Pull Request Process

#### PR Checklist

- [ ] **Tests pass**: All existing tests continue to work
- [ ] **New tests added**: For new functionality
- [ ] **Documentation updated**: README.md, docstrings, etc.
- [ ] **Code style**: Follows project conventions
- [ ] **Performance**: No significant performance regression
- [ ] **Backwards compatibility**: Existing code still works

#### PR Description Template

```markdown
## 🎯 What does this PR do?

Brief description of the changes.

## 🔧 Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement

## 🧪 Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing performed

## 📋 Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated

## 📊 Performance Impact

- Memory usage: [No change/+X MB/-X MB]
- Training speed: [No change/+X%/-X%]
- Inference speed: [No change/+X ms/-X ms]
```

### 🎯 Areas for Contribution

#### High Priority

1. **Performance optimization**:
   - Memory usage reduction
   - Training speed improvements
   - Inference optimization

2. **New model architectures**:
   - Additional DINO variants
   - Novel multi-scale strategies
   - Hybrid architectures

3. **Documentation improvements**:
   - Tutorial notebooks
   - Video guides
   - API documentation

#### Medium Priority

1. **Export formats**:
   - ONNX optimization
   - TensorRT support
   - Mobile deployment

2. **Training improvements**:
   - Advanced augmentation strategies
   - Learning rate scheduling
   - Multi-GPU optimization

3. **Evaluation tools**:
   - Benchmark scripts
   - Visualization tools
   - Performance profiling

### 🛠️ Development Environment

#### Recommended Setup

```bash
# Python environment
Python 3.9+
PyTorch 2.0+
CUDA 11.8+

# Development tools
git
pre-commit
black (code formatting)
flake8 (linting)
mypy (type checking)
```

#### IDE Configuration

**VS Code** settings:
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"]
}
```

### 📞 Getting Help

- **GitHub Discussions**: For questions and ideas
- **GitHub Issues**: For bug reports
- **Email**: For private inquiries

### 🏆 Recognition

Contributors will be:
- **Listed in CONTRIBUTORS.md**
- **Mentioned in release notes**
- **Featured in documentation** (for significant contributions)

---

Thank you for contributing to YOLOv13-DINO! Your contributions help make object detection better for everyone. 🙏