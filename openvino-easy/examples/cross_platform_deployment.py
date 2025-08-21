#!/usr/bin/env python3
"""
Cross-Platform Deployment Examples

This example demonstrates how to deploy OpenVINO-Easy models across different platforms
and environments, including Windows, Linux, macOS, Docker, and cloud platforms.

Features:
- Platform detection and optimization
- Environment-specific configurations
- Docker containerization
- Cloud deployment patterns
- CI/CD integration examples
- Platform-specific performance tuning

Requirements:
    pip install openvino-easy psutil docker-py kubernetes
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import shutil

try:
    import oe
    import psutil
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Install with: pip install openvino-easy psutil")
    sys.exit(1)

# Optional dependencies for advanced features
try:
    import docker

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    from kubernetes import client, config

    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False


class PlatformDetector:
    """Advanced platform detection and analysis for cross-platform deployment optimization."""

    @staticmethod
    def get_platform_info() -> Dict[str, Any]:
        """Extract comprehensive platform configuration and hardware specifications."""
        info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.architecture()[0],
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_memory_gb": round(
                psutil.virtual_memory().available / (1024**3), 2
            ),
        }

        # Add platform-specific details
        if info["os"] == "Windows":
            info.update(PlatformDetector._get_windows_info())
        elif info["os"] == "Linux":
            info.update(PlatformDetector._get_linux_info())
        elif info["os"] == "Darwin":  # macOS
            info.update(PlatformDetector._get_macos_info())

        return info

    @staticmethod
    def _get_windows_info() -> Dict[str, Any]:
        """Get Windows-specific information."""
        info = {}

        try:
            # Get Windows edition
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
            )
            info["windows_edition"] = winreg.QueryValueEx(key, "ProductName")[0]
            info["windows_build"] = winreg.QueryValueEx(key, "CurrentBuild")[0]
            winreg.CloseKey(key)
        except Exception:
            info["windows_edition"] = "Unknown"
            info["windows_build"] = "Unknown"

        # Check for WSL
        info["wsl_available"] = shutil.which("wsl") is not None

        return info

    @staticmethod
    def _get_linux_info() -> Dict[str, Any]:
        """Get Linux-specific information."""
        info = {}

        # Get distribution info
        try:
            with open("/etc/os-release", "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("ID="):
                        info["distribution"] = line.split("=")[1].strip().strip('"')
                    elif line.startswith("VERSION_ID="):
                        info["version_id"] = line.split("=")[1].strip().strip('"')
        except FileNotFoundError:
            info["distribution"] = "Unknown"
            info["version_id"] = "Unknown"

        # Check for Docker
        info["docker_available"] = shutil.which("docker") is not None

        # Check for container environment
        info["in_container"] = Path("/.dockerenv").exists()

        return info

    @staticmethod
    def _get_macos_info() -> Dict[str, Any]:
        """Get macOS-specific information."""
        info = {}

        try:
            # Get macOS version
            result = subprocess.run(
                ["sw_vers", "-productVersion"], capture_output=True, text=True
            )
            info["macos_version"] = result.stdout.strip()

            # Get hardware info
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
            )
            output = result.stdout

            if "Apple M1" in output or "Apple M2" in output or "Apple M3" in output:
                info["apple_silicon"] = True
                info["architecture"] = "ARM64"
            else:
                info["apple_silicon"] = False
                info["architecture"] = "Intel"

        except Exception:
            info["macos_version"] = "Unknown"
            info["apple_silicon"] = False

        return info


class PlatformOptimizer:
    """Platform optimization engine for deployment across heterogeneous environments."""

    def __init__(self, platform_info: Dict[str, Any]):
        self.platform_info = platform_info
        self.target_platform = platform_info["os"]

    def get_optimal_config(self, model_type: str = "general") -> Dict[str, Any]:
        """Generate optimized deployment configuration based on platform analysis and model characteristics."""
        config = {
            "device_preference": self._get_device_preference(),
            "threading": self._get_threading_config(),
            "memory": self._get_memory_config(),
            "performance": self._get_performance_config(model_type),
            "platform_specific": self._get_platform_specific_config(),
        }

        return config

    def _get_device_preference(self) -> List[str]:
        """Get device preference order based on platform."""
        if self.os_name == "Windows":
            # Windows typically has good Intel GPU support
            return ["NPU", "GPU", "CPU"]
        elif self.os_name == "Linux":
            # Linux has excellent CPU performance and good GPU support
            return ["GPU", "NPU", "CPU"]
        elif self.os_name == "Darwin":  # macOS
            # macOS primarily uses CPU (limited GPU acceleration)
            if self.platform_info.get("apple_silicon"):
                return ["CPU"]  # Apple Silicon is very fast for CPU inference
            else:
                return ["CPU"]  # Intel Macs also mainly use CPU
        else:
            return ["CPU"]

    def _get_threading_config(self) -> Dict[str, Any]:
        """Get optimal threading configuration."""
        cpu_count = self.platform_info["cpu_count"]
        physical_cores = self.platform_info["physical_cores"]

        # Conservative threading to avoid oversubscription
        if self.os_name == "Windows":
            # Windows tends to benefit from fewer threads
            optimal_threads = min(physical_cores, 8)
        elif self.os_name == "Linux":
            # Linux can handle more aggressive threading
            optimal_threads = min(cpu_count, 16)
        elif self.os_name == "Darwin":
            # macOS benefits from conservative threading
            if self.platform_info.get("apple_silicon"):
                # Apple Silicon has efficiency/performance cores
                optimal_threads = physical_cores
            else:
                optimal_threads = min(physical_cores, 8)
        else:
            optimal_threads = physical_cores

        return {
            "cpu_threads": optimal_threads,
            "use_all_cores": cpu_count == optimal_threads,
            "thread_affinity": self.os_name
            == "Linux",  # Linux supports thread affinity
        }

    def _get_memory_config(self) -> Dict[str, Any]:
        """Get optimal memory configuration."""
        total_memory = self.platform_info["memory_gb"]
        available_memory = self.platform_info["available_memory_gb"]

        # Conservative memory allocation
        if total_memory >= 16:
            memory_fraction = 0.6  # Use up to 60% of memory for large systems
        elif total_memory >= 8:
            memory_fraction = 0.4  # More conservative for medium systems
        else:
            memory_fraction = 0.3  # Very conservative for small systems

        return {
            "max_memory_gb": min(
                available_memory * memory_fraction, total_memory * 0.8
            ),
            "enable_memory_optimization": total_memory < 16,
            "cache_models": total_memory >= 8,
        }

    def _get_performance_config(self, model_type: str) -> Dict[str, Any]:
        """Get performance configuration based on model type and platform."""
        config = {
            "precision": "FP32",  # Default
            "batch_size": 1,
            "performance_hint": "THROUGHPUT",
        }

        # Platform-specific optimizations
        if self.os_name == "Windows":
            # Windows Intel GPU driver optimizations
            if model_type == "vision":
                config["precision"] = "FP16"
                config["batch_size"] = 4
        elif self.os_name == "Linux":
            # Linux typically handles higher precision well
            if model_type in ["text", "language"]:
                config["precision"] = "FP32"
                config["performance_hint"] = "LATENCY"
        elif self.os_name == "Darwin":
            # macOS Apple Silicon optimizations
            if self.platform_info.get("apple_silicon"):
                config["precision"] = "FP32"  # Apple Silicon handles FP32 very well
                if model_type == "vision":
                    config["batch_size"] = 2

        return config

    def _get_platform_specific_config(self) -> Dict[str, Any]:
        """Get platform-specific configuration options."""
        config = {}

        if self.os_name == "Windows":
            config.update(
                {
                    "use_intel_extensions": True,
                    "enable_gpu_scheduling": True,
                    "prefer_dedicated_gpu": True,
                }
            )
        elif self.os_name == "Linux":
            config.update(
                {
                    "use_numa_optimization": True,
                    "enable_transparent_hugepages": True,
                    "use_intel_mkl": True,
                }
            )
        elif self.os_name == "Darwin":
            config.update(
                {
                    "use_accelerate_framework": self.platform_info.get(
                        "apple_silicon", False
                    ),
                    "optimize_for_apple_silicon": self.platform_info.get(
                        "apple_silicon", False
                    ),
                }
            )

        return config


class DeploymentManager:
    """Manage deployments across different platforms and environments."""

    def __init__(self):
        self.platform_info = PlatformDetector.get_platform_info()
        self.optimizer = PlatformOptimizer(self.platform_info)

    def create_optimized_pipeline(
        self, model_name: str, model_type: str = "general"
    ) -> oe.Pipeline:
        """Create an optimized pipeline for the current platform."""
        print(
            f"Creating optimized pipeline for {self.platform_info['os']} {self.platform_info['architecture']}"
        )

        # Get optimal configuration
        config = self.optimizer.get_optimal_config(model_type)

        print(f"Optimal device preference: {config['device_preference']}")
        print(f"Threading: {config['threading']['cpu_threads']} threads")
        print(f"Memory limit: {config['memory']['max_memory_gb']:.1f}GB")

        # Load model with optimization
        device = oe.detect_best_device(preference=config["device_preference"])

        pipeline = oe.load(
            model_name,
            device=device,
            precision=config["performance"]["precision"],
            optimize_memory=config["memory"]["enable_memory_optimization"],
            performance_mode=config["performance"]["performance_hint"].lower(),
        )

        print(f"✓ Pipeline created on device: {pipeline.device}")
        return pipeline

    def generate_dockerfile(
        self, model_name: str, base_image: str = "python:3.11-slim"
    ) -> str:
        """Generate optimized Dockerfile for the model."""
        dockerfile_content = f'''# Generated Dockerfile for {model_name}
FROM {base_image}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    wget \\
    ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install OpenVINO-Easy with appropriate extras
RUN pip install --no-cache-dir "openvino-easy[full]"

# Copy application files
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set environment variables for optimization
ENV OE_DEFAULT_DEVICE=auto
ENV OE_OPTIMIZE_MEMORY=true
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python -c "import oe; print('OK')" || exit 1

# Default command
CMD ["python", "app.py"]

# Labels
LABEL maintainer="OpenVINO-Easy"
LABEL model="{model_name}"
LABEL platform="linux/amd64"
'''
        return dockerfile_content

    def generate_docker_compose(
        self, model_name: str, services: List[str] = None
    ) -> str:
        """Generate Docker Compose configuration."""
        services = services or ["inference"]

        compose_content = f"""version: '3.8'

services:
  inference:
    build: .
    container_name: {model_name.replace("/", "_")}_inference
    ports:
      - "8000:8000"
    environment:
      - OE_DEFAULT_DEVICE=auto
      - OE_CACHE_DIR=/app/cache
    volumes:
      - ./models:/app/models
      - ./cache:/app/cache
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Optional: Redis for caching
  cache:
    image: redis:7-alpine
    container_name: {model_name.replace("/", "_")}_cache
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:

networks:
  default:
    name: {model_name.replace("/", "_")}_network
"""
        return compose_content

    def generate_kubernetes_manifests(
        self, model_name: str, replicas: int = 2
    ) -> Dict[str, str]:
        """Generate Kubernetes deployment manifests."""
        app_name = model_name.replace("/", "-").lower()

        # Deployment manifest
        deployment = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {app_name}-deployment
  labels:
    app: {app_name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {app_name}
  template:
    metadata:
      labels:
        app: {app_name}
    spec:
      containers:
      - name: {app_name}
        image: {app_name}:latest
        ports:
        - containerPort: 8000
        env:
        - name: OE_DEFAULT_DEVICE
          value: "auto"
        - name: OE_OPTIMIZE_MEMORY
          value: "true"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: model-cache
          mountPath: /app/cache
      volumes:
      - name: model-cache
        emptyDir: {{}}
      nodeSelector:
        beta.kubernetes.io/arch: amd64
"""

        # Service manifest
        service = f"""apiVersion: v1
kind: Service
metadata:
  name: {app_name}-service
  labels:
    app: {app_name}
spec:
  selector:
    app: {app_name}
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
"""

        # HorizontalPodAutoscaler manifest
        hpa = f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {app_name}-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {app_name}-deployment
  minReplicas: {replicas}
  maxReplicas: {replicas * 5}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""

        return {"deployment.yaml": deployment, "service.yaml": service, "hpa.yaml": hpa}

    def generate_github_actions_workflow(self, model_name: str) -> str:
        """Generate GitHub Actions CI/CD workflow."""
        workflow_content = f"""name: Build and Deploy {model_name}

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{{{ github.repository }}}}

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{{{ matrix.python-version }}}}
      uses: actions/setup-python@v4
      with:
        python-version: ${{{{ matrix.python-version }}}}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install "openvino-easy[full]"
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest tests/ --cov=./ --cov-report=xml

    - name: Test model loading
      run: |
        python -c "import oe; oe.load('{model_name}', device_preference=['CPU']); print('Model loaded successfully'); oe.unload()"

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{{{ env.REGISTRY }}}}
        username: ${{{{ github.actor }}}}
        password: ${{{{ secrets.GITHUB_TOKEN }}}}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{{{ env.REGISTRY }}}}/${{{{ env.IMAGE_NAME }}}}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{{{ steps.meta.outputs.tags }}}}
        labels: ${{{{ steps.meta.outputs.labels }}}}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
        # Add your deployment commands here

    - name: Run integration tests
      run: |
        echo "Running integration tests"
        # Add integration test commands here

    - name: Deploy to production
      if: success()
      run: |
        echo "Deploying to production environment"
        # Add production deployment commands here
"""
        return workflow_content


def demonstrate_platform_detection():
    """Demonstrate platform detection capabilities."""
    print("Platform Detection and Optimization")
    print("=" * 40)

    detector = PlatformDetector()
    platform_info = detector.get_platform_info()

    print("System Information:")
    print(f"  Operating System: {platform_info['os']} {platform_info['os_version']}")
    print(f"  Architecture: {platform_info['architecture']}")
    print(f"  Processor: {platform_info['processor']}")
    print(
        f"  CPU Cores: {platform_info['physical_cores']} physical, {platform_info['cpu_count']} logical"
    )
    print(
        f"  Memory: {platform_info['memory_gb']:.1f}GB total, {platform_info['available_memory_gb']:.1f}GB available"
    )
    print(f"  Python: {platform_info['python_version']}")

    # Platform-specific information
    if platform_info["os"] == "Windows":
        print(f"  Windows Edition: {platform_info.get('windows_edition', 'Unknown')}")
        print(f"  Windows Build: {platform_info.get('windows_build', 'Unknown')}")
        print(f"  WSL Available: {platform_info.get('wsl_available', False)}")
    elif platform_info["os"] == "Linux":
        print(f"  Distribution: {platform_info.get('distribution', 'Unknown')}")
        print(f"  Version: {platform_info.get('version_id', 'Unknown')}")
        print(f"  Docker Available: {platform_info.get('docker_available', False)}")
        print(f"  In Container: {platform_info.get('in_container', False)}")
    elif platform_info["os"] == "Darwin":
        print(f"  macOS Version: {platform_info.get('macos_version', 'Unknown')}")
        print(f"  Apple Silicon: {platform_info.get('apple_silicon', False)}")

    # Generate optimal configuration
    optimizer = PlatformOptimizer(platform_info)
    config = optimizer.get_optimal_config("text")

    print("\nOptimal Configuration:")
    print(f"  Device Preference: {' > '.join(config['device_preference'])}")
    print(f"  CPU Threads: {config['threading']['cpu_threads']}")
    print(f"  Memory Limit: {config['memory']['max_memory_gb']:.1f}GB")
    print(f"  Precision: {config['performance']['precision']}")
    print(f"  Performance Hint: {config['performance']['performance_hint']}")


def demonstrate_docker_generation():
    """Demonstrate Docker configuration generation."""
    print("\nDocker Configuration Generation")
    print("=" * 40)

    manager = DeploymentManager()
    model_name = "microsoft/DialoGPT-medium"

    # Generate Dockerfile
    dockerfile = manager.generate_dockerfile(model_name)
    print("Generated Dockerfile:")
    print("-" * 20)
    print(dockerfile[:500] + "..." if len(dockerfile) > 500 else dockerfile)

    # Generate Docker Compose
    compose = manager.generate_docker_compose(model_name)
    print("\nGenerated docker-compose.yml:")
    print("-" * 20)
    print(compose[:500] + "..." if len(compose) > 500 else compose)

    # Save to files if requested
    save_files = (
        input("\nSave Docker files to current directory? (y/n): ").lower() == "y"
    )

    if save_files:
        with open("Dockerfile", "w") as f:
            f.write(dockerfile)
        with open("docker-compose.yml", "w") as f:
            f.write(compose)
        print("✓ Files saved: Dockerfile, docker-compose.yml")


def demonstrate_kubernetes_generation():
    """Demonstrate Kubernetes manifests generation."""
    print("\nKubernetes Manifests Generation")
    print("=" * 40)

    manager = DeploymentManager()
    model_name = "microsoft/DialoGPT-medium"

    # Generate Kubernetes manifests
    manifests = manager.generate_kubernetes_manifests(model_name, replicas=3)

    print("Generated Kubernetes manifests:")
    for filename, content in manifests.items():
        print(f"\n{filename}:")
        print("-" * 20)
        print(content[:300] + "..." if len(content) > 300 else content)

    # Save to files if requested
    save_files = (
        input("\nSave Kubernetes manifests to k8s/ directory? (y/n): ").lower() == "y"
    )

    if save_files:
        k8s_dir = Path("k8s")
        k8s_dir.mkdir(exist_ok=True)

        for filename, content in manifests.items():
            with open(k8s_dir / filename, "w") as f:
                f.write(content)

        print(f"✓ Files saved to k8s/ directory: {', '.join(manifests.keys())}")


def demonstrate_cicd_generation():
    """Demonstrate CI/CD workflow generation."""
    print("\nCI/CD Workflow Generation")
    print("=" * 40)

    manager = DeploymentManager()
    model_name = "microsoft/DialoGPT-medium"

    # Generate GitHub Actions workflow
    workflow = manager.generate_github_actions_workflow(model_name)

    print("Generated GitHub Actions workflow:")
    print("-" * 20)
    print(workflow[:600] + "..." if len(workflow) > 600 else workflow)

    # Save to file if requested
    save_files = input("\nSave workflow to .github/workflows/? (y/n): ").lower() == "y"

    if save_files:
        workflows_dir = Path(".github/workflows")
        workflows_dir.mkdir(parents=True, exist_ok=True)

        workflow_file = workflows_dir / "deploy.yml"
        with open(workflow_file, "w") as f:
            f.write(workflow)

        print(f"✓ Workflow saved to {workflow_file}")


def demonstrate_optimized_inference():
    """Demonstrate platform-optimized inference."""
    print("\nPlatform-Optimized Inference")
    print("=" * 40)

    manager = DeploymentManager()

    # Use a simple, fast-loading model for demonstration
    model_name = "microsoft/DialoGPT-small"  # Smaller model for faster demo

    try:
        print(f"Creating optimized pipeline for {model_name}...")
        manager.create_optimized_pipeline(model_name, model_type="text")

        # Test inference
        test_input = "Hello, how are you?"
        print(f"\nTesting inference with: '{test_input}'")

        result = oe.infer(test_input)  # NEW API
        print(f"Result: {result}")

        # Benchmark performance
        print("\nRunning performance benchmark...")
        benchmark_results = oe.benchmark(warmup_runs=2, benchmark_runs=5)  # NEW API

        if "avg_latency_ms" in benchmark_results:
            print(f"Average latency: {benchmark_results['avg_latency_ms']:.2f}ms")
            print(f"Throughput: {benchmark_results.get('throughput_fps', 'N/A')} FPS")

        print("✓ Platform-optimized inference completed successfully")

    except Exception as e:
        print(f"Failed to create optimized pipeline: {e}")
        print("This might be due to missing model or network issues")


def demonstrate_environment_detection():
    """Demonstrate environment detection (Docker, Kubernetes, etc.)."""
    print("\nEnvironment Detection")
    print("=" * 40)

    # Check for containerized environment
    in_docker = Path("/.dockerenv").exists()
    in_k8s = os.environ.get("KUBERNETES_SERVICE_HOST") is not None

    print(f"Running in Docker container: {in_docker}")
    print(f"Running in Kubernetes: {in_k8s}")
    print(f"Docker available: {DOCKER_AVAILABLE}")
    print(f"Kubernetes client available: {K8S_AVAILABLE}")

    # Check cloud environment
    cloud_providers = {
        "AWS": "AWS_REGION" in os.environ or "EC2_INSTANCE_ID" in os.environ,
        "Azure": "AZURE_CLIENT_ID" in os.environ,
        "GCP": "GOOGLE_CLOUD_PROJECT" in os.environ,
        "Local": not any(
            [
                "AWS_REGION" in os.environ,
                "AZURE_CLIENT_ID" in os.environ,
                "GOOGLE_CLOUD_PROJECT" in os.environ,
            ]
        ),
    }

    detected_cloud = [
        provider for provider, detected in cloud_providers.items() if detected
    ]
    print(f"Detected environment: {', '.join(detected_cloud)}")

    # Environment-specific recommendations
    print("\nEnvironment-specific recommendations:")
    if in_docker:
        print("- Running in Docker: Consider mounting model cache volume")
        print("- Use environment variables for configuration")
        print("- Monitor memory usage in container")

    if in_k8s:
        print("- Running in Kubernetes: Use ConfigMaps for configuration")
        print("- Consider HorizontalPodAutoscaler for scaling")
        print("- Use persistent volumes for model cache")

    if "AWS" in detected_cloud:
        print("- AWS detected: Consider using EC2 instances with Intel Xeon")
        print("- Use ECS or EKS for container orchestration")
        print("- Consider AWS Inferentia for specialized inference")


def main():
    """Main demonstration function."""
    print("OpenVINO-Easy Cross-Platform Deployment Examples")
    print("=" * 50)

    demonstrations = [
        ("Platform Detection", demonstrate_platform_detection),
        ("Optimized Inference", demonstrate_optimized_inference),
        ("Docker Generation", demonstrate_docker_generation),
        ("Kubernetes Generation", demonstrate_kubernetes_generation),
        ("CI/CD Generation", demonstrate_cicd_generation),
        ("Environment Detection", demonstrate_environment_detection),
    ]

    print("\nAvailable demonstrations:")
    for i, (name, _) in enumerate(demonstrations, 1):
        print(f"{i}. {name}")
    print("0. Run all demonstrations")

    try:
        choice = input("\nSelect demonstration (0-6): ").strip()

        if choice == "0":
            # Run all demonstrations
            for name, func in demonstrations:
                print(f"\n{'=' * 20} {name} {'=' * 20}")
                try:
                    func()
                except Exception as e:
                    print(f"Error in {name}: {e}")
        elif choice.isdigit() and 1 <= int(choice) <= len(demonstrations):
            name, func = demonstrations[int(choice) - 1]
            print(f"\n{'=' * 20} {name} {'=' * 20}")
            func()
        else:
            print("Invalid choice. Running platform detection.")
            demonstrate_platform_detection()

    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"Error during demonstration: {e}")


if __name__ == "__main__":
    main()
