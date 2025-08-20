# Production Deployment Guide

> **📝 Note**: This guide uses the new **3-function API** (`oe.load()`, `oe.infer()`, `oe.benchmark()`, `oe.unload()`). 
> For legacy Pipeline class patterns, see [Pipeline API Reference](api/pipeline.rst).

This guide provides comprehensive instructions for deploying OpenVINO-Easy in production environments, covering everything from basic deployment to enterprise-scale operations.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Basic Production Setup](#basic-production-setup)
3. [Container Deployment](#container-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Monitoring & Observability](#monitoring--observability)
6. [Security Considerations](#security-considerations)
7. [Performance Optimization](#performance-optimization)
8. [Scaling Strategies](#scaling-strategies)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- Python 3.8+
- 4GB RAM
- 2 CPU cores
- 10GB storage

**Recommended for Production:**
- Python 3.10+
- 16GB+ RAM
- 8+ CPU cores
- 50GB+ SSD storage
- Intel hardware with NPU/GPU support

### Software Dependencies

```bash
# Core dependencies
pip install "openvino-easy[cpu]"  # Basic CPU support

# For GPU acceleration
pip install "openvino-easy[gpu]"  # Adds GPU drivers

# For NPU acceleration (Intel hardware)
pip install "openvino-easy[npu]"  # Adds NPU drivers

# Full production environment
pip install "openvino-easy[all]"  # All features + dev tools
```

### Hardware Validation

```python
import oe

# Check available devices
devices = oe.devices()
print(f"Available devices: {devices}")

# Validate NPU functionality (Intel hardware)
from oe._core import check_npu_driver
npu_status = check_npu_driver()
print(f"NPU status: {npu_status}")
```

## Basic Production Setup

### 1. Environment Configuration

Create a dedicated production environment:

```bash
# Create production environment
python -m venv openvino_prod
source openvino_prod/bin/activate  # Linux/Mac
# or
openvino_prod\Scripts\activate     # Windows

# Install production dependencies
pip install --upgrade pip
pip install "openvino-easy[all]"
```

### 2. Model Preparation

Pre-cache models for faster startup:

```python
import oe
from pathlib import Path

# Production model registry
PRODUCTION_MODELS = {
    "text_generator": "distilgpt2",
    "image_classifier": "microsoft/resnet-18", 
    "speech_recognizer": "openai/whisper-tiny",
}

# Pre-load and cache models
cache_dir = Path("/opt/openvino_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

for name, model_id in PRODUCTION_MODELS.items():
    print(f"Caching {name}...")
    oe.load(
        model_id,
        device_preference=["NPU", "GPU", "CPU"],
        dtype="int8",  # Optimized for production
        cache_dir=str(cache_dir)
    )
    info = oe.get_info()
    print(f"✅ {name} cached on {info['device']}")
    oe.unload()  # Unload after caching
```

### 3. Configuration Management

Use environment variables for configuration:

```python
import os
from dataclasses import dataclass
from typing import List

@dataclass
class ProductionConfig:
    # Model settings
    default_device_preference: List[str]
    default_dtype: str
    cache_dir: str
    
    # Performance settings
    max_concurrent_requests: int
    request_timeout: float
    batch_size: int
    
    # Monitoring settings
    enable_metrics: bool
    metrics_port: int
    log_level: str
    
    @classmethod
    def from_env(cls):
        return cls(
            default_device_preference=os.getenv("OE_DEVICES", "NPU,GPU,CPU").split(","),
            default_dtype=os.getenv("OE_DTYPE", "int8"),
            cache_dir=os.getenv("OE_CACHE_DIR", "/opt/openvino_cache"),
            max_concurrent_requests=int(os.getenv("OE_MAX_REQUESTS", "10")),
            request_timeout=float(os.getenv("OE_TIMEOUT", "30.0")),
            batch_size=int(os.getenv("OE_BATCH_SIZE", "4")),
            enable_metrics=os.getenv("OE_ENABLE_METRICS", "true").lower() == "true",
            metrics_port=int(os.getenv("OE_METRICS_PORT", "8080")),
            log_level=os.getenv("OE_LOG_LEVEL", "INFO")
        )

# Usage
config = ProductionConfig.from_env()
```

### 4. Basic Production Service

```python
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import oe

# Production service
app = FastAPI(title="OpenVINO-Easy Production API")

# Global model state tracking
current_model = None
current_model_name = None

class InferenceRequest(BaseModel):
    model_name: str
    input_data: Any
    options: Dict[str, Any] = {}

class InferenceResponse(BaseModel):
    result: Any
    inference_time: float
    model_info: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Validate models on startup."""
    config = ProductionConfig.from_env()
    
    print("🚀 Validating production models...")
    for name, model_id in PRODUCTION_MODELS.items():
        print(f"Validating {name}...")
        oe.load(
            model_id,
            device_preference=config.default_device_preference,
            dtype=config.default_dtype,
            cache_dir=config.cache_dir
        )
        info = oe.get_info()
        print(f"✅ {name} validated on {info['device']}")
        oe.unload()
    
    print("✅ All models validated")

def ensure_model_loaded(model_name: str):
    """Ensure the correct model is loaded."""
    global current_model, current_model_name
    
    if current_model_name != model_name:
        # Unload current model if any
        if current_model_name is not None:
            oe.unload()
        
        # Load requested model
        if model_name not in PRODUCTION_MODELS:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        config = ProductionConfig.from_env()
        oe.load(
            PRODUCTION_MODELS[model_name],
            device_preference=config.default_device_preference,
            dtype=config.default_dtype,
            cache_dir=config.cache_dir
        )
        current_model_name = model_name
        current_model = True

async def infer(request: InferenceRequest):
    """Run inference on specified model."""
    try:
        # Ensure correct model is loaded
        ensure_model_loaded(request.model_name)
        
        import time
        start_time = time.time()
        result = oe.infer(request.input_data)
        inference_time = time.time() - start_time
        
        return InferenceResponse(
            result=result,
            inference_time=inference_time,
            model_info=oe.get_info()
        )
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up models on shutdown."""
    global current_model_name
    if current_model_name is not None:
        oe.unload()
        current_model_name = None
        print("🧹 Models unloaded on shutdown")

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models": list(models.keys()),
        "timestamp": time.time()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Container Deployment

### Dockerfile

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application code
COPY . .

# Create cache directory
RUN mkdir -p /opt/openvino_cache

# Set environment variables
ENV OE_CACHE_DIR=/opt/openvino_cache
ENV OE_DEVICES=CPU
ENV OE_DTYPE=int8
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "production_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  openvino-easy:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OE_CACHE_DIR=/opt/openvino_cache
      - OE_DEVICES=CPU,GPU
      - OE_DTYPE=int8
      - OE_MAX_REQUESTS=20
      - OE_TIMEOUT=30.0
      - OE_ENABLE_METRICS=true
    volumes:
      - model_cache:/opt/openvino_cache
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'
  
  # Optional: Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  model_cache:
  grafana_data:
```

## Cloud Deployment

### AWS Deployment

#### EC2 Instance Setup

```bash
# Launch EC2 instance (recommend c5.2xlarge or better)
# Install dependencies
sudo apt update
sudo apt install -y python3-pip docker.io docker-compose
sudo usermod -aG docker $USER

# Clone and deploy
git clone https://github.com/your-org/openvino-easy.git
cd openvino-easy
docker-compose up -d
```

#### ECS Task Definition

```json
{
  "family": "openvino-easy-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "openvino-easy",
      "image": "your-registry/openvino-easy:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "OE_DEVICES", "value": "CPU"},
        {"name": "OE_DTYPE", "value": "int8"},
        {"name": "OE_MAX_REQUESTS", "value": "10"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/openvino-easy",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Azure Container Instances

```yaml
apiVersion: 2019-12-01
location: eastus
name: openvino-easy-instance
properties:
  containers:
  - name: openvino-easy
    properties:
      image: your-registry/openvino-easy:latest
      resources:
        requests:
          cpu: 2
          memoryInGb: 8
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: OE_DEVICES
        value: CPU
      - name: OE_DTYPE
        value: int8
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8000
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openvino-easy-deployment
  labels:
    app: openvino-easy
spec:
  replicas: 3
  selector:
    matchLabels:
      app: openvino-easy
  template:
    metadata:
      labels:
        app: openvino-easy
    spec:
      containers:
      - name: openvino-easy
        image: your-registry/openvino-easy:latest
        ports:
        - containerPort: 8000
        env:
        - name: OE_DEVICES
          value: "CPU"
        - name: OE_DTYPE
          value: "int8"
        - name: OE_MAX_REQUESTS
          value: "10"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: openvino-easy-service
spec:
  selector:
    app: openvino-easy
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

## Monitoring & Observability

### Metrics Collection

```python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter('openvino_requests_total', 'Total requests', ['model', 'status'])
REQUEST_DURATION = Histogram('openvino_request_duration_seconds', 'Request duration', ['model'])
ACTIVE_REQUESTS = Gauge('openvino_active_requests', 'Active requests', ['model'])
MODEL_LOAD_TIME = Histogram('openvino_model_load_seconds', 'Model load time', ['model'])

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/infer":
            start_time = time.time()
            
            # Track active requests
            model_name = "unknown"  # Extract from request
            ACTIVE_REQUESTS.labels(model=model_name).inc()
            
            try:
                await self.app(scope, receive, send)
                REQUEST_COUNT.labels(model=model_name, status="success").inc()
            except Exception:
                REQUEST_COUNT.labels(model=model_name, status="error").inc()
                raise
            finally:
                duration = time.time() - start_time
                REQUEST_DURATION.labels(model=model_name).observe(duration)
                ACTIVE_REQUESTS.labels(model=model_name).dec()
        else:
            await self.app(scope, receive, send)

# Add metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(
        prometheus_client.generate_latest(),
        media_type="text/plain"
    )
```

### Logging Configuration

```python
import logging
import json
from datetime import datetime

class ProductionFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'model_name'):
            log_entry['model_name'] = record.model_name
        if hasattr(record, 'inference_time'):
            log_entry['inference_time'] = record.inference_time
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
            
        return json.dumps(log_entry)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/openvino-easy.log')
    ]
)

# Add formatter
for handler in logging.getLogger().handlers:
    handler.setFormatter(ProductionFormatter())
```

## Security Considerations

### API Security

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token."""
    try:
        payload = jwt.decode(
            credentials.credentials,
            os.getenv("JWT_SECRET_KEY"),
            algorithms=["HS256"]
        )
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

@app.post("/infer")
async def infer(request: InferenceRequest, token: dict = Depends(verify_token)):
    """Secured inference endpoint."""
    # Implementation...
```

### Input Validation

```python
from pydantic import BaseModel, validator
from typing import Union
import numpy as np

class SecureInferenceRequest(BaseModel):
    model_name: str
    input_data: Union[str, list, dict]
    
    @validator('model_name')
    def validate_model_name(cls, v):
        # Whitelist allowed models
        allowed_models = ['text_generator', 'image_classifier', 'speech_recognizer']
        if v not in allowed_models:
            raise ValueError(f'Model {v} not allowed')
        return v
    
    @validator('input_data')
    def validate_input_size(cls, v):
        # Limit input size to prevent DoS
        if isinstance(v, str) and len(v) > 10000:
            raise ValueError('Text input too long')
        if isinstance(v, list) and len(v) > 1000:
            raise ValueError('List input too long')
        return v
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/infer")
@limiter.limit("10/minute")
async def infer(request: Request, inference_request: InferenceRequest):
    """Rate-limited inference endpoint."""
    # Implementation...
```

## Performance Optimization

### Model Optimization

```python
# Pre-optimize models for production
def optimize_model_for_production(model_id: str, target_device: str = "CPU"):
    """Optimize model for production deployment."""
    
    # Load with optimal settings
    pipeline = oe.load(
        model_id,
        device_preference=[target_device],
        dtype="int8",  # Quantization for speed
        cache_dir="/opt/openvino_cache"
    )
    
    # Warmup inference
    if "whisper" in model_id.lower():
        # Audio model warmup
        import wave
        import numpy as np
        
        # Create dummy audio
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        dummy_audio = np.zeros(samples, dtype=np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            with wave.open(f.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(dummy_audio.tobytes())
            
            # Warmup
            for _ in range(3):
                pipeline.infer(f.name)
    
    elif any(term in model_id.lower() for term in ["resnet", "vit", "image"]):
        # Image model warmup
        dummy_image = np.random.randn(1, 3, 224, 224).astype(np.float32)
        for _ in range(3):
            pipeline.infer(dummy_image)
    
    else:
        # Text model warmup
        for _ in range(3):
            pipeline.infer("warmup text")
    
    return pipeline
```

### Connection Pooling

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

class ModelPool:
    """Pool of model instances for high concurrency."""
    
    def __init__(self, model_id: str, pool_size: int = 4):
        self.model_id = model_id
        self.pool_size = pool_size
        self.models = []
        self.executor = ThreadPoolExecutor(max_workers=pool_size)
        
        # Initialize pool
        for _ in range(pool_size):
            model = optimize_model_for_production(model_id)
            self.models.append(model)
    
    async def infer(self, input_data: Any) -> Any:
        """Run inference using available model from pool."""
        loop = asyncio.get_event_loop()
        
        # Get available model (simplified round-robin)
        model = self.models[hash(input_data) % len(self.models)]
        
        # Run inference in thread pool
        result = await loop.run_in_executor(
            self.executor,
            model.infer,
            input_data
        )
        
        return result

# Usage
model_pools: Dict[str, ModelPool] = {}

@app.on_event("startup")
async def startup():
    for model_name, model_id in PRODUCTION_MODELS.items():
        model_pools[model_name] = ModelPool(model_id, pool_size=4)
```

## Scaling Strategies

### Horizontal Scaling

```bash
# Docker Swarm
docker service create \
  --name openvino-easy \
  --replicas 5 \
  --publish 8000:8000 \
  --mount type=volume,src=model-cache,dst=/opt/openvino_cache \
  your-registry/openvino-easy:latest

# Kubernetes Horizontal Pod Autoscaler
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: openvino-easy-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: openvino-easy-deployment
  minReplicas: 3
  maxReplicas: 20
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
EOF
```

### Load Balancing

```nginx
# Nginx configuration
upstream openvino_backend {
    least_conn;
    server openvino-1:8000 max_fails=3 fail_timeout=30s;
    server openvino-2:8000 max_fails=3 fail_timeout=30s;
    server openvino-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://openvino_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    location /health {
        access_log off;
        proxy_pass http://openvino_backend;
    }
}
```

## Troubleshooting

### Common Issues

**Model Loading Failures:**
```bash
# Check available devices
python -c "import oe; print(oe.devices())"

# Verify cache permissions
ls -la /opt/openvino_cache/
chmod -R 755 /opt/openvino_cache/

# Check OpenVINO installation
python -c "import openvino; print(openvino.__version__)"
```

**Performance Issues:**
```python
# Debug inference performance
pipeline = oe.load("model_id", device_preference=["CPU"])
stats = pipeline.benchmark(warmup_runs=5, benchmark_runs=20)
print(f"Performance stats: {stats}")

# Check memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

**Container Issues:**
```bash
# Check container logs
docker logs openvino-easy-container

# Debug container resources
docker stats openvino-easy-container

# Test model loading in container
docker exec -it openvino-easy-container python -c "import oe; print(oe.devices())"
```

### Performance Monitoring

```bash
# Monitor system resources
htop
iotop
nvidia-smi  # For GPU monitoring

# Monitor application metrics
curl http://localhost:8000/metrics | grep openvino

# Check API health
curl http://localhost:8000/health
```

## Conclusion

This production deployment guide covers the essential aspects of deploying OpenVINO-Easy in production environments. Key takeaways:

1. **Prepare thoroughly**: Cache models, configure environments, validate hardware
2. **Monitor everything**: Metrics, logs, health checks, and performance
3. **Secure by design**: Authentication, input validation, rate limiting
4. **Scale intelligently**: Use pools, horizontal scaling, and load balancing
5. **Plan for failure**: Health checks, timeouts, fallbacks, and monitoring

For additional support, check the [Performance Tuning Guide](performance_tuning.md) and [API Reference](api/).