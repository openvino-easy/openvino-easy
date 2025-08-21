"""Installation validator for OpenVINO-Easy system requirements."""

import sys
import platform
import subprocess
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .dep_validator import (
    validate_core_dependencies, 
    validate_system_requirements, 
    comprehensive_dependency_check
)
from .windows_compat import check_windows_compatibility, get_safe_cache_dir


def _safe_log_unicode(level, emoji_msg: str, ascii_msg: str):
    """Log messages with Unicode emoji fallback to ASCII on Windows console."""
    try:
        # Try Unicode first on all platforms
        if level == "info":
            logging.info(emoji_msg)
        elif level == "warning":
            logging.warning(emoji_msg)
        elif level == "error":
            logging.error(emoji_msg)
    except UnicodeEncodeError:
        # Fallback to ASCII version on encoding errors
        if level == "info":
            logging.info(ascii_msg)
        elif level == "warning":
            logging.warning(ascii_msg)
        elif level == "error":
            logging.error(ascii_msg)


def check_python_environment() -> Dict[str, Any]:
    """Check Python environment compatibility."""
    env_check = {
        "python_version": {
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "supported": sys.version_info >= (3, 8),
            "issues": [],
            "recommendations": [],
        },
        "virtual_env": {
            "active": False,
            "path": None,
            "type": None,
            "issues": [],
            "recommendations": [],
        },
        "pip_version": {
            "version": None,
            "issues": [],
            "recommendations": [],
        },
    }
    
    # Python version check
    if not env_check["python_version"]["supported"]:
        env_check["python_version"]["issues"].append(
            f"Python {env_check['python_version']['version']} is not supported (minimum: 3.8)"
        )
        env_check["python_version"]["recommendations"].append("Upgrade to Python 3.8 or newer")
    
    # Virtual environment check
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        env_check["virtual_env"]["active"] = True
        env_check["virtual_env"]["path"] = sys.prefix
        
        # Detect virtual environment type
        if "conda" in sys.prefix.lower() or "anaconda" in sys.prefix.lower():
            env_check["virtual_env"]["type"] = "conda"
        elif "venv" in sys.prefix.lower():
            env_check["virtual_env"]["type"] = "venv"
        else:
            env_check["virtual_env"]["type"] = "unknown"
    else:
        env_check["virtual_env"]["issues"].append("No virtual environment detected")
        env_check["virtual_env"]["recommendations"].append(
            "Consider using a virtual environment to avoid package conflicts"
        )
    
    # Pip version check
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            pip_version = result.stdout.strip().split()[1]
            env_check["pip_version"]["version"] = pip_version
        else:
            env_check["pip_version"]["issues"].append("pip not accessible")
            env_check["pip_version"]["recommendations"].append("Install or repair pip")
    except Exception as e:
        env_check["pip_version"]["issues"].append(f"Could not check pip version: {e}")
        env_check["pip_version"]["recommendations"].append("Ensure pip is properly installed")
    
    return env_check


def check_hardware_requirements() -> Dict[str, Any]:
    """Check hardware requirements for optimal performance."""
    hw_check = {
        "memory": {
            "total_gb": 0,
            "available_gb": 0,
            "sufficient": False,
            "issues": [],
            "recommendations": [],
        },
        "storage": {
            "cache_dir": None,
            "free_gb": 0,
            "sufficient": False,
            "issues": [],
            "recommendations": [],
        },
        "cpu": {
            "architecture": platform.machine(),
            "cores": 0,
            "supported": True,
            "issues": [],
            "recommendations": [],
        },
    }
    
    # Memory check
    try:
        import psutil
        memory = psutil.virtual_memory()
        hw_check["memory"]["total_gb"] = round(memory.total / (1024**3), 1)
        hw_check["memory"]["available_gb"] = round(memory.available / (1024**3), 1)
        
        if hw_check["memory"]["total_gb"] < 4:
            hw_check["memory"]["issues"].append(f"Low system memory: {hw_check['memory']['total_gb']}GB")
            hw_check["memory"]["recommendations"].append("Recommend at least 8GB RAM for optimal performance")
        elif hw_check["memory"]["total_gb"] < 8:
            hw_check["memory"]["recommendations"].append("Consider upgrading to 16GB+ RAM for large models")
        
        hw_check["memory"]["sufficient"] = hw_check["memory"]["total_gb"] >= 4
        
    except ImportError:
        hw_check["memory"]["issues"].append("Could not check memory (psutil not available)")
        hw_check["memory"]["recommendations"].append("Install psutil: pip install psutil")
    except Exception as e:
        hw_check["memory"]["issues"].append(f"Memory check failed: {e}")
    
    # Storage check
    try:
        cache_dir = get_safe_cache_dir()
        hw_check["storage"]["cache_dir"] = str(cache_dir)
        
        import shutil
        total, used, free = shutil.disk_usage(str(cache_dir))
        hw_check["storage"]["free_gb"] = round(free / (1024**3), 1)
        
        if hw_check["storage"]["free_gb"] < 5:
            hw_check["storage"]["issues"].append(f"Low disk space: {hw_check['storage']['free_gb']}GB available")
            hw_check["storage"]["recommendations"].append("Free up at least 10GB of disk space")
        elif hw_check["storage"]["free_gb"] < 10:
            hw_check["storage"]["recommendations"].append("Consider freeing up more space for large models")
        
        hw_check["storage"]["sufficient"] = hw_check["storage"]["free_gb"] >= 5
        
    except Exception as e:
        hw_check["storage"]["issues"].append(f"Storage check failed: {e}")
    
    # CPU check
    try:
        import multiprocessing
        hw_check["cpu"]["cores"] = multiprocessing.cpu_count()
        
        if hw_check["cpu"]["cores"] < 2:
            hw_check["cpu"]["issues"].append(f"Low CPU cores: {hw_check['cpu']['cores']}")
            hw_check["cpu"]["recommendations"].append("Multi-core CPU recommended for better performance")
        
        # Check architecture support
        arch = platform.machine().lower()
        if arch not in ["x86_64", "amd64", "aarch64", "arm64"]:
            hw_check["cpu"]["supported"] = False
            hw_check["cpu"]["issues"].append(f"Unsupported architecture: {arch}")
            hw_check["cpu"]["recommendations"].append("x86_64 or ARM64 architecture required")
            
    except Exception as e:
        hw_check["cpu"]["issues"].append(f"CPU check failed: {e}")
    
    return hw_check


def check_openvino_compatibility() -> Dict[str, Any]:
    """Check OpenVINO specific requirements and device availability."""
    ov_check = {
        "runtime": {
            "installed": False,
            "version": None,
            "compatible": False,
            "issues": [],
            "recommendations": [],
        },
        "devices": {
            "available": [],
            "recommended": [],
            "issues": [],
            "recommendations": [],
        },
    }
    
    # OpenVINO runtime check
    try:
        import openvino as ov
        ov_check["runtime"]["installed"] = True
        ov_check["runtime"]["version"] = ov.__version__
        
        # Check version compatibility
        version_parts = ov.__version__.split(".")
        if len(version_parts) >= 2:
            major, minor = int(version_parts[0]), int(version_parts[1])
            if major >= 2025 or (major == 2024 and minor >= 3):
                ov_check["runtime"]["compatible"] = True
            else:
                ov_check["runtime"]["issues"].append(
                    f"OpenVINO {ov.__version__} may have compatibility issues"
                )
                ov_check["runtime"]["recommendations"].append(
                    "Upgrade to OpenVINO 2024.3+ or 2025.x"
                )
        
        # Check available devices
        try:
            core = ov.Core()
            available_devices = core.available_devices
            ov_check["devices"]["available"] = available_devices
            
            # Prioritize devices
            if "NPU" in available_devices:
                ov_check["devices"]["recommended"].append("NPU")
            if "GPU" in available_devices:
                ov_check["devices"]["recommended"].append("GPU")
            if "CPU" in available_devices:
                ov_check["devices"]["recommended"].append("CPU")
            
            if not ov_check["devices"]["recommended"]:
                ov_check["devices"]["issues"].append("No recommended devices available")
                ov_check["devices"]["recommendations"].append(
                    "Ensure proper OpenVINO installation with device support"
                )
                
        except Exception as e:
            ov_check["devices"]["issues"].append(f"Device detection failed: {e}")
            
    except ImportError:
        ov_check["runtime"]["issues"].append("OpenVINO runtime not installed")
        ov_check["runtime"]["recommendations"].append(
            "Install OpenVINO: pip install openvino"
        )
    except Exception as e:
        ov_check["runtime"]["issues"].append(f"OpenVINO check failed: {e}")
    
    return ov_check


def run_comprehensive_validation() -> Dict[str, Any]:
    """Run comprehensive installation validation."""
    validation_results = {
        "timestamp": str(sys.version_info),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python_env": check_python_environment(),
        "hardware": check_hardware_requirements(),
        "openvino": check_openvino_compatibility(),
        "dependencies": comprehensive_dependency_check(),
        "windows_compat": None,
        "overall": {
            "compatible": True,
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
        },
    }
    
    # Windows-specific checks
    if sys.platform == "win32":
        validation_results["windows_compat"] = check_windows_compatibility()
    
    # Aggregate issues
    critical_issues = []
    warnings = []
    recommendations = []
    
    # Check each component
    for component_name, component_data in validation_results.items():
        if component_name in ["timestamp", "platform", "overall"]:
            continue
            
        if isinstance(component_data, dict):
            _extract_issues_recursive(component_data, critical_issues, warnings, recommendations)
    
    # Determine overall compatibility
    validation_results["overall"]["compatible"] = len(critical_issues) == 0
    validation_results["overall"]["critical_issues"] = critical_issues
    validation_results["overall"]["warnings"] = warnings
    validation_results["overall"]["recommendations"] = recommendations
    
    return validation_results


def _extract_issues_recursive(data: Dict[str, Any], critical_issues: List[str], 
                            warnings: List[str], recommendations: List[str]):
    """Recursively extract issues from nested dictionaries."""
    for key, value in data.items():
        if key == "issues" and isinstance(value, list):
            critical_issues.extend(value)
        elif key == "warnings" and isinstance(value, list):
            warnings.extend(value)
        elif key == "recommendations" and isinstance(value, list):
            recommendations.extend(value)
        elif isinstance(value, dict):
            _extract_issues_recursive(value, critical_issues, warnings, recommendations)


def print_validation_report(results: Dict[str, Any], verbose: bool = False):
    """Print formatted validation report."""
    
    def print_status(status: bool, good_msg: str, bad_msg: str):
        if status:
            _safe_log_unicode("info", f"✅ {good_msg}", f"[OK] {good_msg}")
        else:
            _safe_log_unicode("error", f"❌ {bad_msg}", f"[FAIL] {bad_msg}")
    
    # Header
    _safe_log_unicode(
        "info",
        "🔍 OpenVINO-Easy Installation Validation Report",
        "[VALIDATION] OpenVINO-Easy Installation Validation Report"
    )
    print("=" * 60)
    
    # Overall status
    overall = results["overall"]
    print_status(
        overall["compatible"],
        "System is compatible with OpenVINO-Easy",
        "System has compatibility issues"
    )
    
    # Python environment
    python_env = results["python_env"]
    print_status(
        python_env["python_version"]["supported"],
        f"Python {python_env['python_version']['version']} is supported",
        f"Python {python_env['python_version']['version']} is not supported"
    )
    
    # OpenVINO runtime
    ov_check = results["openvino"]
    if ov_check["runtime"]["installed"]:
        print_status(
            ov_check["runtime"]["compatible"],
            f"OpenVINO {ov_check['runtime']['version']} is compatible",
            f"OpenVINO {ov_check['runtime']['version']} may have issues"
        )
    else:
        print_status(False, "", "OpenVINO runtime not installed")
    
    # Hardware
    hw_check = results["hardware"]
    print_status(
        hw_check["memory"]["sufficient"],
        f"Memory: {hw_check['memory']['total_gb']}GB available",
        f"Memory: {hw_check['memory']['total_gb']}GB (insufficient)"
    )
    
    print_status(
        hw_check["storage"]["sufficient"],
        f"Storage: {hw_check['storage']['free_gb']}GB free",
        f"Storage: {hw_check['storage']['free_gb']}GB free (insufficient)"
    )
    
    # Critical issues
    if overall["critical_issues"]:
        _safe_log_unicode(
            "error",
            "\n🚨 Critical Issues:",
            "\n[CRITICAL] Critical Issues:"
        )
        for issue in overall["critical_issues"]:
            print(f"  • {issue}")
    
    # Recommendations
    if overall["recommendations"]:
        _safe_log_unicode(
            "info",
            "\n💡 Recommendations:",
            "\n[RECOMMEND] Recommendations:"
        )
        for rec in overall["recommendations"][:5]:  # Show top 5
            print(f"  • {rec}")
    
    # Verbose details
    if verbose:
        print("\n" + "=" * 60)
        _safe_log_unicode(
            "info",
            "📋 Detailed Report:",
            "[DETAILS] Detailed Report:"
        )
        
        if ov_check["devices"]["available"]:
            print(f"Available devices: {', '.join(ov_check['devices']['available'])}")
        
        if results.get("windows_compat"):
            win_compat = results["windows_compat"]
            print(f"Windows compatibility: {'✅' if win_compat['compatible'] else '❌'}")


def validate_installation(verbose: bool = False) -> bool:
    """
    Validate OpenVINO-Easy installation and system compatibility.
    
    Args:
        verbose: Show detailed validation report
        
    Returns:
        True if installation is valid and compatible
    """
    results = run_comprehensive_validation()
    print_validation_report(results, verbose=verbose)
    
    return results["overall"]["compatible"]