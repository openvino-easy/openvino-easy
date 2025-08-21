"""Dependency validation and auto-resolution for OpenVINO-Easy."""

import sys
import subprocess
import importlib
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path


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


# Dependency mappings for different model types
DEPENDENCY_MAPPINGS = {
    "text": {
        "packages": ["transformers", "tokenizers", "optimum[intel]"],
        "pip_extras": "openvino-easy[text]",
        "description": "Text models (GPT, BERT, etc.)",
    },
    "vision": {
        "packages": ["timm", "transformers", "optimum[intel]"],
        "pip_extras": "openvino-easy[vision]",
        "description": "Computer vision models",
    },
    "audio": {
        "packages": ["librosa", "transformers", "optimum[intel]"],
        "pip_extras": "openvino-easy[audio]",
        "description": "Audio and speech models",
    },
    "stable-diffusion": {
        "packages": ["diffusers", "optimum[intel]", "accelerate"],
        "pip_extras": "openvino-easy[stable-diffusion]",
        "description": "Stable Diffusion and image generation",
    },
    "quantization": {
        "packages": ["nncf"],
        "pip_extras": "openvino-easy[quant]",
        "description": "Model quantization",
    },
    "full": {
        "packages": [
            "transformers", "tokenizers", "optimum[intel]", "diffusers",
            "librosa", "timm", "nncf", "accelerate"
        ],
        "pip_extras": "openvino-easy[full]",
        "description": "Complete development environment",
    },
}

# Core packages that should always be available
CORE_PACKAGES = {
    "openvino": {
        "alternatives": ["openvino-cpu", "openvino-runtime"],
        "description": "OpenVINO runtime",
    },
    "huggingface-hub": {
        "alternatives": [],
        "description": "Hugging Face model hub client",
    },
    "numpy": {
        "alternatives": [],
        "description": "Numerical computing",
    },
}


def check_package_installed(package_name: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a package is installed and return version info.

    Args:
        package_name: Name of the package to check

    Returns:
        Tuple of (is_installed, version_or_error)
    """
    try:
        # Handle package names with extras like "optimum[intel]"
        base_package = package_name.split("[")[0]
        module = importlib.import_module(base_package)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error checking {package_name}: {e}"


def validate_core_dependencies() -> Dict[str, Dict]:
    """Validate that core dependencies are installed."""
    results = {}

    for package, info in CORE_PACKAGES.items():
        is_installed, version_or_error = check_package_installed(package)

        results[package] = {
            "installed": is_installed,
            "version": version_or_error if is_installed else None,
            "error": version_or_error if not is_installed else None,
            "description": info["description"],
            "alternatives": info["alternatives"],
        }

    return results


def validate_model_dependencies(model_type: str) -> Dict[str, Dict]:
    """
    Validate dependencies for a specific model type.

    Args:
        model_type: Type of model (text, vision, audio, etc.)

    Returns:
        Dictionary with validation results
    """
    if model_type not in DEPENDENCY_MAPPINGS:
        return {"error": f"Unknown model type: {model_type}"}

    dep_info = DEPENDENCY_MAPPINGS[model_type]
    results = {
        "model_type": model_type,
        "description": dep_info["description"],
        "pip_extras": dep_info["pip_extras"],
        "packages": {},
        "missing_packages": [],
        "all_satisfied": True,
    }

    for package in dep_info["packages"]:
        is_installed, version_or_error = check_package_installed(package)

        results["packages"][package] = {
            "installed": is_installed,
            "version": version_or_error if is_installed else None,
            "error": version_or_error if not is_installed else None,
        }

        if not is_installed:
            results["missing_packages"].append(package)
            results["all_satisfied"] = False

    return results


def auto_install_dependencies(
    packages: List[str],
    use_pip_extras: Optional[str] = None,
    confirm: bool = True
) -> Dict[str, any]:
    """
    Automatically install missing dependencies.

    Args:
        packages: List of packages to install
        use_pip_extras: Use pip extras syntax instead (e.g. "openvino-easy[text]")
        confirm: Whether to require user confirmation

    Returns:
        Installation results
    """
    if not packages and not use_pip_extras:
        return {"status": "success", "message": "No packages to install"}

    # Build installation command
    if use_pip_extras:
        install_target = use_pip_extras
        display_target = f"package bundle '{use_pip_extras}'"
    else:
        install_target = " ".join(packages)
        display_target = f"packages: {', '.join(packages)}"

    if confirm:
        _safe_log_unicode(
            "warning",
            f"⚠️  Missing dependencies detected. Would install: {display_target}",
            f"[WARNING] Missing dependencies detected. Would install: {display_target}"
        )
        return {
            "status": "confirmation_required",
            "message": f"Use auto_install=True to automatically install {display_target}",
            "install_command": f"pip install {install_target}",
        }

    try:
        _safe_log_unicode(
            "info",
            f"📦 Installing {display_target}...",
            f"[INSTALLING] Installing {display_target}..."
        )

        # Run pip install
        cmd = [sys.executable, "-m", "pip", "install"] + install_target.split()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode == 0:
            _safe_log_unicode(
                "info",
                f"✅ Successfully installed {display_target}",
                f"[SUCCESS] Successfully installed {display_target}"
            )
            return {
                "status": "success",
                "message": f"Successfully installed {display_target}",
                "stdout": result.stdout,
            }
        else:
            _safe_log_unicode(
                "error",
                f"❌ Failed to install {display_target}",
                f"[FAILED] Failed to install {display_target}"
            )
            return {
                "status": "failed",
                "message": f"Failed to install {display_target}",
                "error": result.stderr,
                "stdout": result.stdout,
                "install_command": f"pip install {install_target}",
            }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "message": f"Installation of {display_target} timed out",
            "install_command": f"pip install {install_target}",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error installing {display_target}: {e}",
            "install_command": f"pip install {install_target}",
        }


def validate_system_requirements() -> Dict[str, any]:
    """Validate system requirements for OpenVINO-Easy."""
    results = {
        "python_version": {
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "supported": sys.version_info >= (3, 8),
            "minimum": "3.8",
        },
        "platform": {
            "system": sys.platform,
            "supported": sys.platform in ["win32", "linux", "darwin"],
        },
        "pip_available": False,
        "virtual_env": {
            "active": False,
            "path": None,
        },
    }

    # Check pip availability
    try:
        import pip
        results["pip_available"] = True
    except ImportError:
        results["pip_available"] = False

    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        results["virtual_env"]["active"] = True
        results["virtual_env"]["path"] = sys.prefix

    # Overall system compatibility
    results["system_compatible"] = (
        results["python_version"]["supported"] and
        results["platform"]["supported"] and
        results["pip_available"]
    )

    return results


def get_installation_suggestions(model_type: Optional[str] = None) -> Dict[str, any]:
    """
    Get installation suggestions based on missing dependencies.

    Args:
        model_type: Specific model type to check, or None for general suggestions

    Returns:
        Installation suggestions and commands
    """
    system_check = validate_system_requirements()
    core_check = validate_core_dependencies()

    suggestions = {
        "system_issues": [],
        "core_issues": [],
        "model_issues": [],
        "recommended_commands": [],
    }

    # Check system issues
    if not system_check["python_version"]["supported"]:
        suggestions["system_issues"].append(
            f"Python {system_check['python_version']['minimum']}+ required, "
            f"found {system_check['python_version']['version']}"
        )

    if not system_check["pip_available"]:
        suggestions["system_issues"].append("pip not available for package installation")

    if not system_check["virtual_env"]["active"]:
        suggestions["system_issues"].append(
            "Virtual environment not active - consider using one to avoid conflicts"
        )

    # Check core dependencies
    for package, info in core_check.items():
        if not info["installed"]:
            suggestions["core_issues"].append(f"Missing core package: {package}")
            if info["alternatives"]:
                suggestions["recommended_commands"].append(
                    f"pip install {' '.join(info['alternatives'])}"
                )
            else:
                suggestions["recommended_commands"].append(f"pip install {package}")

    # Check model-specific dependencies
    if model_type:
        model_check = validate_model_dependencies(model_type)
        if not model_check.get("all_satisfied", True):
            suggestions["model_issues"].append(
                f"Missing dependencies for {model_type}: {', '.join(model_check['missing_packages'])}"
            )
            suggestions["recommended_commands"].append(
                f"pip install '{model_check['pip_extras']}'"
            )

    return suggestions


def comprehensive_dependency_check() -> Dict[str, any]:
    """Run a comprehensive dependency check for all components."""
    return {
        "system": validate_system_requirements(),
        "core": validate_core_dependencies(),
        "model_types": {
            model_type: validate_model_dependencies(model_type)
            for model_type in DEPENDENCY_MAPPINGS.keys()
        },
        "suggestions": get_installation_suggestions(),
    }
