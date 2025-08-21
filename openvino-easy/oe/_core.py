"""Core utilities for OpenVINO-Easy (device detection, etc)."""

import openvino as ov
from typing import List, Optional, Union


def get_available_devices() -> List[str]:
    """
    Get list of available OpenVINO devices with validation.

    Returns:
        List of validated device names
    """
    core = ov.Core()
    available_devices = core.available_devices
    validated_devices = []

    for device in available_devices:
        if _validate_device(core, device):
            validated_devices.append(device)

    return validated_devices


def _validate_device(core: ov.Core, device: str) -> bool:
    """
    Validate that a device is actually functional.

    Args:
        core: OpenVINO Core instance
        device: Device name to validate

    Returns:
        True if device is functional, False otherwise
    """
    try:
        # For NPU, do additional validation
        if device == "NPU":
            return _validate_npu(core)

        # For other devices, try to get basic properties
        device_name = core.get_property(device, "FULL_DEVICE_NAME")
        return len(device_name) > 0

    except Exception:
        # Device is not functional
        return False


def _validate_npu(core: ov.Core) -> bool:
    """
    Validate NPU device availability and driver functionality.
    Enhanced for Arrow Lake/Lunar Lake NPU support.

    Args:
        core: OpenVINO Core instance

    Returns:
        True if NPU is functional, False otherwise
    """
    try:
        # Check if NPU device exists
        if "NPU" not in core.available_devices:
            return False

        # Try to get NPU-specific properties
        try:
            device_name = core.get_property("NPU", "FULL_DEVICE_NAME")

            # Check if it's a real NPU (not a virtual/stub device)
            if (
                not device_name
                or "stub" in device_name.lower()
                or "virtual" in device_name.lower()
            ):
                return False

            # Try to get additional NPU properties to ensure driver is loaded
            try:
                # These properties should exist if NPU driver is properly loaded
                core.get_property("NPU", "SUPPORTED_PROPERTIES")
                return True
            except:
                # If we can't get supported properties, driver might not be loaded
                return False

        except Exception:
            # Can't get device properties, NPU not functional
            return False

    except Exception:
        # Any other error means NPU is not available
        return False


def detect_device(device_preference: Optional[Union[List[str], tuple]] = None) -> str:
    """
    Detect the best available device based on preference order.

    Args:
        device_preference: Tuple or list of preferred devices in order

    Returns:
        Best available device name
    """
    if device_preference is None:
        device_preference = ["NPU", "GPU", "CPU"]

    # Convert tuple to list if needed
    if isinstance(device_preference, tuple):
        device_preference = list(device_preference)

    available_devices = get_available_devices()

    # Find the first preferred device that's available
    for preferred in device_preference:
        if preferred in available_devices:
            return preferred

    # Fallback to CPU if nothing else is available (even if not in available_devices)
    # CPU should always be available in OpenVINO
    return "CPU"


def get_npu_generation(device_name: str) -> str:
    """
    Detect NPU generation based on device name.

    Args:
        device_name: Full device name from OpenVINO

    Returns:
        NPU generation identifier
    """
    if not device_name:
        return "unknown"

    device_lower = device_name.lower()

    # Intel Core Ultra (Arrow Lake/Lunar Lake) NPUs
    if any(gen in device_lower for gen in ["arrow lake", "lunar lake", "core ultra"]):
        if "lunar lake" in device_lower:
            return "lunar_lake"  # Latest generation with FP16-NF4
        elif "arrow lake" in device_lower:
            return "arrow_lake"  # Desktop variant
        else:
            return "core_ultra"  # Generic Core Ultra

    # Legacy NPU generations
    elif any(gen in device_lower for gen in ["meteor lake", "raptor lake"]):
        return "meteor_lake"  # First-gen Core Ultra NPUs

    # Intel discrete NPUs (Gaudi, etc.)
    elif "gaudi" in device_lower:
        return "gaudi"

    # Generic Intel NPU
    elif "intel" in device_lower and "npu" in device_lower:
        return "intel_npu"

    return "unknown"


def get_npu_capabilities(npu_generation: str) -> dict:
    """
    Get expected capabilities for different NPU generations.

    Args:
        npu_generation: NPU generation from get_npu_generation()

    Returns:
        Dictionary with capabilities and performance expectations
    """
    capabilities = {
        "lunar_lake": {
            "supports_fp16": True,
            "supports_fp16_nf4": True,  # New in OpenVINO 2025.2
            "supports_int8": True,
            "max_compute_units": 8,
            "expected_stable_diffusion_fps": 2.3,  # >2.3 img/s target
            "expected_dialog_gpt_tps": 50,  # tokens per second
            "generation": "3rd",
            "features": [
                "Advanced quantization",
                "FP16-NF4 mixed precision",
                "Enhanced efficiency",
            ],
        },
        "arrow_lake": {
            "supports_fp16": True,
            "supports_fp16_nf4": True,
            "supports_int8": True,
            "max_compute_units": 8,
            "expected_stable_diffusion_fps": 2.2,
            "expected_dialog_gpt_tps": 48,
            "generation": "3rd",
            "features": [
                "Desktop optimization",
                "FP16-NF4 mixed precision",
                "High performance",
            ],
        },
        "core_ultra": {
            "supports_fp16": True,
            "supports_fp16_nf4": False,  # Depends on specific generation
            "supports_int8": True,
            "max_compute_units": 6,
            "expected_stable_diffusion_fps": 1.8,
            "expected_dialog_gpt_tps": 40,
            "generation": "2nd",
            "features": ["Integrated AI acceleration", "Power efficient"],
        },
        "meteor_lake": {
            "supports_fp16": True,
            "supports_fp16_nf4": False,
            "supports_int8": True,
            "max_compute_units": 4,
            "expected_stable_diffusion_fps": 1.5,
            "expected_dialog_gpt_tps": 35,
            "generation": "1st",
            "features": ["First-gen integrated NPU", "Basic AI acceleration"],
        },
        "unknown": {
            "supports_fp16": True,
            "supports_fp16_nf4": False,
            "supports_int8": True,
            "max_compute_units": None,
            "expected_stable_diffusion_fps": 1.0,
            "expected_dialog_gpt_tps": 25,
            "generation": "unknown",
            "features": ["Generic NPU support"],
        },
    }

    return capabilities.get(npu_generation, capabilities["unknown"])


def check_npu_driver() -> dict:
    """
    Check NPU driver status and provide diagnostic information.
    Enhanced for Arrow Lake/Lunar Lake NPU detection.

    Returns:
        Dictionary with NPU driver status and diagnostic info
    """
    core = ov.Core()

    result = {
        "npu_in_available_devices": "NPU" in core.available_devices,
        "npu_functional": False,
        "device_name": None,
        "npu_generation": "unknown",
        "capabilities": {},
        "driver_status": "unknown",
        "recommendations": [],
    }

    if not result["npu_in_available_devices"]:
        result["driver_status"] = "not_detected"
        result["recommendations"].append(
            "Install Intel NPU driver from intel.com/content/www/us/en/support"
        )
        result["recommendations"].append(
            "Check if NPU is enabled in BIOS/UEFI settings"
        )
        result["recommendations"].append(
            "Verify you have a supported Intel processor (Core Ultra series)"
        )
        return result

    # NPU is listed, check if it's functional
    try:
        device_name = core.get_property("NPU", "FULL_DEVICE_NAME")
        result["device_name"] = device_name

        # Detect NPU generation and capabilities
        npu_generation = get_npu_generation(device_name)
        result["npu_generation"] = npu_generation
        result["capabilities"] = get_npu_capabilities(npu_generation)

        if not device_name:
            result["driver_status"] = "stub_device"
            result["recommendations"].append(
                "NPU device is virtual/stub - install proper driver"
            )
        elif "stub" in device_name.lower() or "virtual" in device_name.lower():
            result["driver_status"] = "stub_device"
            result["recommendations"].append(
                "NPU device is virtual/stub - install proper driver"
            )
        else:
            # Try to get more properties to validate driver
            try:
                core.get_property("NPU", "SUPPORTED_PROPERTIES")
                result["npu_functional"] = True
                result["driver_status"] = "functional"

                # Add generation-specific recommendations
                if npu_generation in ["lunar_lake", "arrow_lake"]:
                    result["recommendations"].append(
                        f"🎉 Latest generation NPU detected: {npu_generation}"
                    )
                    result["recommendations"].append(
                        "Consider using FP16-NF4 precision for optimal performance"
                    )
                    result["recommendations"].append(
                        f"Expected Stable Diffusion performance: ~{result['capabilities']['expected_stable_diffusion_fps']:.1f} img/s"
                    )
                elif npu_generation == "core_ultra":
                    result["recommendations"].append(
                        "Core Ultra NPU detected - good performance expected"
                    )
                elif npu_generation == "meteor_lake":
                    result["recommendations"].append(
                        "First-generation NPU - consider upgrading for better performance"
                    )

            except Exception as e:
                result["driver_status"] = "driver_incomplete"
                result["recommendations"].append(
                    "NPU driver may be incomplete - reinstall latest driver"
                )
                result["recommendations"].append(f"Driver error details: {str(e)}")

    except Exception as e:
        result["driver_status"] = "error"
        result["recommendations"].append(f"NPU driver error: {str(e)}")
        result["recommendations"].append("Try reinstalling Intel NPU drivers")

    return result
