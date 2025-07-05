"""Core utilities for OpenVINO-Easy (device detection, etc)."""

import openvino as ov
from typing import List, Optional
import warnings

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
        
    except Exception as e:
        # Device is not functional
        return False

def _validate_npu(core: ov.Core) -> bool:
    """
    Validate NPU device availability and driver functionality.
    
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
            if not device_name or "stub" in device_name.lower() or "virtual" in device_name.lower():
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

def detect_device(device_preference: Optional[List[str]] = None) -> str:
    """
    Detect the best available device based on preference order.
    
    Args:
        device_preference: List of preferred devices in order
        
    Returns:
        Best available device name
    """
    if device_preference is None:
        device_preference = ["NPU", "GPU", "CPU"]
    
    available_devices = get_available_devices()
    
    # Find the first preferred device that's available
    for preferred in device_preference:
        if preferred in available_devices:
            return preferred
    
    # Fallback to CPU if nothing else is available
    if "CPU" in available_devices:
        return "CPU"
    
    # If even CPU is not available, return the first available device
    if available_devices:
        return available_devices[0]
    
    # This should never happen, but return CPU as final fallback
    warnings.warn("No OpenVINO devices detected. Falling back to CPU.")
    return "CPU"

def check_npu_driver() -> dict:
    """
    Check NPU driver status and provide diagnostic information.
    
    Returns:
        Dictionary with NPU driver status and diagnostic info
    """
    core = ov.Core()
    
    result = {
        "npu_in_available_devices": "NPU" in core.available_devices,
        "npu_functional": False,
        "device_name": None,
        "driver_status": "unknown",
        "recommendations": []
    }
    
    if not result["npu_in_available_devices"]:
        result["driver_status"] = "not_detected"
        result["recommendations"].append("Install Intel NPU driver")
        result["recommendations"].append("Check if NPU is enabled in BIOS")
        return result
    
    # NPU is listed, check if it's functional
    try:
        device_name = core.get_property("NPU", "FULL_DEVICE_NAME")
        result["device_name"] = device_name
        
        if not device_name:
            result["driver_status"] = "stub_device"
            result["recommendations"].append("NPU device is virtual/stub - install proper driver")
        elif "stub" in device_name.lower() or "virtual" in device_name.lower():
            result["driver_status"] = "stub_device"
            result["recommendations"].append("NPU device is virtual/stub - install proper driver")
        else:
            # Try to get more properties to validate driver
            try:
                core.get_property("NPU", "SUPPORTED_PROPERTIES")
                result["npu_functional"] = True
                result["driver_status"] = "functional"
            except:
                result["driver_status"] = "driver_incomplete"
                result["recommendations"].append("NPU driver may be incomplete - reinstall driver")
                
    except Exception as e:
        result["driver_status"] = "error"
        result["recommendations"].append(f"NPU driver error: {str(e)}")
    
    return result 