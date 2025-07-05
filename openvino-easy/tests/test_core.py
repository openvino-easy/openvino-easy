"""Unit tests for oe._core."""

import pytest
from unittest.mock import patch, MagicMock
from oe._core import detect_device

class TestCore:
    """Test core utilities."""
    
    @patch('oe._core.ov.Core')
    def test_detect_device_npu_available(self, mock_core):
        """Test device detection when NPU is available."""
        mock_core_instance = MagicMock()
        mock_core_instance.available_devices = ["NPU", "GPU", "CPU"]
        mock_core.return_value = mock_core_instance
        
        result = detect_device()
        assert result == "NPU"
    
    @patch('oe._core.ov.Core')
    def test_detect_device_npu_not_available(self, mock_core):
        """Test device detection when NPU is not available."""
        mock_core_instance = MagicMock()
        mock_core_instance.available_devices = ["GPU", "CPU"]
        mock_core.return_value = mock_core_instance
        
        result = detect_device()
        assert result == "GPU"
    
    @patch('oe._core.ov.Core')
    def test_detect_device_only_cpu_available(self, mock_core):
        """Test device detection when only CPU is available."""
        mock_core_instance = MagicMock()
        mock_core_instance.available_devices = ["CPU"]
        mock_core.return_value = mock_core_instance
        
        result = detect_device()
        assert result == "CPU"
    
    @patch('oe._core.ov.Core')
    def test_detect_device_custom_preference(self, mock_core):
        """Test device detection with custom preference order."""
        mock_core_instance = MagicMock()
        mock_core_instance.available_devices = ["GPU", "CPU", "NPU"]
        mock_core.return_value = mock_core_instance
        
        result = detect_device(("CPU", "GPU", "NPU"))
        assert result == "CPU"
    
    @patch('oe._core.ov.Core')
    def test_detect_device_no_preferred_devices_available(self, mock_core):
        """Test device detection when no preferred devices are available."""
        mock_core_instance = MagicMock()
        mock_core_instance.available_devices = ["MYRIAD", "HDDL"]
        mock_core.return_value = mock_core_instance
        
        result = detect_device(("NPU", "GPU", "CPU"))
        assert result == "CPU"  # Should fall back to CPU
    
    @patch('oe._core.ov.Core')
    def test_detect_device_empty_available_devices(self, mock_core):
        """Test device detection when no devices are available."""
        mock_core_instance = MagicMock()
        mock_core_instance.available_devices = []
        mock_core.return_value = mock_core_instance
        
        result = detect_device()
        assert result == "CPU"  # Should fall back to CPU
    
    @patch('oe._core.ov.Core')
    def test_detect_device_default_preference(self, mock_core):
        """Test that default preference is used when none provided."""
        mock_core_instance = MagicMock()
        mock_core_instance.available_devices = ["NPU", "GPU", "CPU"]
        mock_core.return_value = mock_core_instance
        
        result = detect_device()
        # Should use default preference: ("NPU", "GPU", "CPU")
        assert result == "NPU" 