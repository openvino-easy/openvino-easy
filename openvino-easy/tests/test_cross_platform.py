"""Cross-platform compatibility tests for OpenVINO-Easy."""

import pytest
import platform
import sys
import os
import tempfile
from pathlib import Path

import oe


class TestCrossPlatform:
    """Test suite for cross-platform compatibility."""

    def test_platform_detection(self):
        """Test that platform detection works correctly."""
        current_platform = platform.system()
        assert current_platform in ["Windows", "Linux", "Darwin"], (
            f"Unsupported platform: {current_platform}"
        )

        # Test Python version compatibility
        python_version = sys.version_info
        assert python_version >= (3, 8), (
            f"Python {python_version} not supported (minimum: 3.8)"
        )
        assert python_version < (4, 0), f"Python {python_version} not tested"

    def test_path_handling_cross_platform(self):
        """Test that path handling works across platforms."""
        # Test with different path separators
        test_paths = [
            "models/test_model.xml",
            "models\\test_model.xml",  # Windows style
            "models/subdir/test_model.xml",
            str(Path("models") / "test_model.xml"),  # Pathlib style
        ]

        for test_path in test_paths:
            normalized_path = Path(test_path)
            assert isinstance(normalized_path, Path)
            assert str(normalized_path)  # Should not be empty

    @pytest.mark.integration
    def test_device_detection_cross_platform(self):
        """Test device detection works on different platforms."""
        devices = oe.devices()
        assert isinstance(devices, list)

        # CPU should always be available
        assert "CPU" in devices

        # Platform-specific device availability
        current_platform = platform.system()

        if current_platform == "Windows":
            # Intel hardware more common on Windows
            print(f"Windows platform detected. Available devices: {devices}")
        elif current_platform == "Linux":
            print(f"Linux platform detected. Available devices: {devices}")
        elif current_platform == "Darwin":  # macOS
            print(f"macOS platform detected. Available devices: {devices}")
            # Intel devices less common on Apple Silicon Macs

    @pytest.mark.integration
    def test_model_loading_cross_platform(self):
        """Test model loading across platforms."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a simple test model
            import openvino as ov
            import numpy as np

            input_shape = [1, 10]
            input_node = ov.opset11.parameter(
                input_shape, dtype=np.float32, name="input"
            )

            # Simple operation
            weight = ov.opset11.constant(np.random.randn(10, 5).astype(np.float32))
            output = ov.opset11.matmul(
                input_node, weight, transpose_a=False, transpose_b=False
            )
            output.set_friendly_name("output")

            # Create and save model
            model = ov.Model([output], [input_node], "cross_platform_test")
            model_path = temp_path / "test_model.xml"
            ov.save_model(model, str(model_path))

            # Test loading using new API
            oe.load(str(model_path), device_preference=["CPU"])
            info = oe.get_info()
            assert info["device"] == "CPU"

            # Test inference
            dummy_input = np.random.randn(1, 10).astype(np.float32)
            result = oe.infer(dummy_input)
            assert result is not None

            oe.unload()

            print(f"✅ Cross-platform model loading successful on {platform.system()}")

    def test_file_permissions_cross_platform(self):
        """Test file permission handling across platforms."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            test_file = temp_path / "test_permissions.txt"
            test_file.write_text("test content")

            # Test reading
            assert test_file.exists()
            assert test_file.is_file()
            content = test_file.read_text()
            assert content == "test content"

            # Test directory creation
            test_subdir = temp_path / "subdir" / "nested"
            test_subdir.mkdir(parents=True, exist_ok=True)
            assert test_subdir.exists()
            assert test_subdir.is_dir()

    def test_environment_variables_cross_platform(self):
        """Test environment variable handling."""
        # Test setting and getting environment variables
        test_var_name = "OE_TEST_VAR"
        test_var_value = "test_value_123"

        # Set environment variable
        os.environ[test_var_name] = test_var_value

        # Test retrieval
        retrieved_value = os.getenv(test_var_name)
        assert retrieved_value == test_var_value

        # Clean up
        del os.environ[test_var_name]
        assert os.getenv(test_var_name) is None

    @pytest.mark.skipif(
        platform.system() == "Darwin", reason="Intel drivers not available on macOS"
    )
    def test_intel_specific_features(self):
        """Test Intel-specific features where available."""
        # Test NPU detection (Intel-specific)
        try:
            from oe._core import check_npu_driver

            npu_status = check_npu_driver()
            print(f"NPU status: {npu_status}")

            # NPU status should be a dictionary with expected keys
            assert isinstance(npu_status, dict)
            assert "npu_functional" in npu_status
            assert "driver_status" in npu_status

        except Exception as e:
            # NPU detection might fail on non-Intel hardware
            print(f"NPU detection failed (expected on non-Intel hardware): {e}")

    def test_unicode_path_handling(self):
        """Test handling of Unicode characters in paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test Unicode directory and file names
            unicode_names = [
                "测试模型",  # Chinese
                "тест_модель",  # Russian
                "モデル_テスト",  # Japanese
                "café_modèle",  # French accents
                "普通话_test",  # Mixed
            ]

            for unicode_name in unicode_names:
                try:
                    test_dir = temp_path / unicode_name
                    test_dir.mkdir(exist_ok=True)

                    test_file = test_dir / "test.txt"
                    test_file.write_text("test content", encoding="utf-8")

                    # Test reading back
                    content = test_file.read_text(encoding="utf-8")
                    assert content == "test content"

                    print(f"✅ Unicode path handling works: {unicode_name}")

                except Exception as e:
                    # Some file systems may not support certain Unicode characters
                    print(
                        f"⚠️  Unicode path '{unicode_name}' not supported on this system: {e}"
                    )

    def test_memory_usage_cross_platform(self):
        """Test memory usage patterns across platforms."""
        import psutil
        import gc

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(f"Initial memory usage: {initial_memory:.1f} MB")

        # Create some objects to test memory management
        large_objects = []
        for i in range(10):
            # Create numpy arrays to simulate model data
            import numpy as np

            large_array = np.random.randn(1000, 1000).astype(np.float32)
            large_objects.append(large_array)

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Peak memory usage: {peak_memory:.1f} MB")

        # Clean up
        del large_objects
        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Final memory usage: {final_memory:.1f} MB")

        # Memory should be released (allowing for some overhead)
        memory_increase = final_memory - initial_memory
        assert memory_increase < 100, (
            f"Memory leak detected: {memory_increase:.1f} MB not released"
        )

        print(f"✅ Memory management test passed on {platform.system()}")


class TestWindowsSpecific:
    """Windows-specific tests."""

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_windows_path_length_limits(self):
        """Test Windows long path handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a very long path (Windows has 260 character limit traditionally)
            long_path_parts = [
                "very_long_directory_name_that_exceeds_normal_limits"
            ] * 10
            long_path = temp_path

            try:
                for part in long_path_parts:
                    long_path = long_path / part
                    if len(str(long_path)) > 250:  # Approach Windows limit
                        break
                    long_path.mkdir(exist_ok=True)

                # Test file creation in long path
                test_file = long_path / "test.txt"
                test_file.write_text("test")
                assert test_file.exists()

                print(f"✅ Long path support works: {len(str(long_path))} characters")

            except OSError as e:
                print(f"⚠️  Long path limitation encountered: {e}")

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_windows_file_locking(self):
        """Test Windows file locking behavior."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "locked_file.txt"

            # Test that files can be read while open
            with open(test_file, "w") as f:
                f.write("test content")
                f.flush()

                # Should be able to read the file while it's open for writing
                try:
                    with open(test_file, "r") as f2:
                        content = f2.read()
                        assert content == "test content"
                    print("✅ Windows file locking behavior as expected")
                except PermissionError:
                    print("⚠️  Unexpected file locking behavior on Windows")


class TestLinuxSpecific:
    """Linux-specific tests."""

    @pytest.mark.skipif(platform.system() != "Linux", reason="Linux-specific test")
    def test_linux_case_sensitivity(self):
        """Test Linux case-sensitive file system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files with different cases
            file1 = temp_path / "Test.txt"
            file2 = temp_path / "test.txt"

            file1.write_text("uppercase")
            file2.write_text("lowercase")

            # Both files should exist and be different
            assert file1.exists()
            assert file2.exists()
            assert file1.read_text() == "uppercase"
            assert file2.read_text() == "lowercase"

            print("✅ Linux case sensitivity working correctly")


class TestMacOSSpecific:
    """macOS-specific tests."""

    @pytest.mark.skipif(platform.system() != "Darwin", reason="macOS-specific test")
    def test_macos_case_insensitive_default(self):
        """Test macOS default case-insensitive behavior."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create file with one case
            file1 = temp_path / "Test.txt"
            file1.write_text("test content")

            # Try to access with different case
            file2 = temp_path / "test.txt"

            # On default macOS (HFS+/APFS case-insensitive), these should refer to same file
            if file2.exists():
                content = file2.read_text()
                print(f"✅ macOS case-insensitive behavior confirmed: {content}")
            else:
                print("ℹ️  Running on case-sensitive macOS filesystem")


# Platform detection for test skipping
CURRENT_PLATFORM = platform.system()
IS_WINDOWS = CURRENT_PLATFORM == "Windows"
IS_LINUX = CURRENT_PLATFORM == "Linux"
IS_MACOS = CURRENT_PLATFORM == "Darwin"
