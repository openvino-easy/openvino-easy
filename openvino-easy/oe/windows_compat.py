"""Windows compatibility utilities for OpenVINO-Easy."""

import sys
import os
import stat
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Union
import logging


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


def is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == "win32"


def normalize_path(path: Union[str, Path]) -> Path:
    """Normalize path for cross-platform compatibility."""
    path_obj = Path(path)
    
    if is_windows():
        # Handle Windows-specific path issues
        
        # Convert forward slashes to backslashes
        str_path = str(path_obj).replace("/", "\\")
        
        # Handle long path limitation on Windows
        if len(str_path) > 260:
            # Try to use the UNC path prefix for long paths
            if not str_path.startswith("\\\\?\\"):
                if str_path.startswith("\\\\"):
                    # Network path
                    str_path = "\\\\?\\UNC\\" + str_path[2:]
                else:
                    # Local path
                    str_path = "\\\\?\\" + str_path
        
        path_obj = Path(str_path)
    
    return path_obj.resolve()


def safe_rmtree(path: Union[str, Path]) -> bool:
    """Safely remove directory tree with Windows-specific handling."""
    import shutil
    
    path_obj = Path(path)
    if not path_obj.exists():
        return True
    
    try:
        if is_windows():
            # On Windows, handle read-only files and long paths
            def handle_remove_readonly(func, path, exc):
                """Handle removal of read-only files on Windows."""
                if os.path.exists(path):
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
            
            shutil.rmtree(str(path_obj), onerror=handle_remove_readonly)
        else:
            shutil.rmtree(str(path_obj))
        
        return True
    except Exception as e:
        _safe_log_unicode(
            "warning",
            f"⚠️  Failed to remove directory {path_obj}: {e}",
            f"[WARNING] Failed to remove directory {path_obj}: {e}"
        )
        return False


def safe_makedirs(path: Union[str, Path], exist_ok: bool = True) -> bool:
    """Safely create directories with Windows-specific handling."""
    path_obj = normalize_path(path)
    
    try:
        path_obj.mkdir(parents=True, exist_ok=exist_ok)
        return True
    except Exception as e:
        _safe_log_unicode(
            "warning",
            f"⚠️  Failed to create directory {path_obj}: {e}",
            f"[WARNING] Failed to create directory {path_obj}: {e}"
        )
        return False


def get_temp_dir() -> Path:
    """Get temporary directory with Windows compatibility."""
    if is_windows():
        # Use a shorter temp path on Windows to avoid long path issues
        temp_base = os.environ.get("TEMP", tempfile.gettempdir())
        temp_path = Path(temp_base) / "oe_temp"
        safe_makedirs(temp_path)
        return temp_path
    else:
        return Path(tempfile.gettempdir()) / "oe_temp"


def check_path_length(path: Union[str, Path]) -> bool:
    """Check if path length is problematic on Windows."""
    if not is_windows():
        return True
    
    path_str = str(path)
    if len(path_str) > 260:
        _safe_log_unicode(
            "warning",
            f"⚠️  Path may be too long for Windows: {len(path_str)} characters",
            f"[WARNING] Path may be too long for Windows: {len(path_str)} characters"
        )
        return False
    
    return True


def fix_windows_permissions(path: Union[str, Path]) -> bool:
    """Fix Windows file permissions issues."""
    if not is_windows():
        return True
    
    path_obj = Path(path)
    
    try:
        if path_obj.exists():
            # Make sure we can read/write the path
            if path_obj.is_file():
                os.chmod(str(path_obj), stat.S_IREAD | stat.S_IWRITE)
            elif path_obj.is_dir():
                for root, dirs, files in os.walk(str(path_obj)):
                    for d in dirs:
                        dir_path = os.path.join(root, d)
                        os.chmod(dir_path, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
                    for f in files:
                        file_path = os.path.join(root, f)
                        os.chmod(file_path, stat.S_IREAD | stat.S_IWRITE)
        return True
    except Exception as e:
        _safe_log_unicode(
            "warning",
            f"⚠️  Failed to fix permissions for {path_obj}: {e}",
            f"[WARNING] Failed to fix permissions for {path_obj}: {e}"
        )
        return False


def handle_symlink_warning():
    """Handle Windows symlink warnings."""
    if is_windows():
        # Check if Developer Mode is enabled
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock"
            )
            allow_development, _ = winreg.QueryValueEx(key, "AllowDevelopmentWithoutDevLicense")
            winreg.CloseKey(key)
            
            if not allow_development:
                _safe_log_unicode(
                    "warning",
                    "⚠️  Developer Mode not enabled. Some features may not work optimally.",
                    "[WARNING] Developer Mode not enabled. Some features may not work optimally."
                )
                return False
        except Exception:
            # Assume Developer Mode is not enabled if we can't check
            _safe_log_unicode(
                "info",
                "💡 For best experience on Windows, enable Developer Mode in Settings",
                "[INFO] For best experience on Windows, enable Developer Mode in Settings"
            )
            return False
    
    return True


def safe_copy_file(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """Safely copy file with Windows-specific handling."""
    import shutil
    
    src_path = normalize_path(src)
    dst_path = normalize_path(dst)
    
    # Ensure destination directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if is_windows():
            # On Windows, handle long paths and permissions
            fix_windows_permissions(src_path)
            
        shutil.copy2(str(src_path), str(dst_path))
        
        if is_windows():
            fix_windows_permissions(dst_path)
        
        return True
    except Exception as e:
        _safe_log_unicode(
            "warning",
            f"⚠️  Failed to copy {src_path} to {dst_path}: {e}",
            f"[WARNING] Failed to copy {src_path} to {dst_path}: {e}"
        )
        return False


def get_safe_cache_dir(base_dir: Optional[Union[str, Path]] = None) -> Path:
    """Get a safe cache directory path for Windows."""
    if base_dir:
        cache_path = normalize_path(base_dir)
    else:
        if is_windows():
            # Use LOCALAPPDATA on Windows to avoid OneDrive sync issues
            local_app_data = os.environ.get("LOCALAPPDATA")
            if local_app_data:
                cache_path = Path(local_app_data) / "openvino-easy"
            else:
                cache_path = Path.home() / "AppData" / "Local" / "openvino-easy"
        else:
            cache_path = Path.home() / ".cache" / "openvino-easy"
    
    # Ensure path length is reasonable
    if is_windows() and len(str(cache_path)) > 200:
        # Use a shorter path
        cache_path = Path("C:/oe_cache") if Path("C:/").exists() else get_temp_dir()
    
    # Create the directory
    safe_makedirs(cache_path)
    
    return cache_path


def check_windows_compatibility() -> dict:
    """Check Windows compatibility and return status."""
    status = {
        "platform": sys.platform,
        "is_windows": is_windows(),
        "issues": [],
        "recommendations": [],
    }
    
    if not is_windows():
        status["compatible"] = True
        return status
    
    # Check Python version
    if sys.version_info < (3, 8):
        status["issues"].append(f"Python {sys.version_info.major}.{sys.version_info.minor} may have issues on Windows")
        status["recommendations"].append("Upgrade to Python 3.8+")
    
    # Check Developer Mode
    if not handle_symlink_warning():
        status["recommendations"].append("Enable Developer Mode in Windows Settings for better compatibility")
    
    # Check available disk space
    try:
        import shutil
        cache_dir = get_safe_cache_dir()
        total, used, free = shutil.disk_usage(str(cache_dir))
        free_gb = free / (1024**3)
        if free_gb < 5:
            status["issues"].append(f"Low disk space: {free_gb:.1f}GB available")
            status["recommendations"].append("Free up at least 5GB of disk space")
    except Exception as e:
        status["issues"].append(f"Could not check disk space: {e}")
    
    # Check path length support
    test_long_path = "C:/" + "a" * 250 + "/test"
    if not check_path_length(test_long_path):
        status["issues"].append("Long path names may cause issues")
        status["recommendations"].append("Use shorter cache directory paths")
    
    status["compatible"] = len(status["issues"]) == 0
    
    return status