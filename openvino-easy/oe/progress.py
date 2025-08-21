"""Progress indicators for OpenVINO-Easy operations."""

import sys
import time
import threading
from typing import Optional, Dict, Any
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


class ProgressIndicator:
    """Simple progress indicator for long-running operations."""

    def __init__(self, description: str, show_spinner: bool = True):
        self.description = description
        self.show_spinner = show_spinner
        self.active = False
        self.thread = None
        self.start_time = None

        # Spinner characters (ASCII safe)
        self.spinner_chars = ["|", "/", "-", "\\"]
        self.spinner_index = 0

    def start(self):
        """Start the progress indicator."""
        if self.active:
            return

        self.active = True
        self.start_time = time.time()

        if self.show_spinner:
            self.thread = threading.Thread(target=self._spinner_loop, daemon=True)
            self.thread.start()
        else:
            # Just log the start
            _safe_log_unicode(
                "info",
                f"🔄 {self.description}...",
                f"[PROGRESS] {self.description}..."
            )

    def stop(self, success: bool = True, message: Optional[str] = None):
        """Stop the progress indicator."""
        if not self.active:
            return

        self.active = False

        if self.thread:
            self.thread.join(timeout=0.1)

        elapsed = time.time() - self.start_time if self.start_time else 0

        if message:
            final_message = message
        else:
            final_message = f"{self.description} {'completed' if success else 'failed'}"

        if success:
            _safe_log_unicode(
                "info",
                f"✅ {final_message} ({elapsed:.1f}s)",
                f"[SUCCESS] {final_message} ({elapsed:.1f}s)"
            )
        else:
            _safe_log_unicode(
                "error",
                f"❌ {final_message} ({elapsed:.1f}s)",
                f"[FAILED] {final_message} ({elapsed:.1f}s)"
            )

    def _spinner_loop(self):
        """Run the spinner in a separate thread."""
        while self.active:
            try:
                # Use carriage return to overwrite the line
                spinner_char = self.spinner_chars[self.spinner_index]
                elapsed = time.time() - self.start_time if self.start_time else 0

                # Only show spinner in terminal, not in logs
                if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
                    print(f"\r{spinner_char} {self.description}... ({elapsed:.0f}s)",
                          end="", file=sys.stderr, flush=True)

                self.spinner_index = (self.spinner_index + 1) % len(self.spinner_chars)
                time.sleep(0.2)

            except Exception:
                # Silently handle any output errors
                break

        # Clear the spinner line when done
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            print("\r" + " " * 80 + "\r", end="", file=sys.stderr, flush=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        self.stop(success=success)


class DownloadProgress:
    """Progress tracking for file downloads."""

    def __init__(self, description: str):
        self.description = description
        self.total_size = 0
        self.downloaded_size = 0
        self.start_time = None
        self.last_update = 0

    def start(self, total_size: Optional[int] = None):
        """Start tracking download progress."""
        self.start_time = time.time()
        self.total_size = total_size or 0
        self.downloaded_size = 0

        _safe_log_unicode(
            "info",
            f"📥 Starting download: {self.description}",
            f"[DOWNLOAD] Starting download: {self.description}"
        )

    def update(self, downloaded_bytes: int):
        """Update download progress."""
        self.downloaded_size = downloaded_bytes
        current_time = time.time()

        # Only update every 2 seconds to avoid spam
        if current_time - self.last_update < 2.0:
            return

        self.last_update = current_time

        if self.total_size > 0:
            percent = (self.downloaded_size / self.total_size) * 100
            progress_msg = f"📥 {self.description}: {percent:.1f}% ({self._format_bytes(self.downloaded_size)}/{self._format_bytes(self.total_size)})"
            ascii_msg = f"[DOWNLOAD] {self.description}: {percent:.1f}% ({self._format_bytes(self.downloaded_size)}/{self._format_bytes(self.total_size)})"
        else:
            progress_msg = f"📥 {self.description}: {self._format_bytes(self.downloaded_size)}"
            ascii_msg = f"[DOWNLOAD] {self.description}: {self._format_bytes(self.downloaded_size)}"

        _safe_log_unicode("info", progress_msg, ascii_msg)

    def complete(self, success: bool = True):
        """Mark download as complete."""
        elapsed = time.time() - self.start_time if self.start_time else 0

        if success:
            _safe_log_unicode(
                "info",
                f"✅ Download completed: {self.description} ({self._format_bytes(self.downloaded_size)}, {elapsed:.1f}s)",
                f"[SUCCESS] Download completed: {self.description} ({self._format_bytes(self.downloaded_size)}, {elapsed:.1f}s)"
            )
        else:
            _safe_log_unicode(
                "error",
                f"❌ Download failed: {self.description}",
                f"[FAILED] Download failed: {self.description}"
            )

    def _format_bytes(self, bytes_val: int) -> str:
        """Format byte count as human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f}{unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f}TB"


class ConversionProgress:
    """Progress tracking for model conversion."""

    def __init__(self, model_type: str, description: str):
        self.model_type = model_type
        self.description = description
        self.start_time = None
        self.steps = []
        self.current_step = 0

    def start(self, steps: Optional[list] = None):
        """Start tracking conversion progress."""
        self.start_time = time.time()
        self.steps = steps or ["Converting model"]
        self.current_step = 0

        _safe_log_unicode(
            "info",
            f"🔧 Starting conversion: {self.description}",
            f"[CONVERSION] Starting conversion: {self.description}"
        )

    def next_step(self, step_description: Optional[str] = None):
        """Move to the next conversion step."""
        if step_description:
            self.steps.append(step_description)

        self.current_step += 1

        if self.current_step <= len(self.steps):
            current_desc = self.steps[self.current_step - 1]
            progress = f"({self.current_step}/{len(self.steps)})"

            _safe_log_unicode(
                "info",
                f"🔧 {current_desc} {progress}",
                f"[CONVERSION] {current_desc} {progress}"
            )

    def complete(self, success: bool = True, output_info: Optional[Dict[str, Any]] = None):
        """Mark conversion as complete."""
        elapsed = time.time() - self.start_time if self.start_time else 0

        if success:
            info_str = ""
            if output_info:
                info_parts = []
                if "inputs" in output_info:
                    info_parts.append(f"{output_info['inputs']} inputs")
                if "outputs" in output_info:
                    info_parts.append(f"{output_info['outputs']} outputs")
                if info_parts:
                    info_str = f" ({', '.join(info_parts)})"

            _safe_log_unicode(
                "info",
                f"✅ Conversion completed: {self.description}{info_str} ({elapsed:.1f}s)",
                f"[SUCCESS] Conversion completed: {self.description}{info_str} ({elapsed:.1f}s)"
            )
        else:
            _safe_log_unicode(
                "error",
                f"❌ Conversion failed: {self.description} ({elapsed:.1f}s)",
                f"[FAILED] Conversion failed: {self.description} ({elapsed:.1f}s)"
            )


# Convenience functions for common operations
def with_progress(description: str, show_spinner: bool = True):
    """Decorator to add progress indication to a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with ProgressIndicator(description, show_spinner):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def download_with_progress(description: str):
    """Create a download progress tracker."""
    return DownloadProgress(description)


def conversion_with_progress(model_type: str, description: str):
    """Create a conversion progress tracker."""
    return ConversionProgress(model_type, description)
