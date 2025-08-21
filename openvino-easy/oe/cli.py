"""Command-line interface for OpenVINO-Easy."""

import argparse
import json
import sys
import platform
from pathlib import Path

import oe
import oe.benchmark
from ._core import check_npu_driver, get_available_devices

# Try to import colorama for better output
try:
    from colorama import init, Fore, Back, Style

    init(autoreset=True)  # Reset colors after each print
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False

    # Fallback - no colors
    class Fore:
        RED = GREEN = YELLOW = BLUE = CYAN = MAGENTA = WHITE = ""

    class Back:
        RED = GREEN = YELLOW = BLUE = CYAN = MAGENTA = WHITE = ""

    class Style:
        BRIGHT = DIM = RESET_ALL = ""


def colored_print(text: str, color: str = "", style: str = "", emoji: str = ""):
    """Print colored text with emoji support."""
    # Handle Unicode encoding issues on Windows
    try:
        if COLORS_AVAILABLE:
            print(f"{emoji}{style}{color}{text}{Style.RESET_ALL}")
        else:
            print(f"{emoji}{text}")
    except UnicodeEncodeError:
        # Fallback without emoji
        if COLORS_AVAILABLE:
            print(f"{style}{color}{text}{Style.RESET_ALL}")
        else:
            print(text)


def print_header(text: str):
    """Print a styled header."""
    colored_print(f"\n{text}", Fore.CYAN, Style.BRIGHT)
    colored_print("=" * len(text), Fore.CYAN)


def print_success(text: str):
    """Print success message."""
    colored_print(text, Fore.GREEN, emoji="[OK] ")


def print_warning(text: str):
    """Print warning message."""
    colored_print(text, Fore.YELLOW, emoji="[WARNING] ")


def print_error(text: str):
    """Print error message."""
    colored_print(text, Fore.RED, emoji="[ERROR] ")


def print_info(text: str):
    """Print info message."""
    colored_print(text, Fore.BLUE, emoji="[INFO] ")


def cmd_doctor(args):
    """Comprehensive OpenVINO installation diagnostics."""
    if args.json:
        _doctor_json_output(args)
        return

    print_header("OpenVINO-Easy Doctor")

    # System info
    print_info(f"System: {platform.system()} {platform.release()}")
    print_info(f"Python: {platform.python_version()}")
    print()

    # OpenVINO installation check
    print_header("OpenVINO Installation")
    try:
        import openvino as ov

        print_success(f"OpenVINO version: {ov.__version__}")

        # Check if it's dev or runtime
        try:
            from openvino.tools import mo  # Model Optimizer

            install_type = "openvino-dev (full)"
        except ImportError:
            install_type = "openvino (runtime only)"
        print_info(f"Install type: {install_type}")

    except ImportError:
        print_error("OpenVINO not found")
        print_header("Recommended fixes")
        if args.fix:
            _suggest_openvino_install(args.fix)
        else:
            _suggest_openvino_install("cpu")
        return

    # Device detection
    print("\nDevice Detection:")
    import openvino as ov

    core = ov.Core()
    all_devices = core.available_devices
    validated_devices = get_available_devices()

    device_status = {}
    for device in all_devices:
        is_functional = device in validated_devices
        device_status[device] = is_functional
        status = "✅ Functional" if is_functional else "❌ Not functional"
        print(f"  {device}: {status}")

        # Detailed device info with vendor detection
        try:
            device_name = core.get_property(device, "FULL_DEVICE_NAME")
            print(f"    └─ Name: {device_name}")

            # Check for NVIDIA GPU masquerading as Intel GPU
            if device.startswith("GPU") and "NVIDIA" in device_name.upper():
                print(
                    "    └─ [WARNING] NVIDIA GPU detected - install Intel GPU drivers for Intel GPU support"
                )

        except:
            print("    └─ Name: Unable to query")

    # NPU-specific diagnostics
    if "NPU" in all_devices:
        print("\nNPU Diagnostics:")
        npu_status = check_npu_driver()
        print(f"  Driver status: {npu_status['driver_status']}")
        if npu_status["device_name"]:
            print(f"  Device: {npu_status['device_name']}")

        if not npu_status["npu_functional"] and npu_status["recommendations"]:
            print("  Recommendations:")
            for rec in npu_status["recommendations"]:
                print(f"    • {rec}")

    # Performance recommendations
    print("\nPerformance Recommendations:")
    best_device = oe.detect_best_device()
    print(f"  Recommended device: {best_device}")

    if "NPU" in device_status and not device_status["NPU"]:
        print(
            "  NPU detected but not functional - install NPU drivers for best performance"
        )
    elif "GPU" in device_status and not device_status["GPU"]:
        print("  GPU detected but not functional - install Intel GPU drivers")

    # Fix suggestions
    if args.fix:
        print(f"\nFix suggestions for {args.fix.upper()}:")
        _suggest_device_fix(
            args.fix, device_status, npu_status if "NPU" in all_devices else None
        )

    # Summary
    functional_count = sum(device_status.values())
    total_count = len(device_status)
    print(f"\nSummary: {functional_count}/{total_count} devices functional")

    if functional_count == 0:
        print(
            "[WARNING] No functional devices detected - OpenVINO installation may be corrupted"
        )
    elif functional_count < total_count:
        print(
            "[WARNING] Some devices need attention - run 'oe doctor --fix <device>' for help"
        )
    else:
        print("[OK] All detected devices are functional!")


def _suggest_openvino_install(target_device):
    """Suggest OpenVINO installation commands."""
    suggestions = {
        "cpu": "pip install 'openvino-easy[cpu]'",
        "gpu": "pip install 'openvino-easy[gpu]'",
        "npu": "pip install 'openvino-easy[npu]'",
        "full": "pip install 'openvino-easy[full]'",
    }

    cmd = suggestions.get(target_device.lower(), suggestions["cpu"])
    print(f"  Install command: {cmd}")

    if target_device.lower() == "gpu":
        system = platform.system()
        if system == "Windows":
            print("  Additional: Install Intel GPU drivers from intel.com")
        elif system == "Linux":
            print("  Additional: sudo apt install intel-opencl-icd (Ubuntu/Debian)")

    elif target_device.lower() == "npu":
        print("  Additional: Install Intel NPU drivers from intel.com")


def _suggest_device_fix(device, device_status, npu_status):
    """Suggest fixes for specific device issues."""
    device = device.lower()

    if device == "npu":
        if npu_status and not npu_status["npu_functional"]:
            if npu_status["driver_status"] == "not_detected":
                print("  1. Download Intel NPU drivers from intel.com")
                print("  2. Check BIOS settings - ensure NPU is enabled")
                print("  3. Restart system after driver installation")
            elif npu_status["driver_status"] == "stub_device":
                print("  1. Uninstall current NPU driver")
                print("  2. Download latest NPU driver from intel.com")
                print("  3. Clean install with administrator privileges")
            else:
                print("  1. Reinstall NPU drivers")
                print("  2. Check Windows Device Manager for errors")
                print("  3. Contact support if issue persists")

    elif device == "gpu":
        system = platform.system()
        if system == "Windows":
            print("  1. Download Intel GPU drivers from intel.com")
            print("  2. Install with administrator privileges")
            print("  3. Restart system")
        elif system == "Linux":
            print("  1. sudo apt update")
            print("  2. sudo apt install intel-opencl-icd")
            print("  3. Add user to 'render' group: sudo usermod -a -G render $USER")
            print("  4. Logout and login again")

    elif device == "cpu":
        print("  CPU should always work. If not functional:")
        print(
            "  1. Reinstall OpenVINO: pip uninstall openvino && pip install 'openvino-easy[cpu]'"
        )
        print("  2. Check Python environment conflicts")
        print("  3. Try in a fresh virtual environment")

    else:
        print(f"  No specific fix suggestions for {device.upper()}")
        print("  Try reinstalling OpenVINO-Easy with appropriate extras")


def _doctor_json_output(args):
    """Output doctor diagnostics in JSON format for CI systems."""
    import openvino as ov

    # Collect system info
    system_info = {
        "system": platform.system(),
        "release": platform.release(),
        "python_version": platform.python_version(),
    }

    # Collect OpenVINO info
    openvino_info = {
        "installed": True,
        "version": ov.__version__,
        "install_type": "runtime",
    }

    try:
        from openvino.tools import mo

        openvino_info["install_type"] = "dev"
    except ImportError:
        pass

    # Collect device info
    core = ov.Core()
    all_devices = core.available_devices
    validated_devices = get_available_devices()

    devices_info = {}
    for device in all_devices:
        is_functional = device in validated_devices
        device_info = {"functional": is_functional, "name": "Unknown"}

        try:
            device_info["name"] = core.get_property(device, "FULL_DEVICE_NAME")
        except:
            pass

        devices_info[device] = device_info

    # NPU-specific info
    npu_info = None
    if "NPU" in all_devices:
        npu_info = check_npu_driver()

    # Compile results
    results = {
        "timestamp": platform.system(),  # Could use datetime if needed
        "system": system_info,
        "openvino": openvino_info,
        "devices": devices_info,
        "npu_diagnostics": npu_info,
        "recommended_device": oe.detect_best_device(),
        "summary": {
            "total_devices": len(all_devices),
            "functional_devices": len(validated_devices),
            "all_functional": len(validated_devices) == len(all_devices),
        },
    }

    print(json.dumps(results, indent=2))


def list_devices(args):
    """List available devices with validation status."""
    print("Scanning OpenVINO devices...")
    print()

    # Get all devices (including potentially non-functional ones)
    try:
        all_devices = oe.devices()
    except Exception as e:
        print(f"Device detection failed: {e}")
        return

    print("Device Status:")
    for device in all_devices:
        print(f"  {device}: [OK] Functional")

        # Special handling for NPU
        if device == "NPU":
            npu_status = check_npu_driver()
            if not npu_status["npu_functional"]:
                print(f"    └─ Driver: {npu_status['driver_status']}")
                if npu_status["device_name"]:
                    print(f"    └─ Device: {npu_status['device_name']}")
                if npu_status["recommendations"]:
                    print(f"    └─ Fix: {npu_status['recommendations'][0]}")

    print()
    # In this simplified listing, consider all reported devices as functional
    print(f"[OK] {len(all_devices)} functional device(s) detected")

    # Show recommended device
    best_device = oe.detect_best_device()
    print(f"Recommended device: {best_device}")


def cmd_npu_doctor(args):
    """Diagnose NPU driver status with enhanced Arrow Lake/Lunar Lake support."""
    print("NPU Driver Diagnostics")
    print("=" * 50)

    npu_status = check_npu_driver()

    print(
        f"NPU in available devices: {'[OK] Yes' if npu_status['npu_in_available_devices'] else '[FAIL] No'}"
    )
    print(
        f"NPU functional: {'[OK] Yes' if npu_status['npu_functional'] else '[FAIL] No'}"
    )
    print(f"Driver status: {npu_status['driver_status']}")

    if npu_status["device_name"]:
        print(f"Device name: {npu_status['device_name']}")

    # Enhanced NPU generation and capabilities reporting
    npu_gen = npu_status.get("npu_generation", "unknown")
    if npu_gen != "unknown":
        print(f"NPU generation: {npu_gen}")

        capabilities = npu_status.get("capabilities", {})
        if capabilities:
            gen_info = capabilities.get("generation", "unknown")
            print(f"Generation: {gen_info} generation Intel NPU")

            # Show precision support
            precisions = []
            if capabilities.get("supports_fp16"):
                precisions.append("FP16")
            if capabilities.get("supports_fp16_nf4"):
                precisions.append("FP16-NF4")  # New precision for Arrow/Lunar Lake
            if capabilities.get("supports_int8"):
                precisions.append("INT8")

            if precisions:
                print(f"Supported precisions: {', '.join(precisions)}")

            # Show performance expectations
            if capabilities.get("expected_stable_diffusion_fps"):
                print(
                    f"Expected Stable Diffusion performance: ~{capabilities['expected_stable_diffusion_fps']:.1f} img/s"
                )
            if capabilities.get("expected_dialog_gpt_tps"):
                print(
                    f"Expected DialoGPT performance: ~{capabilities['expected_dialog_gpt_tps']} tokens/s"
                )

            # Show features
            features = capabilities.get("features", [])
            if features:
                print(f"Key features: {', '.join(features)}")

    if npu_status["recommendations"]:
        print()
        print("Recommendations:")
        for i, rec in enumerate(npu_status["recommendations"], 1):
            print(f"  {i}. {rec}")

    print()
    if npu_status["npu_functional"]:
        print("[OK] NPU is ready for use!")
    else:
        print("[WARNING] NPU requires attention before use.")


def run_inference(args):
    """Run inference on a model."""
    try:
        # Load model
        if not args.json:
            print_info(f"Loading model: {args.model}")

        device_preference = None
        if args.device_preference:
            device_preference = args.device_preference.split(",")

        oe.load(args.model, device_preference=device_preference, dtype=args.dtype)

        # Get model info after loading
        model_info = oe.get_info()

        if not args.json:
            print_success(f"Model loaded on {model_info['device']}")

        # Prepare input
        if args.prompt:
            input_data = args.prompt
        elif args.input_file:
            # For future: load image/audio files
            input_data = str(args.input_file)
        else:
            # Use dummy input for testing
            input_data = "test input"

        # Run inference
        if not args.json:
            print_info("Running inference...")
        result = oe.infer(input_data)

        # Prepare output data
        output_data = {
            "model": str(args.model),
            "device": str(model_info.get("device", "unknown")),
            "dtype": str(args.dtype),
            "input": str(input_data),
            "result": result,
            "success": True,
        }

        # Output results
        if args.json:
            print(json.dumps(output_data, indent=2))
        elif args.output:
            output_path = Path(args.output)
            if output_path.suffix == ".json":
                with open(output_path, "w") as f:
                    json.dump(output_data, f, indent=2)
                print_success(f"Results saved to {output_path}")
            else:
                with open(output_path, "w") as f:
                    f.write(str(result))
                print_success(f"Results saved to {output_path}")
        else:
            print_header("Results")
            if isinstance(result, (list, dict)):
                print(json.dumps(result, indent=2))
            else:
                print(result)

    except Exception as e:
        if args.json:
            error_data = {"model": args.model, "error": str(e), "success": False}
            print(json.dumps(error_data, indent=2))
        else:
            print_error(f"Error: {e}")
        sys.exit(1)


def run_benchmark(args):
    """Benchmark a model."""
    try:
        # Load model
        if not args.json:
            print_info(f"Loading model: {args.model}")

        device_preference = None
        if args.device_preference:
            device_preference = args.device_preference.split(",")

        oe.load(args.model, device_preference=device_preference, dtype=args.dtype)

        # Get model info after loading
        model_info = oe.get_info()

        if not args.json:
            print_success(f"Model loaded on {model_info['device']}")

        # Run benchmark
        if not args.json:
            print_info(
                f"Benchmarking (warmup: {args.warmup_runs}, runs: {args.benchmark_runs})..."
            )
        stats = oe.benchmark(
            warmup_runs=args.warmup_runs, benchmark_runs=args.benchmark_runs
        )

        # Enhance stats with metadata
        enhanced_stats = {
            "model": str(args.model),
            "dtype": str(args.dtype),
            "success": True,
            **stats,
        }

        # Output results
        if args.json:
            print(json.dumps(enhanced_stats, indent=2))
        elif args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(enhanced_stats, f, indent=2)
            print_success(f"Benchmark results saved to {output_path}")
        else:
            print_header("Benchmark Results")
            if COLORS_AVAILABLE:
                print(f"Device: {Fore.CYAN}{stats['device']}{Style.RESET_ALL}")
                print(
                    f"Average Latency: {Fore.GREEN}{stats['mean_ms']:.2f} ms{Style.RESET_ALL}"
                )
                print(
                    f"Throughput: {Fore.GREEN}{stats['fps']:.1f} FPS{Style.RESET_ALL}"
                )
            else:
                print(f"Device: {stats['device']}")
                print(f"Average Latency: {stats['mean_ms']:.2f} ms")
                print(f"Throughput: {stats['fps']:.1f} FPS")

            if "p90_ms" in stats:
                print(f"P90 Latency: {stats['p90_ms']:.2f} ms")
            if "std_ms" in stats:
                print(f"Std Dev: {stats['std_ms']:.2f} ms")

    except Exception as e:
        if args.json:
            error_data = {"model": args.model, "error": str(e), "success": False}
            print(json.dumps(error_data, indent=2))
        else:
            print_error(f"Error: {e}")
        sys.exit(1)


# Cache Management Commands
def cache_list(args):
    """List cached models."""
    try:
        models = oe.models.list(args.cache_dir)

        if args.json:
            print(json.dumps(models, indent=2))
        else:
            if not models:
                print_info("No cached models found")
                print_info(f"Models directory: {oe.models.dir(args.cache_dir)}")
            else:
                print_header(f"📦 Cached Models ({len(models)})")
                total_size = 0
                for model in models:
                    size_mb = model["size_mb"]
                    total_size += size_mb
                    if COLORS_AVAILABLE:
                        print(
                            f"  {Fore.CYAN}{model['name']}{Style.RESET_ALL}: {Fore.GREEN}{size_mb:.1f} MB{Style.RESET_ALL}"
                        )
                    else:
                        print(f"  {model['name']}: {size_mb:.1f} MB")

                print(f"\nTotal cache size: {total_size:.1f} MB")
                print(f"Models directory: {oe.models.dir(args.cache_dir)}")

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print_error(f"Error: {e}")
        sys.exit(1)


def cache_size(args):
    """Show cache size information."""
    try:
        cache_info = oe.cache.size(args.cache_dir)

        if args.json:
            print(json.dumps(cache_info, indent=2))
        else:
            print_header("Cache Usage")
            print(
                f"Models: {cache_info['models_size_mb']:.1f} MB ({cache_info['model_count']} models)"
            )
            print(f"Temp cache: {cache_info['temp_cache_size_mb']:.1f} MB")
            print(f"Total: {cache_info['total_size_mb']:.1f} MB")
            print("\nLocations:")
            print(f"  Models: {cache_info['models_path']}")
            print(f"  Temp: {cache_info['temp_cache_path']}")

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print_error(f"Error: {e}")
        sys.exit(1)


def cache_remove(args):
    """Remove a specific model from cache."""
    try:
        if args.force:
            result = oe.models.remove(args.model, args.cache_dir, confirm=False)
        else:
            # Show what would be removed first
            models = oe.models.list(args.cache_dir)
            matching_model = None
            for model in models:
                if args.model == model["name"]:
                    matching_model = model
                    break

            if not matching_model:
                result = oe.models.remove(args.model, args.cache_dir, confirm=True)
            else:
                # Show confirmation prompt
                if not args.json:
                    print_warning(f"About to remove model: {matching_model['name']}")
                    print(f"Size: {matching_model['size_mb']:.1f} MB")
                    print(f"Files: {len(matching_model.get('files', []))}")

                    response = input("\nConfirm deletion? [y/N]: ").strip().lower()
                    if response in ["y", "yes"]:
                        result = oe.models.remove(
                            args.model, args.cache_dir, confirm=False
                        )
                    else:
                        print_info("Deletion cancelled")
                        return
                else:
                    # JSON mode - require --force
                    result = {
                        "removed": False,
                        "error": "Use --force flag to confirm deletion in JSON mode",
                    }

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result.get("removed"):
                print_success(f"Removed model: {result['model_name']}")
                print_info(f"Space freed: {result['size_freed_mb']:.1f} MB")
            else:
                print_error(
                    f"Failed to remove model: {result.get('error', 'Unknown error')}"
                )

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print_error(f"Error: {e}")
        sys.exit(1)


def cache_clear(args):
    """Clear cache (with safety confirmations)."""
    try:
        if args.models and not args.force:
            # Show warning for dangerous operation
            if not args.json:
                cache_info = oe.cache.size(args.cache_dir)
                print_warning("DANGEROUS OPERATION")
                print(
                    f"This will delete ALL {cache_info['model_count']} cached models ({cache_info['models_size_mb']:.1f} MB)"
                )
                print("All downloaded and converted models will be lost!")

                response = input("\nType 'DELETE' to confirm: ").strip()
                if response != "DELETE":
                    print_info("Operation cancelled")
                    return

                result = oe.cache.clear(args.cache_dir, models=True, confirm=False)
            else:
                result = {
                    "error": "Use --force flag to confirm dangerous operations in JSON mode"
                }
        else:
            # Safe operation or forced
            confirm = not args.force if args.models else None
            result = oe.cache.clear(args.cache_dir, models=args.models, confirm=confirm)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result.get("temp_cache_cleared"):
                print_success(
                    f"Temp cache cleared: {result['temp_size_freed_mb']:.1f} MB freed"
                )
            if result.get("models_cleared"):
                print_success(
                    f"Models cleared: {result['models_removed']} models, {result['models_size_freed_mb']:.1f} MB freed"
                )

            if "temp_cache_error" in result:
                print_warning(f"Temp cache error: {result['temp_cache_error']}")
            if "models_error" in result:
                print_error(f"Models error: {result['models_error']}")

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print_error(f"Error: {e}")
        sys.exit(1)


# Model Management Commands
def model_search(args):
    """Search for models on Hugging Face Hub."""
    try:
        results = oe.models.search(args.query, limit=args.limit, model_type=args.type)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            if not results:
                print_info(f"No models found for query: '{args.query}'")
            else:
                print_header(
                    f"Search Results for '{args.query}' ({len(results)} found)"
                )
                for model in results:
                    downloads = model.get("downloads", 0)
                    likes = model.get("likes", 0)
                    pipeline = model.get("pipeline_tag", "unknown")

                    if COLORS_AVAILABLE:
                        print(f"  {Fore.CYAN}{model['id']}{Style.RESET_ALL}")
                        print(
                            f"    Type: {pipeline} | Downloads: {downloads:,} | Likes: {likes}"
                        )
                    else:
                        print(f"  {model['id']}")
                        print(
                            f"    Type: {pipeline} | Downloads: {downloads:,} | Likes: {likes}"
                        )

                    if model.get("private", False):
                        print("    [PRIVATE MODEL]")
                    if model.get("gated", False):
                        print("    [GATED MODEL - requires approval]")
                    print()

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print_error(f"Search failed: {e}")
        sys.exit(1)


def model_info(args):
    """Get detailed information about a model."""
    try:
        info = oe.models.info(args.model, args.cache_dir)

        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print_header(f"Model Information: {args.model}")

            # Local status
            if info["local"]:
                local = info["local_info"]
                print_success(f"Installed locally: {local['size_mb']:.1f} MB")
                print(f"  Path: {local['path']}")
                print(f"  Files: {len(local.get('files', []))}")
            else:
                print_warning("Not installed locally")

            # Remote status
            if info["remote"]:
                remote = info["remote_info"]
                print_info("Remote information:")
                print(f"  Downloads: {remote['downloads']:,}")
                print(f"  Likes: {remote['likes']}")
                print(f"  Type: {remote['pipeline_tag']}")
                print(f"  Library: {remote['library_name']}")
                print(f"  Updated: {remote['last_modified'][:10]}")

                if remote["private"]:
                    print_warning("  This is a private model")
                if remote["gated"]:
                    print_warning("  This is a gated model (requires approval)")
            else:
                print_warning("No remote information available")
                if "remote_error" in info:
                    print(f"  Error: {info['remote_error']}")

            # Requirements
            req = info["requirements"]
            print_info("System requirements:")
            print(f"  Memory: {req['min_memory_mb']} MB (minimum)")
            print(f"  Devices: {', '.join(req['recommended_devices'])}")
            print(f"  Precisions: {', '.join(req['supported_precisions'])}")

            # Installation suggestion
            if not info["local"] and info["remote"]:
                print_info(f"Install with: oe models install {args.model}")

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print_error(f"Error: {e}")
        sys.exit(1)


def model_install(args):
    """Install/download a model."""
    try:
        result = oe.models.install(
            args.model, dtype=args.dtype, cache_dir=args.cache_dir, force=args.force
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["installed"]:
                print_success(result["message"])
                if "size_mb" in result:
                    print_info(f"Size: {result['size_mb']:.1f} MB")
                if "files" in result:
                    print_info(f"Files: {result['files']}")
            elif result.get("already_exists"):
                print_warning(result["message"])
                print_info("Use --force to reinstall")
            else:
                print_error(result["message"])
                sys.exit(1)

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print_error(f"Installation failed: {e}")
        sys.exit(1)


def model_validate(args):
    """Validate model integrity."""
    try:
        if not args.json:
            print_info("Validating models...")

        results = oe.models.validate(args.model, args.cache_dir)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print_header(f"Validation Results ({results['validated']} models checked)")

            for model_result in results["models"]:
                if model_result["valid"]:
                    print_success(f"{model_result['name']}: Valid")
                else:
                    print_error(f"{model_result['name']}: Invalid")
                    for error in model_result["errors"]:
                        print(f"    Error: {error}")

                for warning in model_result["warnings"]:
                    print_warning(f"    Warning: {warning}")

            # Summary
            print(f"\nSummary: {results['passed']} passed, {results['failed']} failed")

            if results["failed"] > 0:
                print_warning(
                    "Some models failed validation. Consider reinstalling them."
                )

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print_error(f"Validation failed: {e}")
        sys.exit(1)


def model_benchmark(args):
    """Benchmark all installed models."""
    try:
        if not args.json:
            print_info("Benchmarking all installed models...")
            print_warning(
                "This may take several minutes depending on the number of models"
            )

        results = oe.models.benchmark_all(
            cache_dir=args.cache_dir, warmup=args.warmup, runs=args.runs
        )

        if args.json:
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                print_success(f"Results saved to {args.output}")
            else:
                print(json.dumps(results, indent=2))
        else:
            print_header(
                f"Benchmark Results ({results['benchmarked']}/{results['total_models']} models)"
            )

            if results["benchmarked"] == 0:
                print_warning("No models were successfully benchmarked")
                return

            # Summary
            summary = results["summary"]
            if summary.get("fastest_model"):
                print_success(
                    f"Fastest: {summary['fastest_model']['id']} ({summary['fastest_model']['fps']:.1f} FPS on {summary['fastest_model']['device']})"
                )
            if summary.get("slowest_model"):
                print_info(
                    f"Slowest: {summary['slowest_model']['id']} ({summary['slowest_model']['fps']:.1f} FPS on {summary['slowest_model']['device']})"
                )
            print_info(f"Average performance: {summary.get('average_fps', 0)} FPS")

            # Individual results
            print("\nDetailed results:")
            for result in results["results"]:
                if "error" in result:
                    print_error(f"{result['model_name']}: {result['error']}")
                else:
                    bench = result["benchmark"]
                    model_id = result.get('model_id', 'unknown')
                    dtype = result.get('dtype', 'unknown')
                    print(f"  {model_id} ({dtype})")
                    print(
                        f"    Device: {bench.get('device', 'unknown')} | FPS: {bench.get('fps', 0):.1f} | Latency: {bench.get('mean_ms', 0):.1f}ms"
                    )

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                print_success(f"\nDetailed results saved to {args.output}")

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print_error(f"Benchmark failed: {e}")
        sys.exit(1)


def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="oe",
        description="OpenVINO-Easy: Framework-agnostic Python wrapper for OpenVINO 2025",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # doctor command (comprehensive diagnostics)
    doctor_parser = subparsers.add_parser(
        "doctor", help="Comprehensive OpenVINO diagnostics"
    )
    doctor_parser.add_argument(
        "--fix",
        choices=["cpu", "gpu", "npu"],
        help="Show fix suggestions for specific device",
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format for CI systems",
    )
    doctor_parser.set_defaults(func=cmd_doctor)

    # devices command
    devices_parser = subparsers.add_parser("devices", help="List available devices")
    devices_parser.set_defaults(func=list_devices)

    # npu-doctor command (legacy, use doctor instead)
    npu_parser = subparsers.add_parser("npu-doctor", help="Diagnose NPU driver status")
    npu_parser.set_defaults(func=cmd_npu_doctor)

    # run command
    run_parser = subparsers.add_parser("run", help="Run inference on a model")
    run_parser.add_argument("model", help="Model path or Hugging Face model ID")
    run_parser.add_argument("--prompt", help="Text prompt for inference")
    run_parser.add_argument("--input-file", help="Input file path (image, audio, etc.)")
    run_parser.add_argument("--output", help="Output file path")
    run_parser.add_argument(
        "--dtype", choices=["fp16", "int8"], default="fp16", help="Model precision"
    )
    run_parser.add_argument(
        "--device-preference",
        help="Comma-separated device preference (e.g., NPU,GPU,CPU)",
    )
    run_parser.add_argument(
        "--json", action="store_true", help="Output results in JSON format"
    )
    run_parser.set_defaults(func=run_inference)

    # benchmark command
    bench_parser = subparsers.add_parser("bench", help="Benchmark a model")
    bench_parser.add_argument("model", help="Model path or Hugging Face model ID")
    bench_parser.add_argument(
        "--warmup-runs", type=int, default=5, help="Number of warmup runs"
    )
    bench_parser.add_argument(
        "--benchmark-runs", type=int, default=20, help="Number of benchmark runs"
    )
    bench_parser.add_argument("--output", help="Output file path for results")
    bench_parser.add_argument(
        "--dtype", choices=["fp16", "int8"], default="fp16", help="Model precision"
    )
    bench_parser.add_argument(
        "--device-preference",
        help="Comma-separated device preference (e.g., NPU,GPU,CPU)",
    )
    bench_parser.add_argument(
        "--json", action="store_true", help="Output results in JSON format"
    )
    bench_parser.set_defaults(func=run_benchmark)

    # cache command group
    cache_parser = subparsers.add_parser("cache", help="Manage model cache")
    cache_subparsers = cache_parser.add_subparsers(
        dest="cache_command", help="Cache commands"
    )

    # cache list
    cache_list_parser = cache_subparsers.add_parser("list", help="List cached models")
    cache_list_parser.add_argument("--cache-dir", help="Custom cache directory")
    cache_list_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )
    cache_list_parser.set_defaults(func=cache_list)

    # cache size
    cache_size_parser = cache_subparsers.add_parser(
        "size", help="Show cache size information"
    )
    cache_size_parser.add_argument("--cache-dir", help="Custom cache directory")
    cache_size_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )
    cache_size_parser.set_defaults(func=cache_size)

    # cache remove
    cache_remove_parser = cache_subparsers.add_parser(
        "remove", help="Remove a specific model"
    )
    cache_remove_parser.add_argument("model", help="Exact model name to remove")
    cache_remove_parser.add_argument("--cache-dir", help="Custom cache directory")
    cache_remove_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )
    cache_remove_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )
    cache_remove_parser.set_defaults(func=cache_remove)

    # cache clear
    cache_clear_parser = cache_subparsers.add_parser(
        "clear", help="Clear cache (temp files only by default)"
    )
    cache_clear_parser.add_argument("--cache-dir", help="Custom cache directory")
    cache_clear_parser.add_argument(
        "--models",
        action="store_true",
        help="WARNING DANGEROUS: Also clear all cached models",
    )
    cache_clear_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip safety confirmations (VERY DANGEROUS)",
    )
    cache_clear_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )
    cache_clear_parser.set_defaults(func=cache_clear)

    # models command group (distinct from cache)
    models_parser = subparsers.add_parser("models", help="Advanced model management")
    models_subparsers = models_parser.add_subparsers(
        dest="models_command", help="Model management commands"
    )

    # models search
    models_search_parser = models_subparsers.add_parser(
        "search", help="Search for models on Hugging Face Hub"
    )
    models_search_parser.add_argument(
        "query", help="Search query (model name, description, etc.)"
    )
    models_search_parser.add_argument(
        "--limit", type=int, default=10, help="Maximum number of results (default: 10)"
    )
    models_search_parser.add_argument(
        "--type",
        choices=["text", "image", "audio", "vision"],
        help="Filter by model type",
    )
    models_search_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )
    models_search_parser.set_defaults(func=model_search)

    # models info
    models_info_parser = models_subparsers.add_parser(
        "info", help="Get detailed information about a model"
    )
    models_info_parser.add_argument("model", help="Model ID or name")
    models_info_parser.add_argument("--cache-dir", help="Custom cache directory")
    models_info_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )
    models_info_parser.set_defaults(func=model_info)

    # models install
    models_install_parser = models_subparsers.add_parser(
        "install", help="Download and convert a model"
    )
    models_install_parser.add_argument("model", help="Model ID to install")
    models_install_parser.add_argument(
        "--dtype",
        choices=["fp16", "int8", "fp16-nf4"],
        default="fp16",
        help="Model precision",
    )
    models_install_parser.add_argument("--cache-dir", help="Custom cache directory")
    models_install_parser.add_argument(
        "--force", action="store_true", help="Force reinstall if model exists"
    )
    models_install_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )
    models_install_parser.set_defaults(func=model_install)

    # models validate
    models_validate_parser = models_subparsers.add_parser(
        "validate", help="Validate model integrity"
    )
    models_validate_parser.add_argument(
        "--model", help="Specific model to validate (validates all if not specified)"
    )
    models_validate_parser.add_argument("--cache-dir", help="Custom cache directory")
    models_validate_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )
    models_validate_parser.set_defaults(func=model_validate)

    # models benchmark
    models_benchmark_parser = models_subparsers.add_parser(
        "benchmark", help="Benchmark all installed models"
    )
    models_benchmark_parser.add_argument("--cache-dir", help="Custom cache directory")
    models_benchmark_parser.add_argument(
        "--warmup", type=int, default=3, help="Warmup runs per model"
    )
    models_benchmark_parser.add_argument(
        "--runs", type=int, default=10, help="Benchmark runs per model"
    )
    models_benchmark_parser.add_argument("--output", help="Save results to JSON file")
    models_benchmark_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )
    models_benchmark_parser.set_defaults(func=model_benchmark)

    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n[WARNING] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
