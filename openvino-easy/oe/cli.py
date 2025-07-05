"""Command-line interface for OpenVINO-Easy."""

import argparse
import json
import sys
import platform
import subprocess
from pathlib import Path
from typing import Optional

import oe
from ._core import check_npu_driver, get_available_devices

def cmd_doctor(args):
    """Comprehensive OpenVINO installation diagnostics."""
    if args.json:
        _doctor_json_output(args)
        return
        
    print("🩺 OpenVINO-Easy Doctor")
    print("=" * 50)
    
    # System info
    print(f"🖥️  System: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {platform.python_version()}")
    print()
    
    # OpenVINO installation check
    print("📦 OpenVINO Installation:")
    try:
        import openvino as ov
        print(f"  ✅ OpenVINO version: {ov.__version__}")
        
        # Check if it's dev or runtime
        try:
            from openvino.tools import mo  # Model Optimizer
            install_type = "openvino-dev (full)"
        except ImportError:
            install_type = "openvino (runtime only)"
        print(f"  📋 Install type: {install_type}")
        
    except ImportError:
        print("  ❌ OpenVINO not found")
        print("\n🔧 Recommended fixes:")
        if args.fix:
            _suggest_openvino_install(args.fix)
        else:
            _suggest_openvino_install("cpu")
        return
    
    # Device detection
    print("\n🔌 Device Detection:")
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
                print(f"    └─ ⚠️  NVIDIA GPU detected - install Intel GPU drivers for Intel GPU support")
                
        except:
            print(f"    └─ Name: Unable to query")
    
    # NPU-specific diagnostics
    if "NPU" in all_devices:
        print("\n🧠 NPU Diagnostics:")
        npu_status = check_npu_driver()
        print(f"  Driver status: {npu_status['driver_status']}")
        if npu_status["device_name"]:
            print(f"  Device: {npu_status['device_name']}")
        
        if not npu_status["npu_functional"] and npu_status["recommendations"]:
            print("  🔧 Recommendations:")
            for rec in npu_status["recommendations"]:
                print(f"    • {rec}")
    
    # Performance recommendations
    print("\n⚡ Performance Recommendations:")
    best_device = oe.detect_best_device()
    print(f"  🎯 Recommended device: {best_device}")
    
    if "NPU" in device_status and not device_status["NPU"]:
        print("  💡 NPU detected but not functional - install NPU drivers for best performance")
    elif "GPU" in device_status and not device_status["GPU"]:
        print("  💡 GPU detected but not functional - install Intel GPU drivers")
    
    # Fix suggestions
    if args.fix:
        print(f"\n🔧 Fix suggestions for {args.fix.upper()}:")
        _suggest_device_fix(args.fix, device_status, npu_status if "NPU" in all_devices else None)
    
    # Summary
    functional_count = sum(device_status.values())
    total_count = len(device_status)
    print(f"\n📊 Summary: {functional_count}/{total_count} devices functional")
    
    if functional_count == 0:
        print("⚠️  No functional devices detected - OpenVINO installation may be corrupted")
    elif functional_count < total_count:
        print("⚠️  Some devices need attention - run 'oe doctor --fix <device>' for help")
    else:
        print("🎉 All detected devices are functional!")

def _suggest_openvino_install(target_device):
    """Suggest OpenVINO installation commands."""
    suggestions = {
        "cpu": "pip install 'openvino-easy[cpu]'",
        "gpu": "pip install 'openvino-easy[gpu]'", 
        "npu": "pip install 'openvino-easy[npu]'",
        "full": "pip install 'openvino-easy[full]'"
    }
    
    cmd = suggestions.get(target_device.lower(), suggestions["cpu"])
    print(f"  💾 Install command: {cmd}")
    
    if target_device.lower() == "gpu":
        system = platform.system()
        if system == "Windows":
            print("  📋 Additional: Install Intel GPU drivers from intel.com")
        elif system == "Linux":
            print("  📋 Additional: sudo apt install intel-opencl-icd (Ubuntu/Debian)")
    
    elif target_device.lower() == "npu":
        print("  📋 Additional: Install Intel NPU drivers from intel.com")

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
        print("  1. Reinstall OpenVINO: pip uninstall openvino && pip install 'openvino-easy[cpu]'")
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
        "python_version": platform.python_version()
    }
    
    # Collect OpenVINO info
    openvino_info = {
        "installed": True,
        "version": ov.__version__,
        "install_type": "runtime"
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
        device_info = {
            "functional": is_functional,
            "name": "Unknown"
        }
        
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
            "all_functional": len(validated_devices) == len(all_devices)
        }
    }
    
    print(json.dumps(results, indent=2))

def list_devices(args):
    """List available devices with validation status."""
    print("🔍 Scanning OpenVINO devices...")
    print()
    
    # Get all devices (including potentially non-functional ones)
    import openvino as ov
    core = ov.Core()
    all_devices = core.available_devices
    validated_devices = get_available_devices()
    
    print("Device Status:")
    for device in all_devices:
        status = "✅ Functional" if device in validated_devices else "❌ Not functional"
        print(f"  {device}: {status}")
        
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
    print(f"✅ {len(validated_devices)} functional device(s) detected")
    
    # Show recommended device
    best_device = oe.detect_best_device()
    print(f"🎯 Recommended device: {best_device}")

def cmd_npu_doctor(args):
    """Diagnose NPU driver status."""
    print("🔧 NPU Driver Diagnostics")
    print("=" * 50)
    
    npu_status = check_npu_driver()
    
    print(f"NPU in available devices: {'✅ Yes' if npu_status['npu_in_available_devices'] else '❌ No'}")
    print(f"NPU functional: {'✅ Yes' if npu_status['npu_functional'] else '❌ No'}")
    print(f"Driver status: {npu_status['driver_status']}")
    
    if npu_status["device_name"]:
        print(f"Device name: {npu_status['device_name']}")
    
    if npu_status["recommendations"]:
        print()
        print("📋 Recommendations:")
        for i, rec in enumerate(npu_status["recommendations"], 1):
            print(f"  {i}. {rec}")
    
    print()
    if npu_status["npu_functional"]:
        print("🎉 NPU is ready for use!")
    else:
        print("⚠️  NPU requires attention before use.")

def run_inference(args):
    """Run inference on a model."""
    try:
        # Load model
        print(f"📥 Loading model: {args.model}")
        
        device_preference = None
        if args.device_preference:
            device_preference = args.device_preference.split(",")
        
        pipeline = oe.load(
            args.model,
            device_preference=device_preference,
            dtype=args.dtype
        )
        
        print(f"✅ Model loaded on {pipeline.device}")
        
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
        print(f"🚀 Running inference...")
        result = pipeline.infer(input_data)
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            if output_path.suffix == '.json':
                with open(output_path, 'w') as f:
                    json.dump({"result": result}, f, indent=2)
                print(f"💾 Results saved to {output_path}")
            else:
                with open(output_path, 'w') as f:
                    f.write(str(result))
                print(f"💾 Results saved to {output_path}")
        else:
            print("📊 Results:")
            if isinstance(result, (list, dict)):
                print(json.dumps(result, indent=2))
            else:
                print(result)
                
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def run_benchmark(args):
    """Benchmark a model."""
    try:
        # Load model
        print(f"📥 Loading model: {args.model}")
        
        device_preference = None
        if args.device_preference:
            device_preference = args.device_preference.split(",")
        
        pipeline = oe.load(
            args.model,
            device_preference=device_preference,
            dtype=args.dtype
        )
        
        print(f"✅ Model loaded on {pipeline.device}")
        
        # Run benchmark
        print(f"⏱️  Benchmarking...")
        stats = pipeline.benchmark(
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs
        )
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"💾 Benchmark results saved to {output_path}")
        else:
            print("📊 Benchmark Results:")
            print(json.dumps(stats, indent=2))
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="oe",
        description="OpenVINO-Easy: Framework-agnostic Python wrapper for OpenVINO 2024"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # doctor command (comprehensive diagnostics)
    doctor_parser = subparsers.add_parser("doctor", help="Comprehensive OpenVINO diagnostics")
    doctor_parser.add_argument("--fix", choices=["cpu", "gpu", "npu"], help="Show fix suggestions for specific device")
    doctor_parser.add_argument("--json", action="store_true", help="Output results in JSON format for CI systems")
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
    run_parser.add_argument("--input-file", help="Input file path")
    run_parser.add_argument("--output", help="Output file path")
    run_parser.add_argument("--dtype", choices=["fp16", "int8"], default="fp16", help="Model precision")
    run_parser.add_argument("--device-preference", help="Comma-separated device preference (e.g., NPU,GPU,CPU)")
    run_parser.set_defaults(func=run_inference)
    
    # benchmark command
    bench_parser = subparsers.add_parser("bench", help="Benchmark a model")
    bench_parser.add_argument("model", help="Model path or Hugging Face model ID")
    bench_parser.add_argument("--warmup-runs", type=int, default=5, help="Number of warmup runs")
    bench_parser.add_argument("--benchmark-runs", type=int, default=20, help="Number of benchmark runs")
    bench_parser.add_argument("--output", help="Output file path for results")
    bench_parser.add_argument("--dtype", choices=["fp16", "int8"], default="fp16", help="Model precision")
    bench_parser.add_argument("--device-preference", help="Comma-separated device preference (e.g., NPU,GPU,CPU)")
    bench_parser.set_defaults(func=run_benchmark)
    
    return parser

def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 