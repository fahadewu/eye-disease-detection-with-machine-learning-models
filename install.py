"""
install.py — Smart TensorFlow installer for Eye Disease Detection
=================================================================
Detects Python version, platform, and architecture, then installs
the correct TensorFlow build automatically.

Usage:
    python install.py
    python install.py --no-tf      (skip TF, use API fallback only)
    python install.py --check      (only print what would be installed)
"""

import sys
import os
import platform
import subprocess
import argparse

# ── Helpers ────────────────────────────────────────────────────────────────────

def run(cmd, check=True):
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)

def pip(*args):
    run([sys.executable, "-m", "pip", *args])

def pip_silent(*args):
    subprocess.run([sys.executable, "-m", "pip", *args],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def can_import(pkg):
    result = subprocess.run(
        [sys.executable, "-c", f"import {pkg}"],
        capture_output=True
    )
    return result.returncode == 0

def py_ver():
    return sys.version_info[:2]   # (major, minor)

def is_apple_silicon():
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def is_macos():
    return platform.system() == "Darwin"

def is_linux():
    return platform.system() == "Linux"

def is_windows():
    return platform.system() == "Windows"

# ── TF candidate selection ─────────────────────────────────────────────────────

def choose_tf_package():
    """
    Return (package_list, note) for the best TensorFlow build
    available for this environment, or (None, reason) if unsupported.
    """
    major, minor = py_ver()

    # Python 3.13+ — TF has no wheels yet (as of mid-2025)
    if (major, minor) >= (3, 13):
        return None, (
            f"Python {major}.{minor} is too new — TensorFlow has no wheels yet. "
            "Use Python 3.9–3.12 for full model support. "
            "The app will run in API-fallback-only mode."
        )

    # Python 3.8 or older — TF dropped support
    if (major, minor) < (3, 9):
        return None, (
            f"Python {major}.{minor} is too old — TensorFlow ≥2.13 requires Python 3.9+. "
            "The app will run in API-fallback-only mode."
        )

    # Apple Silicon (M1/M2/M3/M4)
    if is_apple_silicon():
        return (
            ["tensorflow-macos", "tensorflow-metal"],
            "Apple Silicon detected → installing tensorflow-macos + tensorflow-metal"
        )

    # Intel Mac
    if is_macos():
        return (
            ["tensorflow>=2.13.0"],
            "Intel macOS detected → installing tensorflow"
        )

    # Linux / Windows x86-64 — prefer CPU build (lighter, no CUDA needed)
    return (
        ["tensorflow-cpu>=2.13.0"],
        "Linux/Windows detected → installing tensorflow-cpu (no GPU driver needed)"
    )


# ── Core install logic ─────────────────────────────────────────────────────────

def install_base_deps():
    print("\n[1/2] Installing base dependencies (Flask, OpenCV, Pillow, …)")
    pip("install", "-r", "requirements.txt", "--quiet")
    print("      Base deps OK.\n")


def install_tensorflow(dry_run=False):
    packages, note = choose_tf_package()

    print("[2/2] TensorFlow selection:")
    print(f"      {note}\n")

    if packages is None:
        print("  ⚠  Skipping TensorFlow install.")
        print("     The app will still run — low-confidence predictions")
        print("     will be handled by the Hugging Face / Gemini fallback APIs.")
        print("     Configure your API keys in Admin → Settings.\n")
        return False

    if dry_run:
        print(f"  [dry-run] Would install: {', '.join(packages)}")
        return True

    # Try installing
    for attempt, pkg_list in enumerate([packages, ["tensorflow>=2.13.0"], ["tensorflow-cpu>=2.13.0"]], 1):
        print(f"  Attempt {attempt}: pip install {' '.join(pkg_list)}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", *pkg_list, "--quiet"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            # Verify import works
            if can_import("tensorflow"):
                # Get installed version
                ver_result = subprocess.run(
                    [sys.executable, "-c",
                     "import tensorflow as tf; print(tf.__version__)"],
                    capture_output=True, text=True
                )
                ver = ver_result.stdout.strip()
                print(f"\n  ✓ TensorFlow {ver} installed successfully.\n")
                return True
            else:
                print(f"  Install succeeded but import failed — trying next option.")
        else:
            err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
            print(f"  Failed: {err}")

        if attempt == 3:
            break

    print("\n  ⚠  Could not install TensorFlow automatically.")
    print("     The app will still run using API fallback mode.")
    print("     See README.md for manual installation instructions.\n")
    return False


def verify_install():
    print("\n── Verifying installation ─────────────────────────────────")
    checks = [
        ("flask",               "Flask"),
        ("werkzeug",            "Werkzeug"),
        ("PIL",                 "Pillow"),
        ("cv2",                 "OpenCV"),
        ("numpy",               "NumPy"),
        ("requests",            "requests"),
    ]
    all_ok = True
    for mod, name in checks:
        ok = can_import(mod)
        status = "✓" if ok else "✗"
        print(f"  {status}  {name}")
        if not ok:
            all_ok = False

    tf_ok = can_import("tensorflow")
    print(f"  {'✓' if tf_ok else '○'}  TensorFlow {'(loaded)' if tf_ok else '(not installed — fallback mode)'}")

    print()
    if all_ok:
        print("  All core dependencies satisfied.")
    else:
        print("  Some core deps failed — re-run: python install.py")

    print()
    print("  Start the app with:")
    print("    python app.py")
    print("  or")
    print("    bash run.sh")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Smart installer for Eye Disease Detection web app"
    )
    parser.add_argument("--no-tf",  action="store_true",
                        help="Skip TensorFlow; run in API-fallback-only mode")
    parser.add_argument("--check",  action="store_true",
                        help="Print what would be installed without installing")
    args = parser.parse_args()

    print("=" * 60)
    print("  Eye Disease Detection — Smart Installer")
    print("  East West University, Dept. of CSE")
    print("=" * 60)
    print(f"\n  Python : {sys.version.split()[0]}")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print()

    if args.check:
        packages, note = choose_tf_package()
        print(f"  TF selection: {note}")
        if packages:
            print(f"  Would install: pip install {' '.join(packages)}")
        return

    # Upgrade pip silently first
    pip_silent("install", "--upgrade", "pip")

    install_base_deps()

    if not args.no_tf:
        install_tensorflow(dry_run=False)
    else:
        print("[2/2] TensorFlow skipped (--no-tf flag). Running in fallback mode.\n")

    verify_install()


if __name__ == "__main__":
    main()
