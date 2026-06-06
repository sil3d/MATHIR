#!/usr/bin/env python3
"""
Setup script: copies llama.cpp binaries from any source directory the user provides.
NO HARDCODED PATHS. User passes source as CLI argument or env variable.

Usage:
  python setup_binaries.py /path/to/llama.cpp/bin
  python setup_binaries.py --url https://github.com/ggerganov/llama.cpp/releases/latest
  LLAMA_CPP_BIN=/path/to/bin python setup_binaries.py
"""
import os
import sys
import shutil
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime
import argparse
import json
import re


def setup_log():
    """Create setup log file."""
    log_path = Path(__file__).parent / "setup_log.json"
    return log_path


def log_action(action, **kwargs):
    """Log setup action."""
    log_path = setup_log()
    log = []
    if log_path.exists():
        try:
            with open(log_path) as f:
                log = json.load(f)
        except Exception:
            log = []
    log.append({"action": action, "timestamp": datetime.now().isoformat(), **kwargs})
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


def copy_from_local(source_dir: Path, bin_dir: Path, patterns=None):
    """Copy llama.cpp binaries from a local directory."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    bin_dir.mkdir(parents=True, exist_ok=True)

    if patterns is None:
        # Default patterns - match llama.cpp and ggml binaries
        patterns = [
            r"llama-server(\.exe)?$",
            r"llama-cli(\.exe)?$",
            r"llama-quantize(\.exe)?$",
            r"llama\.dll$",
            r"llama\.so.*",
            r"libllama.*\.so.*",
            r"libllama-common.*\.so.*",
            r"libmtmd.*\.so.*",
            r"ggml.*\.dll$",
            r"ggml.*\.so.*",
            r"libggml.*\.so.*",
            r"mtmd.*\.dll$",
            r"cublas.*\.dll$",
            r"cublasLt.*\.dll$",
            r"cudart.*\.dll$",
            r"curand.*\.dll$",
            r"cusolver.*\.dll$",
            r"cusparse.*\.dll$",
            r"libomp.*",
            r"convert_lfm2_to_gguf\.py$",
            r"convert_to_gguf\.py$",
        ]

    copied = 0
    skipped = 0
    failed = []
    matched_files = []

    for f in source_dir.iterdir():
        if not f.is_file():
            continue
        matched = False
        for pattern in patterns:
            if re.search(pattern, f.name, re.IGNORECASE):
                matched = True
                break
        if not matched:
            continue

        matched_files.append(f.name)
        dest = bin_dir / f.name
        try:
            if dest.exists() and dest.stat().st_size == f.stat().st_size:
                skipped += 1
                print(f"  [SKIP] {f.name} (already exists)")
            else:
                shutil.copy2(f, dest)
                copied += 1
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"  [OK]   {f.name} ({size_mb:.1f} MB)")
        except Exception as e:
            failed.append((f.name, str(e)))
            print(f"  [FAIL] {f.name}: {e}")

    log_action("copy_from_local",
               source=str(source_dir),
               matched=len(matched_files),
               copied=copied,
               skipped=skipped,
               failed=len(failed))

    return {
        "source": str(source_dir),
        "matched": len(matched_files),
        "copied": copied,
        "skipped": skipped,
        "failed": failed,
    }


def download_from_github(bin_dir: Path, release_tag: str = "latest"):
    """Download llama.cpp release from GitHub."""
    import json as _json

    # Get release info
    if release_tag == "latest":
        url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
    else:
        url = f"https://api.github.com/repos/ggerganov/llama.cpp/releases/tags/{release_tag}"

    print(f"  Fetching release info from {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MATHIR-setup"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            release = _json.loads(resp.read())
    except Exception as e:
        print(f"  ERROR fetching release: {e}")
        return None

    # Find Windows binary asset
    asset = None
    for a in release.get("assets", []):
        name = a.get("name", "")
        if "win" in name and ("cuda" in name or "cpu" in name) and name.endswith(".zip"):
            asset = a
            break

    if not asset:
        print(f"  No Windows binary found in release {release.get('tag_name')}")
        print(f"  Available assets: {[a.get('name') for a in release.get('assets', [])]}")
        return None

    print(f"  Found: {asset['name']} ({asset['size']/1024/1024:.0f} MB)")

    # Download
    zip_path = bin_dir.parent / asset["name"]
    print(f"  Downloading to {zip_path}")
    try:
        req = urllib.request.Request(asset["browser_download_url"], headers={"User-Agent": "MATHIR-setup"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(resp, f)
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        return None

    # Extract
    print(f"  Extracting...")
    bin_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        # Find files matching our patterns and extract them to bin_dir
        for name in z.namelist():
            base = Path(name).name
            # Skip directories
            if not base:
                continue
            # Check if it matches a binary
            if re.search(r"^(llama|ggml|libllama|libggml|libmtmd|mtmd|cublas|cudart|curand|cusolver|cusparse|libomp).*", base, re.IGNORECASE):
                try:
                    with z.open(name) as src, open(bin_dir / base, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    print(f"    Extracted: {base}")
                except Exception as e:
                    print(f"    Failed: {base}: {e}")

    # Clean up zip
    zip_path.unlink(missing_ok=True)
    log_action("download_from_github", tag=release.get("tag_name"))
    return {"source": f"github:{release.get('tag_name')}"}


def verify_binaries(bin_dir: Path) -> dict:
    """Verify llama-server binary works."""
    result = {"status": "unknown", "details": {}}

    # Check llama-server exists
    server_exe = bin_dir / "llama-server.exe"
    if not server_exe.exists():
        server_exe = bin_dir / "llama-server"
    if not server_exe.exists():
        result["status"] = "missing"
        result["details"]["llama-server"] = "not found"
        return result

    result["details"]["llama-server"] = str(server_exe)

    # Try --version
    try:
        env = os.environ.copy()
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
        proc = subprocess.run([str(server_exe), "--version"], capture_output=True,
                               timeout=10, env=env, text=True)
        if proc.returncode == 0:
            result["status"] = "ok"
            result["details"]["version"] = proc.stdout.strip()[:200]
        else:
            result["status"] = "error"
            result["details"]["stderr"] = proc.stderr[:200]
    except FileNotFoundError as e:
        result["status"] = "dll_missing"
        result["details"]["error"] = str(e)
    except Exception as e:
        result["status"] = "error"
        result["details"]["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Setup llama.cpp binaries for MATHIR vision testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Copy from a local directory
  python setup_binaries.py /path/to/llama.cpp/build/bin

  # Download from GitHub
  python setup_binaries.py --github
  python setup_binaries.py --github b1234

  # Use environment variable
  LLAMA_CPP_BIN=/path/to/bin python setup_binaries.py

  # Verify existing installation
  python setup_binaries.py --verify
        """,
    )
    parser.add_argument("source", nargs="?", help="Path to local llama.cpp build directory")
    parser.add_argument("--github", nargs="?", const="latest", metavar="TAG",
                        help="Download from GitHub release (default: latest)")
    parser.add_argument("--verify", action="store_true", help="Verify existing installation")
    parser.add_argument("--bin-dir", help="Override bin directory (default: ./bin)")

    args = parser.parse_args()

    # Resolve bin dir
    here = Path(__file__).resolve().parent
    if args.bin_dir:
        bin_dir = Path(args.bin_dir).resolve()
    else:
        # Read from config.json
        config_path = here / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            bin_dir = (here / config["bin_dir"]).resolve()
        else:
            bin_dir = here / "bin"

    print("=" * 70)
    print("LLAMA.CPP BINARIES SETUP")
    print("=" * 70)
    print(f"Target: {bin_dir}")

    # Verify only
    if args.verify:
        print("\nVerifying installation...")
        result = verify_binaries(bin_dir)
        print(f"  Status: {result['status']}")
        for k, v in result["details"].items():
            print(f"  {k}: {v}")
        return 0 if result["status"] == "ok" else 1

    # Determine source
    source = None
    source_type = None

    if args.github:
        source_type = "github"
        source = args.github
    elif args.source:
        source_type = "local"
        source = args.source
    elif os.environ.get("LLAMA_CPP_BIN"):
        source_type = "local"
        source = os.environ["LLAMA_CPP_BIN"]
    else:
        parser.print_help()
        print("\nNo source specified. Use:")
        print("  python setup_binaries.py <path>")
        print("  python setup_binaries.py --github")
        return 1

    # Execute
    if source_type == "local":
        print(f"\nCopying from local: {source}")
        result = copy_from_local(Path(source), bin_dir)
        print(f"\nResult: copied={result['copied']}, skipped={result['skipped']}, failed={len(result['failed'])}")
    elif source_type == "github":
        print(f"\nDownloading from GitHub: {source}")
        result = download_from_github(bin_dir, source)
        if not result:
            print("Download failed.")
            return 1

    # Verify
    print(f"\nVerifying installation in {bin_dir}...")
    verify = verify_binaries(bin_dir)
    print(f"  Status: {verify['status']}")
    if verify["status"] == "ok":
        print(f"  Version: {verify['details'].get('version', '?')}")
        print("\nSetup complete!")
    elif verify["status"] == "dll_missing":
        print("  WARNING: DLLs may be missing. Try running with PATH set:")
        print(f"    export PATH={bin_dir}:$PATH")
    else:
        print(f"  Issues: {verify['details']}")

    return 0 if verify["status"] == "ok" else 1


if __name__ == "__main__":
    import subprocess
    sys.exit(main())