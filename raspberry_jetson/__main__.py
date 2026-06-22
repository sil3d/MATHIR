"""Allow ``python -m raspberry_jetson`` to print version info on Pi/Jetson."""
import sys

# Importing the package triggers the Jetson/Pi env defaults in __init__.py
import raspberry_jetson  # noqa: F401

print(f"mathir-raspberry-jetson v{raspberry_jetson.__version__}")
print(f"  mathir-mcp version: {raspberry_jetson.__mcp_version__}")
print(f"  Embedding model:    {__import__('os').environ.get('MATHIR_EMBEDDING_MODEL')}")
print(f"  Embedding dim:      {__import__('os').environ.get('MATHIR_EMBEDDING_DIM')}")
print(f"  Device:             {__import__('os').environ.get('MATHIR_DEVICE')}")
print("")
print("Start the daemon with:  python -m raspberry_jetson --start")
print("                       or: ./start.sh")
sys.exit(0)
