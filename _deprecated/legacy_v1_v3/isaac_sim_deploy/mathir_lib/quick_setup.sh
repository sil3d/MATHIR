#!/bin/bash
# Quick setup script for MATHIR

echo "🚀 MATHIR Quick Setup"
echo "===================="

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.8+ required (found $python_version)"
    exit 1
fi
echo "✅ Python $python_version detected"

# Create virtual environment (optional)
read -p "Create virtual environment? (y/n): " create_venv
if [[ $create_venv == "y" ]]; then
    python3 -m venv mathir_env
    source mathir_env/bin/activate
    echo "✅ Virtual environment activated"
fi

# Run hardware detection
echo "🔍 Detecting hardware..."
python3 configure_mathir.py

# Install dependencies
read -p "Install dependencies based on configuration? (y/n): " install_deps
if [[ $install_deps == "y" ]]; then
    python3 install_dependencies.py
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review config/mathir_optimized.yaml"
echo "2. For training: python train_agent.py"
echo "3. For deployment: python deploy_agent.py"
echo ""
echo "Need help? Check config/hardware_info.json for detected hardware."