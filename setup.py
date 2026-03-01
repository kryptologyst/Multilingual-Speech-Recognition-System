#!/usr/bin/env python3
"""Setup script for multilingual ASR system."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🎤 Multilingual Speech Recognition System Setup")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -e .", "Installing main dependencies"),
        ("pip install -e \".[dev]\"", "Installing development dependencies"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"❌ Setup failed at: {description}")
            sys.exit(1)
    
    # Create necessary directories
    directories = [
        "data/wav",
        "data/meta",
        "outputs",
        "checkpoints",
        "assets",
        "logs",
    ]
    
    print("\n📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    # Test basic functionality
    print("\n🧪 Testing basic functionality...")
    test_script = """
import sys
try:
    from src.models.whisper_model import WhisperASRModel
    from src.utils.device import get_device, set_seed
    from src.data.multilingual_dataset import MultilingualASRDataset
    from src.eval.evaluator import ASREvaluator
    from src.metrics.asr_metrics import ASRMetrics
    print("✅ All imports successful")
    
    # Test device detection
    device = get_device('auto')
    print(f"✅ Device detection: {device}")
    
    # Test seeding
    set_seed(42)
    print("✅ Seeding works")
    
    print("✅ Basic functionality test passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_script], check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Basic functionality test failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        sys.exit(1)
    
    # Final instructions
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the demo script:")
    print("   python scripts/demo.py")
    print("\n2. Launch the interactive demo:")
    print("   streamlit run demo/streamlit_demo.py")
    print("\n3. Run evaluation:")
    print("   python scripts/main.py")
    print("\n4. Run tests:")
    print("   pytest tests/")
    print("\n5. Check code quality:")
    print("   ruff check src/")
    print("   black --check src/")
    print("   mypy src/")
    
    print("\n📚 Documentation:")
    print("   See README.md for detailed usage instructions")
    print("   See configs/ for configuration options")
    
    print("\n⚠️  Privacy Notice:")
    print("   This is a research demonstration system.")
    print("   Please use responsibly and respect privacy guidelines.")


if __name__ == "__main__":
    main()
