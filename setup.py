#!/usr/bin/env python3
"""
Setup script for Cobb Angle Analysis System
This script helps users install dependencies and verify their installation.
"""

import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Error: Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_cuda():
    """Check CUDA availability"""
    print("\n🔍 Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def verify_installation():
    """Verify that all components are working"""
    print("\n🔍 Verifying installation...")
    
    # Check imports
    try:
        import torch
        import numpy as np
        import cv2
        import scipy
        print("✅ All required packages imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Check model file
    model_path = "pretrained_model/model_last.pth"
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
    else:
        print(f"⚠️  Model file not found: {model_path}")
        print("   Please download the pre-trained model and place it in the pretrained_model/ directory")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    directories = ["pretrained_model", "cobb_results", "output"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")

def run_test():
    """Run a simple test to verify functionality"""
    print("\n🧪 Running functionality test...")
    try:
        # Test basic imports and model loading
        import torch
        from models import spinal_net
        
        # Test model initialization
        heads = {'hm': 1, 'reg': 2*1, 'wh': 2*4}
        model = spinal_net.SpineNet(heads=heads, pretrained=False, down_ratio=4, final_kernel=1, head_conv=256)
        print("✅ Model initialization successful")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Cobb Angle Analysis System - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed during package installation")
        print("Please try installing packages manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Check CUDA
    check_cuda()
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Setup verification failed")
        sys.exit(1)
    
    # Run test
    if not run_test():
        print("\n❌ Functionality test failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Download the pre-trained model file (model_last.pth)")
    print("2. Place it in the pretrained_model/ directory")
    print("3. Run: python cobb_angle_inference.py --image_path your_xray.jpg")
    print("\n📖 For more information, see USER_MANUAL.md")

if __name__ == "__main__":
    main() 