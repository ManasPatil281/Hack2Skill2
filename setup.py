#!/usr/bin/env python3
"""
CareerCompass Setup Script
Helps users set up the application quickly
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
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

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    directories = [
        "data",
        "data/session_logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created/verified directory: {directory}")

def check_env_file():
    """Check if .env file exists"""
    print("\n🔑 Checking environment configuration...")
    if not os.path.exists(".env"):
        print("⚠️  No .env file found")
        print("📝 Creating .env file from template...")
        
        if os.path.exists(".env.example"):
            with open(".env.example", "r") as example:
                with open(".env", "w") as env_file:
                    env_file.write(example.read())
            print("✅ Created .env file from .env.example")
            print("🔧 Please edit .env file and add your OpenAI API key")
        else:
            print("❌ .env.example not found")
            return False
    else:
        print("✅ .env file exists")
    
    return True

def check_data_files():
    """Check if required data files exist"""
    print("\n📊 Checking data files...")
    required_files = [
        "data/mock_career_data.json",
        "data/mentors.json"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_exist = False
    
    return all_exist

def main():
    """Main setup function"""
    print("🧭 CareerCompass Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Setup directories
    setup_directories()
    
    # Check environment file
    if not check_env_file():
        return False
    
    # Check data files
    if not check_data_files():
        print("\n❌ Some data files are missing. Please ensure all files are present.")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Run: streamlit run app.py")
    print("3. Click '🚀 Load Demo Session' to try the demo")
    print("\n🌐 The app will open at http://localhost:8501")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)