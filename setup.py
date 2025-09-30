#!/usr/bin/env python3
"""
Setup script for the improved data preparation pipeline
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True

def install_requirements():
    """Install Python requirements"""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    return run_command("pip install -r requirements.txt", "Installing Python packages")

def download_spacy_model():
    """Download spaCy English model"""
    return run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model")

def download_nltk_data():
    """Download NLTK data"""
    nltk_script = """
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    exit(1)
"""
    
    with open("temp_nltk_download.py", "w") as f:
        f.write(nltk_script)
    
    success = run_command("python temp_nltk_download.py", "Downloading NLTK data")
    
    # Clean up
    if Path("temp_nltk_download.py").exists():
        Path("temp_nltk_download.py").unlink()
    
    return success

def create_directories():
    """Create necessary directories"""
    directories = [
        "artifacts",
        "artifacts/splits",
        "artifacts/labels",
        "artifacts/features",
        "artifacts/features/dense",
        "artifacts/features/tfidf",
        "artifacts/features/sequences",
        "artifacts/features/transformers",
        "artifacts/augmented",
        "artifacts/reports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created directory structure")
    return True

def check_dataset_files():
    """Check if dataset files are present"""
    expected_files = [
        "daily_dialog.csv",
        "ISEAR.csv", 
        "goemotions_1.csv",
        "goemotions_2.csv",
        "goemotions_3.csv"
    ]
    
    found_files = []
    missing_files = []
    
    for file in expected_files:
        if Path(file).exists():
            found_files.append(file)
        else:
            missing_files.append(file)
    
    print(f"📁 Dataset files found: {len(found_files)}/{len(expected_files)}")
    
    if found_files:
        print("✅ Found files:")
        for file in found_files:
            print(f"   - {file}")
    
    if missing_files:
        print("⚠️  Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n   You can still run the pipeline with available files,")
        print("   or the example script will create sample data.")
    
    return len(found_files) > 0

def main():
    """Main setup function"""
    print("🚀 Setting up Improved Data Preparation Pipeline")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed at package installation")
        return False
    
    # Download spaCy model
    if not download_spacy_model():
        print("\n⚠️  spaCy model download failed, but you can continue")
        print("   Run manually: python -m spacy download en_core_web_sm")
    
    # Download NLTK data
    if not download_nltk_data():
        print("\n⚠️  NLTK data download failed, but you can continue")
        print("   The script will download data automatically on first run")
    
    # Create directories
    create_directories()
    
    # Check dataset files
    check_dataset_files()
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Place your dataset files in the current directory")
    print("2. Run: python run_data_preparation.py")
    print("3. Or run: python example_usage.py for a demonstration")
    
    print("\n📚 Available scripts:")
    print("   - run_data_preparation.py: Run the pipeline")
    print("   - example_usage.py: See a demonstration")
    print("   - improved_data_preparation.py: Main pipeline code")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)