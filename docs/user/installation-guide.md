# Installation Guide

This comprehensive guide will help you install and set up the Manuscript Journal Matcher on your system. Follow the instructions for your operating system and use case.

## 🎯 Overview

The Manuscript Journal Matcher is a Python-based application that runs locally on your computer. It requires Python 3.8+ and several dependencies for document processing and machine learning.

### ⏱️ Estimated Time
- **Basic Installation**: 10-15 minutes
- **First Run Setup**: 5-10 minutes (one-time model download)
- **Total**: 15-25 minutes

### 💾 System Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **Python** | 3.8+ | 3.10+ | Latest stable version recommended |
| **RAM** | 4GB | 8GB+ | More RAM = better performance |
| **Storage** | 2GB free | 5GB+ | For models, data, and cache |
| **Internet** | Required for setup | Broadband preferred | Model downloads ~200MB |

### 🖥️ Operating System Support

| OS | Status | Notes |
|----|--------|-------|
| **Windows** | ✅ Fully Supported | Windows 10/11 recommended |
| **macOS** | ✅ Fully Supported | macOS 10.14+ (Intel & Apple Silicon) |
| **Linux** | ✅ Fully Supported | Ubuntu 18.04+, Debian, CentOS, etc. |

## 🚀 Quick Installation (Recommended)

### Prerequisites Check

First, verify you have the required software:

```bash
# Check Python version (must be 3.8+)
python --version
# or
python3 --version

# Check pip is available
pip --version
# or
pip3 --version

# Check git is available
git --version
```

**Expected Output:**
```
Python 3.10.8
pip 22.3.1 from /usr/local/lib/python3.10/site-packages/pip (python 3.10)
git version 2.39.0
```

### Step 1: Clone the Repository

```bash
# Clone from GitHub
git clone https://github.com/Cold-Mountain/manuscript-journal-matcher.git

# Navigate to the project directory
cd manuscript-journal-matcher

# Verify you're in the right place
ls -la
```

**Expected Output:**
```
README.md
requirements.txt
src/
data/
tests/
docs/
...
```

### Step 2: Create Virtual Environment

**Why use a virtual environment?** It keeps the project dependencies isolated from your system Python, preventing conflicts.

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# Verify activation (prompt should show (venv))
which python
```

**Expected Output:**
```bash
# Prompt should show:
(venv) user@computer:~/manuscript-journal-matcher$ 

# Python path should be in venv:
/path/to/manuscript-journal-matcher/venv/bin/python
```

### Step 3: Install Dependencies

```bash
# Upgrade pip (recommended)
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Verify installation
pip list
```

**Expected Output (partial):**
```
Package                Version
---------------------- ---------
streamlit             1.28.0
sentence-transformers 2.2.2
faiss-cpu            1.7.4
torch                2.1.0
pandas               2.1.0
...
```

### Step 4: Initial Setup and Test

```bash
# Create necessary directories
python -c "from src.config import ensure_directories_exist; ensure_directories_exist()"

# Test the installation
python -c "
from src.embedder import get_embedding_info
info = get_embedding_info()
print('✅ Installation successful!')
print(f'Model: {info[\"model_name\"]}')
print(f'Dimension: {info[\"dimension\"]}')
"
```

**Expected Output:**
```
✅ Installation successful!
Model: all-MiniLM-L6-v2
Dimension: 384
```

### Step 5: Launch the Application

```bash
# Start the web application
streamlit run src/main.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501
```

🎉 **Success!** The application should open in your web browser automatically.

## 🐧 Platform-Specific Instructions

### Windows Installation

#### Option A: Using Command Prompt

```cmd
# Check Python installation
python --version

# If Python not found, download from https://python.org/downloads/
# Make sure to check "Add Python to PATH" during installation

# Clone repository
git clone https://github.com/Cold-Mountain/manuscript-journal-matcher.git
cd manuscript-journal-matcher

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Launch application
streamlit run src\main.py
```

#### Option B: Using PowerShell

```powershell
# Enable script execution (run as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Follow same steps as Command Prompt
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run src\main.py
```

#### Windows Troubleshooting

**Problem**: `python` command not found
```cmd
# Try python3 instead
python3 --version

# Or use Python Launcher
py --version
py -m pip install -r requirements.txt
```

**Problem**: Permission denied errors
```cmd
# Run Command Prompt as Administrator
# Or use --user flag
pip install --user -r requirements.txt
```

### macOS Installation

#### Using Terminal

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python (if needed)
brew install python@3.10

# Clone and setup
git clone https://github.com/Cold-Mountain/manuscript-journal-matcher.git
cd manuscript-journal-matcher

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Launch
streamlit run src/main.py
```

#### macOS Troubleshooting

**Problem**: SSL certificate errors
```bash
# Update certificates
/Applications/Python\ 3.10/Install\ Certificates.command
```

**Problem**: Apple Silicon compatibility
```bash
# For M1/M2 Macs, ensure you're using ARM64 Python
python -c "import platform; print(platform.machine())"
# Should output: arm64

# If issues with FAISS, try:
pip uninstall faiss-cpu
pip install faiss-cpu --no-cache-dir
```

### Linux Installation

#### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install Python and dependencies
sudo apt install python3 python3-pip python3-venv git

# Clone repository
git clone https://github.com/Cold-Mountain/manuscript-journal-matcher.git
cd manuscript-journal-matcher

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Launch application
streamlit run src/main.py
```

#### CentOS/RHEL/Fedora

```bash
# Install Python and git
# CentOS/RHEL:
sudo yum install python3 python3-pip git
# Fedora:
sudo dnf install python3 python3-pip git

# Follow same steps as Ubuntu
git clone https://github.com/Cold-Mountain/manuscript-journal-matcher.git
cd manuscript-journal-matcher
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run src/main.py
```

#### Linux Troubleshooting

**Problem**: Permission errors for pip
```bash
# Use --user flag
pip install --user -r requirements.txt

# Or fix pip permissions
sudo chown -R $USER:$USER ~/.local
```

**Problem**: Missing development headers
```bash
# Ubuntu/Debian:
sudo apt install python3-dev build-essential

# CentOS/RHEL:
sudo yum install python3-devel gcc gcc-c++
```

## 🔧 Advanced Installation Options

### Docker Installation (Experimental)

```bash
# Clone repository
git clone https://github.com/Cold-Mountain/manuscript-journal-matcher.git
cd manuscript-journal-matcher

# Build Docker image
docker build -t journal-matcher .

# Run container
docker run -p 8501:8501 journal-matcher
```

### Conda Installation

```bash
# Create conda environment
conda create -n journal-matcher python=3.10
conda activate journal-matcher

# Clone repository
git clone https://github.com/Cold-Mountain/manuscript-journal-matcher.git
cd manuscript-journal-matcher

# Install dependencies
pip install -r requirements.txt

# Launch
streamlit run src/main.py
```

### Development Installation

For developers who want to contribute:

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/manuscript-journal-matcher.git
cd manuscript-journal-matcher

# Add upstream remote
git remote add upstream https://github.com/Cold-Mountain/manuscript-journal-matcher.git

# Create development environment
python -m venv venv-dev
source venv-dev/bin/activate  # or venv-dev\Scripts\activate on Windows

# Install with development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists

# Install in editable mode
pip install -e .

# Run tests
pytest tests/
```

## 🗄️ Database Setup

The journal database is built automatically on first run, but you can pre-build it:

```bash
# Activate virtual environment first
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Build journal database (optional - happens automatically)
python scripts/build_database.py --limit 1000

# Check database status
python -c "
from src.journal_db_builder import load_journal_database
journals, embeddings = load_journal_database()
print(f'✅ Database loaded: {len(journals)} journals')
"
```

## 📝 Configuration

### Environment Variables

Create a `.env` file for custom configuration:

```bash
# Copy example environment file
cp .env.example .env

# Edit with your preferred settings
nano .env  # or your preferred editor
```

**Example `.env` file:**
```bash
# Embedding model configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=32

# File processing limits
MAX_FILE_SIZE_MB=50
MAX_TEXT_LENGTH=2048

# Search parameters
MAX_RESULTS=20
MIN_SIMILARITY=0.1

# Cache settings
CACHE_DURATION_HOURS=24

# API keys (optional)
OPENALEX_API_KEY=your_key_here
CROSSREF_MAILTO=your.email@domain.com

# Logging
LOG_LEVEL=INFO
```

### Custom Installation Paths

```bash
# Install to custom directory
git clone https://github.com/Cold-Mountain/manuscript-journal-matcher.git /opt/journal-matcher
cd /opt/journal-matcher

# Use custom data directory
export DATA_DIR=/path/to/your/data
python src/main.py
```

## ✅ Verification and Testing

### System Check Script

Create a verification script to test your installation:

```bash
# Create test script
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""Installation verification script."""

import sys
import os
from pathlib import Path

def test_python_version():
    """Test Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def test_imports():
    """Test critical imports."""
    imports = [
        ('streamlit', 'Streamlit web framework'),
        ('torch', 'PyTorch machine learning'),
        ('sentence_transformers', 'Sentence transformers'),
        ('faiss', 'FAISS similarity search'),
        ('pandas', 'Pandas data processing'),
        ('numpy', 'NumPy numerical computing')
    ]
    
    all_passed = True
    for module, description in imports:
        try:
            __import__(module)
            print(f"✅ {module} - {description}")
        except ImportError:
            print(f"❌ {module} - {description} (MISSING)")
            all_passed = False
    
    return all_passed

def test_project_structure():
    """Test project structure."""
    required_paths = [
        'src/main.py',
        'src/extractor.py', 
        'src/embedder.py',
        'src/match_journals.py',
        'src/config.py',
        'requirements.txt'
    ]
    
    all_exist = True
    for path in required_paths:
        if Path(path).exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path} (MISSING)")
            all_exist = False
    
    return all_exist

def test_model_loading():
    """Test embedding model loading."""
    try:
        from src.embedder import get_embedding_info
        info = get_embedding_info()
        
        if info.get('loaded'):
            print(f"✅ Embedding model: {info['model_name']} ({info['dimension']}D)")
            return True
        else:
            print(f"❌ Embedding model failed to load: {info.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Embedding model test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔍 Testing Manuscript Journal Matcher Installation")
    print("=" * 60)
    
    tests = [
        ("Python Version", test_python_version),
        ("Package Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Model Loading", test_model_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 Installation verification PASSED!")
        print("✅ Ready to use the Manuscript Journal Matcher")
        return True
    else:
        print("❌ Installation verification FAILED!")
        print("💡 Please check the failed tests above and reinstall if needed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

# Run verification
python test_installation.py
```

### Quick Functionality Test

```bash
# Test document extraction
python -c "
from src.extractor import extract_title_and_abstract
text = 'Machine Learning in Healthcare\n\nAbstract: This study explores...'
title, abstract = extract_title_and_abstract(text)
print(f'✅ Extraction test passed: {bool(title and abstract)}')
"

# Test embedding generation
python -c "
from src.embedder import embed_text
embedding = embed_text('test text for embedding')
print(f'✅ Embedding test passed: {embedding.shape == (384,)}')
"

# Test web application (will open browser)
echo "🌐 Testing web application - should open in browser..."
timeout 10s streamlit run src/main.py --server.headless true || echo "✅ Streamlit launches correctly"
```

## 🚨 Common Installation Issues

### Issue 1: Python Not Found

**Symptoms:**
```
'python' is not recognized as an internal or external command
```

**Solutions:**
```bash
# Try python3 instead
python3 --version

# On Windows, try Python Launcher
py --version

# Install Python from https://python.org/downloads/
# Ensure "Add Python to PATH" is checked
```

### Issue 2: Permission Denied

**Symptoms:**
```
Permission denied: '/usr/local/lib/python3.x/site-packages/'
```

**Solutions:**
```bash
# Use --user flag
pip install --user -r requirements.txt

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Issue 3: FAISS Installation Fails

**Symptoms:**
```
Failed building wheel for faiss-cpu
```

**Solutions:**
```bash
# Update pip first
pip install --upgrade pip

# Try installing separately
pip install --no-cache-dir faiss-cpu

# On Apple Silicon Macs:
conda install -c conda-forge faiss-cpu

# Alternative for problematic systems:
pip install faiss-cpu --no-binary faiss-cpu
```

### Issue 4: Model Download Fails

**Symptoms:**
```
HTTPSConnectionPool: Max retries exceeded
```

**Solutions:**
```bash
# Check internet connection
ping huggingface.co

# Try manual download
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('✅ Model downloaded successfully')
"

# Configure proxy if needed
export https_proxy=http://your-proxy:port
pip install -r requirements.txt
```

### Issue 5: Port Already in Use

**Symptoms:**
```
OSError: [Errno 48] Address already in use
```

**Solutions:**
```bash
# Use different port
streamlit run src/main.py --server.port 8502

# Find and kill process using port 8501
lsof -ti:8501 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :8501   # Windows
```

## 🔄 Updating the Installation

### Regular Updates

```bash
# Navigate to project directory
cd manuscript-journal-matcher

# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Pull latest changes
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt

# Clear cache if needed
rm -rf data/api_cache/*

# Restart application
streamlit run src/main.py
```

### Major Version Updates

```bash
# Backup your data (if you have custom files)
cp -r data/ data_backup/

# Pull updates
git pull origin main

# Check for breaking changes
cat CHANGELOG.md

# Recreate virtual environment for major updates
deactivate
rm -rf venv/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test installation
python test_installation.py
```

## 📞 Getting Help

### Installation Support Checklist

Before asking for help, please:

1. **Check your Python version**: `python --version` (must be 3.8+)
2. **Verify you're in the correct directory**: `ls -la` should show README.md
3. **Confirm virtual environment is active**: Prompt should show `(venv)`
4. **Run the verification script**: `python test_installation.py`
5. **Check the logs**: Look for error messages in the terminal output

### Where to Get Help

1. **Search existing issues**: [GitHub Issues](https://github.com/Cold-Mountain/manuscript-journal-matcher/issues)
2. **Check troubleshooting guide**: [troubleshooting.md](troubleshooting.md)
3. **Ask the community**: [GitHub Discussions](https://github.com/Cold-Mountain/manuscript-journal-matcher/discussions)
4. **Create a new issue**: Include your system info and error messages

### System Information for Support

When reporting issues, please include:

```bash
# Generate system information
python -c "
import sys, platform, os
print(f'OS: {platform.system()} {platform.release()}')
print(f'Architecture: {platform.machine()}')
print(f'Python: {sys.version}')
print(f'Current directory: {os.getcwd()}')
print(f'Virtual env: {os.environ.get(\"VIRTUAL_ENV\", \"Not activated\")}')
"
```

---

## 🎉 Next Steps

✅ **Installation Complete!** 

Now you're ready to:
1. **[Try the Quick Start Tutorial](../QUICKSTART.md)** - Get familiar with the interface
2. **[Follow the Usage Tutorial](usage-tutorial.md)** - Learn all the features
3. **[Read the FAQ](faq.md)** - Understand common concepts
4. **[Explore Advanced Features](advanced-features.md)** - Unlock more functionality

**Need help?** Check our [Troubleshooting Guide](troubleshooting.md) or [create an issue](https://github.com/Cold-Mountain/manuscript-journal-matcher/issues).

*Last updated: July 29, 2025 | Installation Guide v2.0*