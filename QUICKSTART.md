# 🚀 Quick Start Guide

**Get up and running with the Manuscript Journal Matcher in 5 minutes!**

## Prerequisites

Before you begin, ensure you have:
- **Python 3.8+** installed ([Download Python](https://python.org/downloads/))
- **4GB+ RAM** (recommended for optimal performance)
- **Internet connection** (for initial setup and journal database)

## 1-Minute Installation

### Option A: Clone from GitHub (Recommended)
```bash
# Clone the repository
git clone https://github.com/Cold-Mountain/manuscript-journal-matcher.git
cd manuscript-journal-matcher

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option B: Download ZIP
1. Download the project ZIP from [GitHub](https://github.com/Cold-Mountain/manuscript-journal-matcher)
2. Extract to your desired location
3. Open terminal/command prompt in that folder
4. Run the virtual environment and installation commands above

## 2-Minute First Run

### Launch the Application
```bash
# Make sure you're in the project directory and virtual environment is active
streamlit run src/main.py
```

The application will:
- 🚀 Start a local web server (usually at `http://localhost:8501`)
- 📊 Load the journal database (7,600+ medical journals)
- 🌐 Open your default browser automatically

### First Time Setup
On first launch, the system will:
1. **Load Models**: Download the AI embedding model (~80MB) - this happens once
2. **Initialize Database**: Load journal metadata and search index
3. **Ready to Use**: Complete setup in 1-2 minutes

## 3-Minute Usage

### Step 1: Upload Your Manuscript
- 📁 **Drag & Drop** or **Browse** to upload your manuscript
- 📄 **Supported Formats**: PDF (.pdf) or Word (.docx) files  
- 📏 **File Size**: Up to 50MB per file
- ✅ **Privacy**: Your files are processed locally and never stored

### Step 2: Configure Search (Optional)
- 🔢 **Number of Results**: How many journal suggestions you want (default: 10)
- 📊 **Quality Filter**: Minimum journal quality (Q1=highest, Q4=lowest)
- 💰 **Cost Filter**: Maximum Article Processing Charge (APC)
- 🆓 **Open Access**: Filter for open access journals only

### Step 3: Get Results
- ⚡ **Instant Results**: Processing typically takes 2-5 seconds
- 🏆 **Quality Ranking**: Results sorted by relevance + journal prestige  
- 📊 **Rich Information**: Journal rankings, costs, publisher details
- 📋 **Export Options**: Download results as CSV for further analysis

## Common Issues & Solutions

### Installation Problems
**Problem**: `pip install` fails with dependency errors
```bash
# Solution: Upgrade pip first
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Problem**: Python version too old
```bash
# Check your Python version
python --version
# Must be 3.8 or higher. If not, install newer Python.
```

### Runtime Issues
**Problem**: "Port already in use" error
```bash
# Solution: Use a different port
streamlit run src/main.py --server.port 8502
```

**Problem**: Model download fails
```bash
# Solution: Manual model download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**Problem**: File upload not working
- ✅ Check file format (PDF or DOCX only)
- ✅ Ensure file size < 50MB
- ✅ Verify file is not corrupted

## Next Steps

### 📖 Learn More
- **[Complete User Guide](docs/user/installation-guide.md)**: Detailed setup instructions
- **[Usage Tutorial](docs/user/usage-tutorial.md)**: Step-by-step workflows
- **[Troubleshooting](docs/user/troubleshooting.md)**: Common issues and solutions

### 🔧 Advanced Features
- **API Usage**: Use the system programmatically
- **Custom Filters**: Advanced search parameters
- **Batch Processing**: Process multiple manuscripts
- **Export Options**: Different output formats

### 🤝 Get Involved
- **[Report Issues](https://github.com/Cold-Mountain/manuscript-journal-matcher/issues)**: Found a bug? Let us know!
- **[Contribute](CONTRIBUTING.md)**: Help improve the system
- **[Discussions](https://github.com/Cold-Mountain/manuscript-journal-matcher/discussions)**: Ask questions, share ideas

---

## 🎯 Success Checklist

After following this guide, you should be able to:
- [ ] ✅ Launch the web application in your browser
- [ ] ✅ Upload a manuscript file successfully  
- [ ] ✅ See journal recommendations with quality metrics
- [ ] ✅ Apply filters to refine results
- [ ] ✅ Export results for further analysis

**Need Help?** Check our [Troubleshooting Guide](docs/user/troubleshooting.md) or [create an issue](https://github.com/Cold-Mountain/manuscript-journal-matcher/issues) for support.

---

<div align="center">

**🎉 Congratulations! You're ready to find the perfect journals for your research.**

[📖 User Guide](docs/user/) • [🔧 Advanced Features](docs/api/) • [🤝 Contributing](CONTRIBUTING.md)

</div>