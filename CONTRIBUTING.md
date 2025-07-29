# Contributing to Manuscript Journal Matcher

🎉 **Thank you for your interest in contributing!** We welcome contributions from researchers, developers, and users who want to help improve this journal matching system.

## 🌟 Ways to Contribute

### 🐛 For Users & Researchers
- **Bug Reports**: Found an issue? [Create a bug report](https://github.com/Cold-Mountain/manuscript-journal-matcher/issues/new?template=bug_report.md)
- **Feature Requests**: Have an idea? [Suggest a new feature](https://github.com/Cold-Mountain/manuscript-journal-matcher/issues/new?template=feature_request.md)  
- **Documentation**: Help improve guides, fix typos, add examples
- **Testing**: Try the system with your manuscripts and provide feedback
- **Journal Data**: Report missing or incorrect journal information

### 👩‍💻 For Developers
- **Code Contributions**: Fix bugs, add features, optimize performance
- **Algorithm Improvements**: Enhance matching accuracy and speed
- **Testing**: Write tests, improve test coverage, add edge cases
- **Infrastructure**: Deployment, monitoring, CI/CD improvements
- **Documentation**: API docs, technical guides, code examples

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.8+** installed
- **Git** for version control
- **GitHub account** for pull requests
- Basic knowledge of Python, ML, or web development (depending on contribution)

### Development Setup

1. **Fork & Clone**
   ```bash
   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/manuscript-journal-matcher.git
   cd manuscript-journal-matcher
   
   # Add upstream remote
   git remote add upstream https://github.com/Cold-Mountain/manuscript-journal-matcher.git
   ```

2. **Environment Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies (including dev dependencies)
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If exists
   ```

3. **Verify Setup**
   ```bash
   # Run tests to ensure everything works
   pytest tests/
   
   # Launch the application
   streamlit run src/main.py
   ```

4. **Create Development Branch**
   ```bash
   # Always create a new branch for your work
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

---

## 🛠️ Development Guidelines

### Code Style
- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Add type annotations for all functions
- **Docstrings**: Use Google-style docstrings for all public functions
- **Naming**: Use descriptive variable and function names
- **Comments**: Explain complex logic with inline comments

### Example Code Style
```python
def search_similar_journals(
    query_text: str, 
    top_k: int = 10,
    min_similarity: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Find journals most similar to the given query text.
    
    Args:
        query_text: The manuscript abstract or content to match
        top_k: Maximum number of results to return
        min_similarity: Minimum similarity threshold (0.0 to 1.0)
        
    Returns:
        List of journal dictionaries with similarity scores
        
    Raises:
        ValueError: If query_text is empty or invalid parameters
        
    Example:
        >>> matcher = JournalMatcher()
        >>> results = matcher.search_similar_journals("AI in medical diagnosis", top_k=5)
        >>> print(f"Found {len(results)} matching journals")
    """
    # Implementation here...
```

### Testing Requirements
- **Write Tests**: All new features must include tests
- **Test Coverage**: Aim for >80% coverage on new code
- **Test Types**: Include unit tests, integration tests, and edge cases
- **Test Data**: Use fixtures for consistent test data

### Example Test
```python
def test_journal_matching_basic():
    """Test basic journal matching functionality."""
    matcher = JournalMatcher()
    results = matcher.search_similar_journals(
        "machine learning in healthcare", 
        top_k=5
    )
    
    assert len(results) <= 5
    assert all('similarity_score' in result for result in results)
    assert all(0 <= result['similarity_score'] <= 1 for result in results)
```

---

## 📋 Contribution Process

### 1. Planning Your Contribution

**For Bug Fixes:**
- Check existing issues to avoid duplicates
- Reproduce the bug locally
- Understand the root cause before coding

**For New Features:**
- Check our [Implementation Roadmap](docs/IMPLEMENTATION_ROADMAP.md)
- Discuss large changes in an issue first
- Consider backward compatibility

### 2. Making Changes

**Code Changes:**
```bash
# Make your changes
git add .
git commit -m "feat: add study type classification system

- Implement text classification for research methodologies
- Add support for RCT, observational, and meta-analysis detection
- Include confidence scoring for classification results
- Add comprehensive tests for new functionality

Fixes #123"
```

**Commit Message Format:**
- `feat:` for new features
- `fix:` for bug fixes  
- `docs:` for documentation changes
- `test:` for test additions/changes
- `refactor:` for code refactoring
- `perf:` for performance improvements

### 3. Testing Your Changes

```bash
# Run the full test suite
pytest tests/

# Run specific test categories
pytest tests/test_matching.py -v
pytest tests/test_integration.py -v

# Check test coverage
pytest --cov=src tests/

# Run linting (if configured)
flake8 src/
black src/ --check
```

### 4. Submitting Your Contribution

```bash
# Push your branch
git push origin feature/your-feature-name

# Create pull request on GitHub
# - Use the PR template
# - Link related issues
# - Add screenshots for UI changes
# - Request review from maintainers
```

---

## 🎯 Specific Contribution Areas

### Algorithm Enhancement
**Skills Needed**: Python, ML, NLP, Information Retrieval
- Improve matching accuracy
- Add new similarity metrics
- Implement ensemble methods
- Optimize performance

**Example Tasks**:
- Study type classification (#45)
- Multi-modal content analysis (#67)
- Citation network analysis (#89)

### Documentation & UX
**Skills Needed**: Technical writing, UI/UX, Testing
- Write user guides and tutorials
- Improve API documentation
- Enhance web interface
- Create video tutorials

**Example Tasks**:
- API reference documentation (#23)
- Mobile-responsive interface (#34)
- Accessibility improvements (#56)

### Infrastructure & DevOps
**Skills Needed**: Docker, Cloud platforms, CI/CD, Monitoring
- Containerization and deployment
- Performance monitoring
- Automated testing pipelines
- Security improvements

**Example Tasks**:
- Docker configuration (#78)
- GitHub Actions CI/CD (#90)
- Performance monitoring (#101)

---

## 🧪 Using Claude Code for Development

We support development using **Claude Code**! Check our [Claude Code Development Guide](docs/CLAUDE_CODE_DEVELOPMENT_GUIDE.md) for:

- **Capability Assessment**: What works well in Claude Code vs. external tools
- **Development Strategy**: Optimal approach for different types of contributions
- **File Structure**: How to organize Claude Code development work
- **Integration Points**: How Claude Code work connects with external infrastructure

### Claude Code Friendly Tasks
✅ **Perfect for Claude Code**:
- Documentation writing and updates
- Algorithm implementation and enhancement
- Test writing and code quality improvements
- Configuration file creation

⚠️ **Partial Claude Code Support**:
- API integration (code writing yes, credential setup no)
- Performance optimization (implementation yes, benchmarking needs external tools)

❌ **Requires External Tools**:
- Production deployment and infrastructure
- Real-time monitoring and alerting
- Security audits and penetration testing

---

## 🏆 Recognition

### Contributors
All contributors are recognized in:
- **README.md**: Contributors section
- **GitHub**: Automatic contributor tracking
- **Releases**: Acknowledgment in release notes
- **Documentation**: Author attribution where appropriate

### Types of Recognition
- 🥇 **Major Contributors**: Significant features or sustained contributions
- 🥈 **Regular Contributors**: Multiple valuable contributions
- 🥉 **First-time Contributors**: Welcome and encouragement for newcomers
- 🏆 **Special Recognition**: Outstanding community support or unique contributions

---

## 🤝 Community Guidelines

### Code of Conduct
- **Be Respectful**: Treat all contributors with respect and kindness
- **Be Inclusive**: Welcome people of all backgrounds and skill levels
- **Be Constructive**: Provide helpful feedback and suggestions
- **Be Patient**: Remember that everyone is learning and contributing their time voluntarily

### Communication
- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and brainstorming
- **Pull Requests**: Keep PR discussions focused on the code changes
- **Reviews**: Provide constructive feedback with specific suggestions

### Quality Standards
- **Functionality**: Contributions must work as intended
- **Testing**: Include appropriate tests for your changes
- **Documentation**: Update documentation for user-facing changes
- **Performance**: Consider performance impact of changes
- **Security**: Follow security best practices

---

## ❓ Need Help?

### Getting Help
- **Documentation**: Check [our guides](docs/) first
- **Search Issues**: Look for similar questions or problems
- **Ask Questions**: Create a [GitHub Discussion](https://github.com/Cold-Mountain/manuscript-journal-matcher/discussions)
- **Join Community**: Participate in issue discussions

### Common Questions

**Q: I'm new to Python/ML/NLP. Can I still contribute?**
A: Absolutely! Start with documentation, testing, or small bug fixes. We're happy to help you learn.

**Q: How do I know what to work on?**
A: Check our [issues labeled "good first issue"](https://github.com/Cold-Mountain/manuscript-journal-matcher/labels/good%20first%20issue) or review our [Implementation Roadmap](docs/IMPLEMENTATION_ROADMAP.md).

**Q: Can I work on something not in the roadmap?**
A: Yes! Please create an issue to discuss your idea first, especially for large changes.

**Q: How long does PR review take?**
A: We aim to respond within 1-2 weeks. Smaller PRs typically get faster reviews.

---

## 📞 Contact

### Maintainers
- **Primary Maintainer**: [@Cold-Mountain](https://github.com/Cold-Mountain)
- **Review Team**: See [CODEOWNERS](.github/CODEOWNERS) file

### Reporting Issues
- **Bugs**: [Bug Report Template](https://github.com/Cold-Mountain/manuscript-journal-matcher/issues/new?template=bug_report.md)
- **Features**: [Feature Request Template](https://github.com/Cold-Mountain/manuscript-journal-matcher/issues/new?template=feature_request.md)
- **Security**: Email maintainers directly for security issues

---

<div align="center">

**🌟 Thank you for contributing to the research community!**

[🐛 Report Bug](https://github.com/Cold-Mountain/manuscript-journal-matcher/issues/new) • [💡 Request Feature](https://github.com/Cold-Mountain/manuscript-journal-matcher/issues/new) • [💬 Discussions](https://github.com/Cold-Mountain/manuscript-journal-matcher/discussions)

*Built with ❤️ by researchers, for researchers*

</div>