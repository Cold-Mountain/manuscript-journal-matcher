# Changelog

All notable changes to the Manuscript Journal Matcher project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Enhanced algorithm with study type classification
- Multi-modal content analysis
- Performance optimization and caching
- Comprehensive API documentation
- Production deployment guides

## [2.0.0] - 2025-07-28

### Added
- **Production Database**: 7,648 medical journals from Medicine Journal Rankings 2024
- **CSV Data Integration**: Complete processing pipeline for Scimago journal data
- **DOAJ Integration**: Open access information and cost transparency
- **Advanced Filtering**: SJR quartiles, rankings, H-index, publisher filtering
- **Quality Metrics**: Comprehensive journal quality indicators
- **Performance Optimization**: Sub-second search times with FAISS vector database
- **Streamlit Web Interface**: Full-featured web application with file upload
- **Comprehensive Testing**: 10+ test modules with integration testing
- **API Caching**: DOAJ and OpenAlex API response caching
- **Geographic Coverage**: Journals from 104 countries and 2,000+ publishers

### Enhanced
- **Semantic Fingerprints**: Enhanced journal profiling with ranking context
- **Search Algorithm**: FAISS-based cosine similarity with quality ranking
- **Database Schema**: Unified schema supporting CSV and API data sources
- **Error Handling**: Robust error recovery and validation throughout system
- **Memory Management**: Optimized for large-scale journal databases

### Technical Improvements
- **Embedding System**: 384-dimensional vectors using all-MiniLM-L6-v2
- **Vector Search**: FAISS IndexFlatIP for efficient similarity search
- **Data Processing**: Chunked processing for memory efficiency
- **Quality Assurance**: 96.6% validation success rate
- **Performance**: 0.005s average search time, 570MB memory usage

## [1.0.0] - 2025-07-25

### Added
- **Core System Architecture**: Modular design with separation of concerns
- **Document Extraction**: PDF and DOCX processing with metadata extraction
- **Embedding Engine**: Sentence transformers integration for semantic analysis
- **Journal Database Builder**: OpenAlex API integration for journal metadata
- **Vector Matching**: Basic cosine similarity matching implementation
- **Test Framework**: Initial test suite with unit and integration tests

### Components
- `src/extractor.py` - Document processing and metadata extraction
- `src/embedder.py` - Text embedding generation and similarity calculation
- `src/journal_db_builder.py` - Database construction and API integration
- `src/match_journals.py` - Journal matching and similarity search
- `src/main.py` - Streamlit web interface
- `src/config.py` - Configuration management
- `src/utils.py` - Utility functions and helpers

### Infrastructure
- **Project Structure**: Organized codebase with clear module separation
- **Documentation**: Basic README and implementation planning
- **Version Control**: Git initialization with proper .gitignore
- **Dependencies**: Requirements.txt with core Python packages
- **Testing**: pytest-based testing framework

## [0.1.0] - 2025-07-20

### Added
- **Project Initialization**: Basic project structure and planning
- **Requirements Analysis**: System requirements and architecture design
- **Technology Stack Selection**: Python, Streamlit, FAISS, Sentence Transformers
- **Development Environment**: Virtual environment and dependency management

---

## Version History Summary

| Version | Release Date | Description | Key Features |
|---------|--------------|-------------|--------------|
| **2.0.0** | 2025-07-28 | Production Release | 7,600+ journals, DOAJ integration, advanced filtering |
| **1.0.0** | 2025-07-25 | MVP Release | Core matching system, web interface, basic database |
| **0.1.0** | 2025-07-20 | Project Start | Initial planning and architecture |

---

## Future Roadmap

### Version 2.1.0 (Planned)
- **Enhanced Documentation**: Complete user and developer guides
- **API Reference**: Auto-generated documentation from docstrings
- **Performance Monitoring**: System metrics and performance tracking
- **Code Quality**: Expanded test coverage and code standardization

### Version 2.2.0 (Planned)  
- **Study Type Classification**: AI-powered manuscript analysis
- **Advanced Content Analysis**: Multi-section document processing
- **Enhanced Algorithms**: Ensemble matching with multiple models
- **Network Analysis**: Citation and author network integration

### Version 3.0.0 (Future)
- **Production Deployment**: Docker containerization and cloud deployment
- **Real-time Intelligence**: Dynamic journal profiling and trend analysis
- **Advanced ML**: Custom models trained on scientific literature
- **Enterprise Features**: Multi-user support, analytics dashboard, API access

---

## Contributing to Changelog

When contributing to this project, please update this changelog with your changes:

### Format Guidelines
- Use semantic versioning (MAJOR.MINOR.PATCH)
- Group changes into categories: Added, Changed, Deprecated, Removed, Fixed, Security
- Include the date of release in YYYY-MM-DD format
- Link to GitHub issues/PRs when applicable
- Write clear, concise descriptions for end users

### Categories
- **Added**: New features
- **Changed**: Changes in existing functionality  
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security-related changes

### Example Entry
```markdown
## [2.1.0] - 2025-08-15

### Added
- API reference documentation with interactive examples
- Performance monitoring dashboard
- Docker deployment configuration

### Changed
- Improved search algorithm with 25% better accuracy
- Updated UI with material design components

### Fixed
- Memory leak in large file processing
- PDF extraction issues with non-standard formats
```

---

*This changelog is maintained by the project maintainers and community contributors.*