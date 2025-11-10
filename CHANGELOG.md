# Changelog

All notable changes to the SKS Tuner project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2025-11-10

### Added - Production Readiness Improvements

#### Error Handling & Validation
- **Custom exception types**: Added `SyntheticDataGenerationError` for better error tracking
- **Input validation**: All public methods now validate parameters with clear error messages
- **API key validation**: Added `Config.validate_api_key()` method to ensure required keys are set
- **Enhanced error messages**: Specific exceptions for different failure modes (timeout, HTTP errors, file validation, etc.)
- **Error context**: Use exception chaining (`raise ... from e`) to preserve stack traces

#### Type Safety & Documentation
- **Type hints**: Added comprehensive type hints to all modules:
  - config.py: Full type annotations on all methods
  - sks_parser.py: Type hints with constants for magic numbers
  - sks_downloader.py: Type hints and return type annotations
  - synthetic_generator.py: Type hints for all parameters and returns
- **Docstrings**: Improved docstrings with Args, Returns, and Raises sections
- **Constants**: Replaced magic numbers with named constants:
  - `MIN_LINE_LENGTH = 131`
  - `CODE_LENGTH_LEVEL_1/2/3` for hierarchy determination
  - `MIN_NOTE_LENGTH = 20`
  - `DEFAULT_TIMEOUT = 30`
  - `VALID_CATEGORIES = {"D", "K", "B", "N", "U", "ZZ"}`

#### Logging & Monitoring
- **Centralized logging**: Added `utils/logging_config.py` module with:
  - `setup_logging()` - Configure console and file logging
  - Rotating file handler support (10 MB files, 5 backups)
  - Customizable log levels and formats
  - Noise reduction for third-party libraries
- **Better log messages**: Enhanced logging with file sizes, line counts, and progress tracking
- **Failure tracking**: `generate_dataset()` now tracks and reports failed codes

#### Code Quality
- **Specific exception handling**: Replaced broad `except Exception` with specific exception types
- **Validation on initialization**: Constructor validation for empty API keys
- **Better file validation**: Enhanced SKS file validation with encoding checks and meaningful errors
- **Download robustness**: Separate handling for Timeout, HTTPError, and ConnectionError
- **Category validation**: Warn about unexpected SKS categories

### Changed

#### Dependency Management
- **Removed unused dependencies**: Cleaned up pyproject.toml by commenting out:
  - PyTorch, Transformers, PEFT (not yet used - planned for future)
  - FastAPI, Uvicorn, Pydantic (not yet used - planned for future)
  - OpenAI (not used)
  - W&B (not yet used - planned for future)
  - pandas, numpy, scikit-learn, tqdm (not yet used)
- **Kept core dependencies**:
  - anthropic (for synthetic data generation)
  - click (CLI framework)
  - requests (HTTP downloads)
  - jinja2 (prompt templates)
  - python-dotenv (environment variables)
  - pyyaml (config files)
- **Faster installation**: Reduced dependency count from 30+ to 7 core packages
- **Clearer intent**: Comments explain which dependencies are for future features

#### Documentation
- **README**: Complete rewrite to reflect actual implementation:
  - Clear status indicator showing Phase 1 complete
  - Honest about planned vs. implemented features
  - Detailed usage examples for all scripts
  - Accurate project structure
  - Development and contribution sections
- **.env.example**: Updated to show only required vs. optional variables
- **Inline documentation**: Improved comments and docstrings throughout

#### Configuration
- **Config validation**: Scripts now use `config.validate_api_key()` instead of manual checks
- **Better error reporting**: Config errors provide actionable messages
- **File existence checks**: ModelConfig.from_yaml() validates file exists before reading

### Fixed
- **Silent failures**: `synthetic_generator.generate_for_code()` now raises exceptions instead of returning empty lists
- **Broad exception catching**: Replaced generic exception handlers with specific types
- **Test compatibility**: Fixed test mocking for Anthropic client initialization
- **Download validation**: Added checks for empty files and encoding errors
- **Missing validation**: ModelConfig.from_yaml() now handles missing files and malformed YAML

### Testing
- **All tests passing**: 18/18 tests pass successfully
- **Test improvements**: Fixed mock setup for SyntheticDataGenerator
- **Better coverage**: Tests now cover error paths and validation logic

### Infrastructure
- **Centralized utilities**: Created `src/skstuner/utils/` package
- **Logging module**: Professional logging configuration with rotation
- **Production-ready error handling**: Consistent error handling strategy across all modules

## [0.1.0] - Initial Release

### Added
- SKS code downloader from official Danish health data source
- Fixed-width SKS file parser with hierarchy building
- Data processing and JSON export
- Synthetic clinical note generation using Claude AI
- Jinja2 template system for prompts
- CLI scripts for all operations
- Comprehensive test suite
- Poetry dependency management
- Configuration management with environment variables

---

## Upgrade Notes

### From 0.1.0 to 0.1.1

1. **Update dependencies**:
   ```bash
   rm poetry.lock
   poetry install
   ```

2. **Update .env file** (if you have one):
   - Remove unused keys like `OPENAI_API_KEY` and `WANDB_*` if not needed
   - Keep `ANTHROPIC_API_KEY` as it's required

3. **Update code using SyntheticDataGenerator**:
   - The generator now raises exceptions instead of returning empty lists
   - Wrap calls in try/except to handle `SyntheticDataGenerationError`
   - Use the new `continue_on_error` parameter in `generate_dataset()` if you want old behavior

4. **Update logging** (optional):
   ```python
   from skstuner.utils.logging_config import setup_logging
   setup_logging(level="INFO", log_file=Path("logs/app.log"))
   ```

## Future Roadmap

- **v0.2.0**: Model training pipeline implementation
- **v0.3.0**: Evaluation metrics and benchmarking
- **v0.4.0**: FastAPI inference service
- **v1.0.0**: Full production system with monitoring and deployment
