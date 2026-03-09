# Contributing to CauST

Thank you for your interest in contributing to CauST!

## Development Setup

```bash
git clone https://github.com/prthmmkhija1/CauST.git
cd CauST
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- Follow PEP 8
- Use type hints for function signatures
- Add docstrings (NumPy style) for public functions

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes and add tests
4. Run `pytest tests/ -v` to ensure all tests pass
5. Commit and push to your fork
6. Open a Pull Request against `main`

## Reporting Issues

Open an issue on GitHub with:

- A clear description of the problem
- Steps to reproduce
- Expected vs actual behaviour
- Python/torch/torch-geometric versions

## Project Structure

See [README.md](README.md#project-structure) for an overview of the codebase.
