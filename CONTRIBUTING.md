# Contributing

Thanks for contributing to K2G-BiCTC.

## Before You Start

- Open an issue first for feature changes or refactors.
- Keep pull requests small and focused.
- Do not commit raw datasets, model checkpoints, or generated outputs.

## Development Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Suggested Workflow

1. Create a feature branch.
2. Make your changes with clear commit messages.
3. Run a quick syntax check:

```bash
python -m compileall src scripts
```

4. Update docs if behavior or CLI arguments changed.
5. Open a pull request and include:
- What changed
- Why it changed
- How you validated it

## Style Notes

- Follow existing code style and naming conventions.
- Keep scripts and config defaults reproducible.
- Avoid hard-coded local absolute paths.

## Reporting Issues

Please include:
- Environment (OS, Python, CUDA if relevant)
- Exact command
- Full error log
- Minimal reproduction steps
