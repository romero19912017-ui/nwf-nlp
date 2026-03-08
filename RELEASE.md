# Release: nwf-nlp

## Publish to PyPI

If version 0.1.0 is already on PyPI, bump version in pyproject.toml first.

```bash
cd c:\nwf\libraries\nwf-nlp
pip install build twine
python -m build
twine upload dist/*
```

## Git

```bash
git add examples/ notebooks/ README.md pyproject.toml .gitignore RELEASE.md
git commit -m "Add examples: 20newsgroups with argparse, viz; notebooks; README with application areas"
git push origin main
```
