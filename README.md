# Identity Factory API

Backend API for the Identity Circuit Factory (FastAPI). The full codebase guide and route catalog live in:
- `identity-factory-api/docs/README.md`
- `identity-factory-api/docs/API_REFERENCE.md`

## Quickstart
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Run the API
python start_api.py --port 8000
```

## Dependencies
SAT-based synthesis requires `sat_revsynth` on `PYTHONPATH` and a SAT solver binary:
```bash
export PYTHONPATH="/path/to/sat_revsynth/src:$PYTHONPATH"
```

