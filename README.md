# Identity Factory API

Backend API for the Identity Circuit Factory - a tool for generating and analyzing reversible circuit identity templates.

## Setup

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

This API requires access to `sat_revsynth` for SAT-based circuit synthesis:

```bash
# Either install it or ensure it's in PYTHONPATH
export PYTHONPATH="/path/to/sat_revsynth/src:$PYTHONPATH"
```

## API Endpoints

- `GET /api/v1/stats` - Database statistics
- `GET /api/v1/eca57-lmdb/stats` - ECA57 enumeration stats
- `GET /api/v1/eca57-lmdb/circuits/{config}` - Get circuits by config
- ... and more

## Structure

```
identity-factory-api/
├── api/           # FastAPI endpoints
├── database.py    # Database handling
├── start_api.py   # Server entry point
└── ...
```
