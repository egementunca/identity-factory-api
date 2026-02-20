FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt requirements-dev.txt pyproject.toml setup.py ./
COPY identity_factory ./identity_factory
COPY start_api.py ./start_api.py

RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["python", "start_api.py", "--host", "0.0.0.0", "--port", "8000"]
