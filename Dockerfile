FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
 git \
 curl \
 build-essential \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for caching
COPY pyproject.toml .
COPY README.md .

# Create src directory structure for editable install
RUN mkdir -p src/data_systems_toolkit

# Install the package with all optional dependencies
RUN pip install --no-cache-dir -e ".[all]"

# Set Python path
ENV PYTHONPATH=/workspace/src

# Default command
CMD ["python"]
