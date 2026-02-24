# GPU-enabled Data Science DevContainer
# Based on PyTorch with CUDA 12.1 (compatible with host driver 12.9)

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
 git \
 curl \
 vim \
 build-essential \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install core data science packages
# Note: PyTorch already included in base image
RUN pip install --no-cache-dir \
 numpy>=1.24.0 \
 pandas>=2.0.0 \
 scikit-learn>=1.4.0 \
 matplotlib>=3.7.0 \
 seaborn>=0.12.0

# Install MLflow for experiment tracking
RUN pip install --no-cache-dir \
 mlflow>=2.10.0

# Install HuggingFace datasets
RUN pip install --no-cache-dir \
 datasets>=2.16.0

# Install Jupyter
RUN pip install --no-cache-dir \
 jupyter>=1.0.0 \
 jupyterlab>=4.0.0 \
 ipywidgets>=8.0.0

# Install dev tools
RUN pip install --no-cache-dir \
 ruff>=0.1.0 \
 pytest>=7.4.0

# Default command
CMD ["bash"]
