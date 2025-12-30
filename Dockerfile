# Use Python 3.11 as base image (required for numpy 2.4.0)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including tkinter for GUI)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    python3-tk \
    tk-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter (not in requirements.txt)
RUN pip install --no-cache-dir jupyter jupyterlab

# Copy project files
COPY . .

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose Jupyter port
EXPOSE 8888

# Default command: run Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
