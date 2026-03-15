FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including X11 libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    python3-tk \
    tk-dev \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxinerama1 \
    libxi6 \
    libxrandr2 \
    libxcursor1 \
    libxtst6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app:$PYTHONPATH
CMD ["python", "main.py"]
