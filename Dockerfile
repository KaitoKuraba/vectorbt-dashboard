FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY backend.py .
COPY index.html .

# Railway injects PORT automatically
ENV PORT=8080

EXPOSE $PORT

CMD ["python", "backend.py"]
