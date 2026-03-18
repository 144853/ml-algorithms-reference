FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY mlops/common/requirements-serve.txt /app/requirements-serve.txt
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy the full repository (algorithms + serving code)
COPY . /app/

# Copy serving code
COPY mlops/common/serve.py /app/serve.py
COPY mlops/common/train_wrapper.py /app/train_wrapper.py

# Default model directory (mounted or copied during deployment)
ENV MODEL_DIR=/opt/ml/model
ENV PORT=8080

EXPOSE 8080

# Default: run the serving endpoint
CMD ["python", "serve.py"]
