FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && \\
    apt-get install --no-install-recommends -y build-essential gcc && \\
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
WORKDIR .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY train.py .
COPY data/ ./data/

# Set the entrypoint
ENTRYPOINT ["python", "-u", "train.py"]
