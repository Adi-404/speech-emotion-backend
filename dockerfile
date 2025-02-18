# Use a lightweight base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for building Python packages and FLAC
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libsndfile1 \
    flac \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose the port for Flask
EXPOSE 5050

# Run the Flask app
CMD ["python", "app.py"]