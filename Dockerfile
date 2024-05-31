# Use the official Python image from the Docker Hub
FROM python:3.9

# Install necessary packages
RUN apt-get update && \
    apt-get install -y cmake && \
    apt-get install -y --no-install-recommends \
    g++ \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Add swap memory
RUN fallocate -l 2G /swapfile && \
    chmod 600 /swapfile && \
    mkswap /swapfile && \
    swapon /swapfile

# Set MAKEFLAGS to limit parallel jobs
ENV MAKEFLAGS="-j1"

# Install specific version of dlib from pre-built binaries
RUN pip install https://files.pythonhosted.org/packages/99/3d/0df5054e98e4e77d8202365b29c1c16d96216a5e4ae8e4f3b09d4ab0f18a/dlib-19.24.0-cp39-cp39-macosx_10_9_x86_64.whl

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any other dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Specify the command to run the application
CMD ["streamlit", "run", "app.py"]
