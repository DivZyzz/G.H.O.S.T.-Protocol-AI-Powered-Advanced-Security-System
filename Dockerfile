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

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Add swap memory
RUN fallocate -l 2G /swapfile && \
    chmod 600 /swapfile && \
    mkswap /swapfile && \
    swapon /swapfile

# Set MAKEFLAGS to limit parallel jobs
ENV MAKEFLAGS="-j1"

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Specify the command to run the application
CMD ["streamlit", "run", "app.py"]
