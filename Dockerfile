# Use the official Python image from the Docker Hub
FROM python:3.11

# Install necessary packages
RUN apt-get update && apt-get install -y cmake g++ && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Add swap memory
RUN fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile

# Set MAKEFLAGS to limit parallel jobs
ENV MAKEFLAGS="-j1"

# Copy the pre-built dlib wheel
COPY dlib-19.24.4-py3.9-macosx-10.9-x86_64.egg .

# Install the pre-built dlib wheel
RUN pip install dlib-19.24.4-cp311-cp311-macosx_10_9_x86_64.whl --no-cache-dir

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any other dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Specify the command to run the application
CMD ["streamlit", "run", "GHOST_Protocol.py"]
