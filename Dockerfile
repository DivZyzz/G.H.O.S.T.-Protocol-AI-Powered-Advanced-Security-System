# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y cmake && \
    apt-get install -y build-essential && \
    apt-get install -y libopenblas-dev liblapack-dev && \
    apt-get install -y libboost-all-dev && \
    apt-get clean

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Specify the command to run the application
CMD ["GHOST_Protocol", "run", "app.py"]

# Install system dependencies
RUN apt-get update && \
    apt-get install -y cmake && \
    cmake --version && \   # Debug step to verify CMake installation
    apt-get install -y build-essential && \
    apt-get install -y libopenblas-dev liblapack-dev && \
    apt-get install -y libboost-all-dev && \
    apt-get clean
