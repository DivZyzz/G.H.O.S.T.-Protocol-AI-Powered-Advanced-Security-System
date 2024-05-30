# Use the official Python image from the Docker Hub
FROM python:3.9

# Install necessary dependencies
RUN apt-get update && apt-get install -y cmake

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install -r /app/requirements.txt

# Install dlib separately to handle CMake dependency
RUN pip install dlib

# Set the working directory
WORKDIR /app

# Copy the rest of the application code
COPY . /app

# Expose the port that Streamlit uses
EXPOSE 8501

# Run Streamlit when the container launches
ENTRYPOINT ["streamlit", "run"]
CMD ["GHOST_Protocol.py"]  # Replace with your Streamlit app's main file
