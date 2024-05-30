# Use the official Python image from the Docker Hub
FROM python:3.9

# Install system dependencies
RUN apt-get update && \
    apt-get install -y cmake && \
    apt-get install -y build-essential && \
    apt-get install -y libopenblas-dev liblapack-dev

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies first to leverage Docker's caching mechanism
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that Streamlit uses
EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["GHOST_Protocol.py"]  # Replace with your Streamlit app's main file
 

