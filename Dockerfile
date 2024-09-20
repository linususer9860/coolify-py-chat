# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama (assuming it's needed for embeddings)
RUN apt-get update && apt-get install -y curl && \
    curl https://ollama.ai/install.sh | sh

# Make port 8000 available to the world outside this container
EXPOSE 3000

# Run app.py when the container launches
CMD ["chainlit", "run", "app.py", "--port", "8000"]
