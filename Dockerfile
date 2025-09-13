# Use an official Python runtime as the base image
FROM python:3.11-slim



# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run the handler application
# This command will be executed by Runpod to start your worker.
CMD ["python", "handler.py"]
