# Use a verified official Runpod image with Python 3.11 and a compatible CUDA version
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Copy your project files
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run your handler
CMD ["python", "handler.py"]
