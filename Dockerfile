# Use a verified official Runpod image with PyTorch and CUDA
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Copy your requirements file and handler script
COPY . /app

# Install the remaining packages from your requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Command to run your handler script
CMD ["python", "handler.py"]
