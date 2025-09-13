# Use a verified, modern Runpod image with Python 3.11
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Copy your project files
COPY . /app

# Install the dependencies from your requirements.txt file.
# This step might temporarily break the torch installation.
RUN pip install --no-cache-dir -r requirements.txt

# As a final step, FORCE the re-installation of a known-good, CUDA-compatible PyTorch stack.
# This overwrites any incompatible versions installed by the dependencies above.
RUN pip install --no-cache-dir --force-reinstall torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Command to run your handler script when the worker starts
CMD ["python", "handler.py"]
