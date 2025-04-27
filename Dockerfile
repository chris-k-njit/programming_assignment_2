# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables (optional if needed)
# ENV PYTHONUNBUFFERED=1

# Run your main training script (adjust the path if needed)
CMD ["python", "training/training_model.py"]
