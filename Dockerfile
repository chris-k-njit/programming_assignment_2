FROM openjdk:11

# Set environment vars
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip3 install pyspark numpy

# Create working directory
WORKDIR /app

# Copy project files
COPY . /app

# Default command
CMD ["python3", "training/local_prediction.py"]
