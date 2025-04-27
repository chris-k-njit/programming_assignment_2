# Use an official Python runtime as a parent image
FROM python:3.11-slim


# Install Java (default-jdk = OpenJDK 17) + system dependencies
RUN apt-get update && apt-get install -y \
    default-jdk \
    procps \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH


# Set working directory
WORKDIR /app

# Copy project files into the container
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Set environment variables (optional if needed)
# ENV PYTHONUNBUFFERED=1

# Run your main training script (adjust the path if needed)
CMD ["python", "app/training/prediction_model.py"]
