# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# cache mount helps speed up rebuilding
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Install the package in editable mode so changes are reflected if volume mounted (optional for prod but good for now)
RUN pip install -e .

# Expose ports for documentation (actual exposure handled by compose)
EXPOSE 8370 8371

# Default command placeholder, will be overridden by docker-compose
CMD ["python", "clari_gen/api/main.py"]
