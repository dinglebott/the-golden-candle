# Base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy and install dependencies first (Docker layer caching)
# This layer only rebuilds if requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]