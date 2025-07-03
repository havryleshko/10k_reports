# Use official Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first (to leverage Docker cache)
COPY requirements.txt .

# Install system dependencies needed for H2O and others
RUN apt-get update && apt-get install -y default-jre && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project files into the container
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "8000"]
