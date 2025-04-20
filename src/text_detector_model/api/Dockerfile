FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
COPY app.py .

# Set environment variables
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Expose the port
EXPOSE 5001

# Run the application
CMD ["python", "app.py"]