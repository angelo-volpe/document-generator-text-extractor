FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential libgl1 libglib2.0-0 wget git

# Copy the jobs
COPY . /app

# Install python dependencies
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "main.py"]