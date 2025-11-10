FROM python:3.11-slim

WORKDIR /app

# Set environment variable for NLTK data location
ENV NLTK_DATA=/app/nltk_data

# Copy dependency files first to leverage Docker cache
COPY requirements.txt ./

# Install dependencies and download NLTK data in a single layer
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader punkt stopwords wordnet

# Copy application code (these change most frequently, done last)
COPY api.py ./api.py
# Uncomment the following lines if the source and configs directories exist
# COPY source ./source
# COPY configs/env.yaml.example ./configs/env.yaml.example

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]