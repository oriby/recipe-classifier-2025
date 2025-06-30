FROM python:3.11-slim

WORKDIR /app
# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# I added this: Install necessary python libraries
RUN pip install numpy scikit-learn sklearn-crfsuite

# Create data directory
RUN mkdir -p /app/data

# Copy the parquet file
COPY src/allrecipes_extended.parquet /app/data

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Expose the port the app runs on
EXPOSE 8080

# Command to run the application - make init.sh executable first
CMD ["sh", "-c", "dos2unix web/init.sh && chmod +x web/init.sh && ./web/init.sh"]