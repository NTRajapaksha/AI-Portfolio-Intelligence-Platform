# Use a lightweight Python 3.11 Linux image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Install system dependencies required for Prophet & compilation
# (Prophet needs C++ compilers to build its underlying math engine)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy requirements first to leverage Docker caching
# This way, if you change app.py, Docker won't re-install pandas (slow)
COPY requirements.txt .

# 3. Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your application code
COPY . .

# 5. Create the assets folder (for plots) to avoid permission errors
RUN mkdir -p assets

# 6. Expose the port Streamlit runs on
EXPOSE 8501

# 7. Health check to ensure the app is actually running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 8. Run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]