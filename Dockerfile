# =====================================================================
# 1. BASE SYSTEM RUNTIME LAYER
# =====================================================================
# Use an official lightweight Python image to minimize container size
FROM python:3.10-slim

# Set environment system configurations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establish the secure system workspace working directory path inside the container
WORKDIR /app

# =====================================================================
# 2. SYSTEM INDEPENDENT PACKAGES LAYER
# =====================================================================
# Install lightweight OS tool build dependencies and clean up cache vectors instantly
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# =====================================================================
# 3. PIP DEPENDENCY COMPILATION LAYER
# =====================================================================
# Copy python requirement files first to cache Docker building step layers safely
COPY requirements.txt .

# Install pinned core library modules directly without saving standard package caches
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =====================================================================
# 4. ENGINE ASSETS STORAGE MIGRATION LAYER
# =====================================================================
# Copy remaining project codebase asset folders and configurations into container memory
COPY . .

# Create internal directory frames to ensure log streams write correctly
RUN mkdir -p models data

# =====================================================================
# 5. NETWORKING & LIFECYCLE MONITORING LAYERS
# =====================================================================
# Expose the network ports required for application execution
# Port 8501 is default for Streamlit | Port 8000 is default for FastAPI
EXPOSE 8501 8000

# Set automated baseline internal node container stability health check guidelines
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# =====================================================================
# 6. APP BOOT ENGINE STARTUP EXECUTION DECREE
# =====================================================================
# Configure standard startup parameters to spin up the Streamlit UI dashboard
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
