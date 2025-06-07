# Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install OS packages
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libpq-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Install NumPy first
COPY requirements-base.txt /app/
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements-base.txt

# 2) Install PyTorch (CPU) + other deps
COPY requirements.txt /app/
RUN pip install --upgrade pip \
 && pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt

# 3) Copy application code
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# 4) Launch Streamlit
CMD ["streamlit", "run", "webapp.py", "--server.enableCORS=false", "--server.address=0.0.0.0"]
