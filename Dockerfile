# Use a lightweight Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install uv for fast dependency management
RUN pip install uv

# Copy only dependency files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Sync dependencies (without dev dependencies)
RUN uv sync --frozen --no-dev

# Copy the rest of the application code
COPY . .

# Use the PORT environment variable provided by Render (default to 8000 if not set)
# We use 'sh -c' to expand the $PORT environment variable
CMD ["sh", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]