# syntax=docker/dockerfile:1

# ── Python 3.11 environment for the FastAPI sketch server ──────────────────────
FROM python:3.11-slim

# Faster, cleaner container Python: no .pyc files, unbuffered stdout/stderr so
# logs show up immediately, and pip doesn't keep its download cache.
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1

# The app uses relative imports (`from .config import ...`) so it must run as a
# package. We copy the source into /app/server_suede and put its parent on the
# import path so `python -m server_suede.app` resolves the package.
ENV PYTHONPATH=/app

WORKDIR /app

# Install dependencies first so this layer is cached unless requirements change.
COPY ./.suede/.dependencies/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source into the package directory. Copy the whole tree
# (not just *.py) so subpackages like arc_line_vectorization_suede/ come along;
# .dockerignore keeps runtime data (pending/sketches/combined/presets.json) out.
COPY . ./server/

# Runtime data/static paths in the code are relative to the CWD
# (e.g. FileResponse("static/draw.html"), Path("pending")), so run from inside
# the package directory.
WORKDIR /app/server

# uvicorn binds 0.0.0.0:5000 (see app.py).
EXPOSE 5000

# `python -m` runs the module with package context (relative imports work) and
# with __name__ == "__main__", which triggers app.py's uvicorn.run(...).
CMD ["python", "-m", "server.app"]
