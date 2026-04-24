# Stage 1 — build frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
COPY frontend/tsconfig.json ./
COPY frontend/vite.config.ts ./
COPY frontend/src ./src

RUN npm ci
RUN npm run build

# Stage 2 — backend runtime
FROM python:3.11-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Minimal system deps for pillow/opencv-headless
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     build-essential \
     libgl1 \
     libglib2.0-0 \
     libsm6 \
     libxrender1 \
     libxext6 \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps (upgrade build tools first)
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel
# Preinstall binary numpy wheel to avoid source builds
RUN python -m pip install numpy==1.23.5 --only-binary=:all:
RUN python -m pip install -r requirements.txt

# Copy app source
COPY . .

# Copy built frontend from builder stage
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port and use $PORT at runtime
ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
