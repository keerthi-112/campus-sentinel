# Built from the REPO ROOT (see docker-compose.yml): the backend imports vision/, rag/ and federated/.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
# CPU-only PyTorch first keeps the image ~2 GB smaller than the default CUDA wheels.
# If that index is unreachable (some networks block it), fall back to the normal PyPI wheels.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install torch torchvision --index-url ${TORCH_INDEX} || pip install torch torchvision
COPY vision/requirements.txt vision/requirements.txt
COPY rag/requirements.txt rag/requirements.txt
COPY federated/requirements.txt federated/requirements.txt
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY . .
RUN chmod +x backend/docker-entrypoint.sh
WORKDIR /workspace/backend
EXPOSE 8000
ENTRYPOINT ["./docker-entrypoint.sh"]
