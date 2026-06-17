FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

ARG OPENVINO_VERSION=2026.2.0
ARG TRANSFORMERS_REF=v5.11.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl ca-certificates build-essential \
    libglib2.0-0 libgl1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --pre --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

RUN pip install pytest openvino==${OPENVINO_VERSION} nncf accelerate pillow sentencepiece protobuf tiktoken numpy

# Fetch the requested Transformers tag during image build and reuse it at runtime.
# This avoids runtime `git clone` flakiness after optimum-intel's editable install
# downgrades Transformers via dependency resolution.
RUN mkdir -p /opt \
 && curl -L --fail --retry 5 --retry-delay 3 --retry-connrefused \
      "https://github.com/huggingface/transformers/archive/refs/tags/${TRANSFORMERS_REF}.tar.gz" \
      -o /tmp/transformers.tar.gz \
 && tar -xzf /tmp/transformers.tar.gz -C /opt \
 && mv /opt/transformers-* /opt/transformers-src \
 && python -m pip install --no-deps /opt/transformers-src \
 && rm -f /tmp/transformers.tar.gz


COPY . /repo
WORKDIR /repo
RUN chmod +x /repo/docker/entrypoint.sh /repo/scripts/apply_patch_rebuild_test.sh
ENTRYPOINT ["/repo/docker/entrypoint.sh"]
