# FROM vllm/vllm-openai:latest
FROM vllm/vllm-openai:v0.10.2

WORKDIR /app

COPY requirements.txt requirements.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    build-essential \
    libsndfile1 \
    libsm6 \
    libxext6 \
    wget \
    && \
    grep -v -E '^vllm(==|$)' requirements.txt > /tmp/requirements.docker.txt \
    && \
    pip install --no-cache-dir --break-system-packages -r /tmp/requirements.docker.txt \
    && \
    apt-get purge -y --auto-remove build-essential \
    && \
    rm -rf /root/.cache/pip /tmp/requirements.docker.txt \
    && \
    rm -rf /var/lib/apt/lists/* \
    && \
    ln -sf /usr/bin/python3 /usr/bin/python

# COPY assets /app/assets
COPY indextts /app/indextts
COPY tools /app/tools
COPY patch_vllm.py /app/patch_vllm.py
COPY api_server.py /app/api_server.py
COPY api_server_v2.py /app/api_server_v2.py
COPY convert_hf_format.py /app/convert_hf_format.py
COPY convert_hf_format.sh /app/convert_hf_format.sh
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh /app/convert_hf_format.sh
# Include pre-downloaded model checkpoints (ensure you have ./checkpoints in build context)

ENTRYPOINT /app/entrypoint.sh
