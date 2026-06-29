FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime libs for opencv-python-headless / openvino (local AI backend).
# INTEL_GPU=1 (default) also installs the Intel NEO OpenCL runtime so OpenVINO
# can target the Intel iGPU (AI_PROVIDER=local + OPENVINO_DEVICE=GPU). Build with
# --build-arg INTEL_GPU=0 for a smaller CPU/Gemini-only image.
ARG INTEL_GPU=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
 && if [ "$INTEL_GPU" = "1" ]; then \
        apt-get install -y --no-install-recommends intel-opencl-icd; \
    fi \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

RUN useradd --create-home --uid 1000 enhancer \
 && chown -R enhancer:enhancer /app
USER enhancer

# Local AI models download here on first use (AI_PROVIDER=local). Mount a
# volume at this path to persist them across container recreates.
ENV LOCAL_MODEL_DIR=/home/enhancer/.cache/unifi-protect-face

EXPOSE 8080

CMD ["python", "-m", "app.main"]
