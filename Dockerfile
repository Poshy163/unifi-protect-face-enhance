FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime libs for opencv-python-headless / openvino (local AI backend).
# INTEL_GPU=1 (default) also installs the Intel NEO OpenCL runtime so OpenVINO
# can target the Intel iGPU (AI_PROVIDER=local + OPENVINO_DEVICE=GPU). The
# intel-opencl-icd package lives in Debian's non-free area, so we enable it
# first. The whole GPU step is best-effort: if it can't install, the image still
# builds and works on CPU (check /api/ai/status -> availableDevices). Build with
# --build-arg INTEL_GPU=0 to skip it entirely (smaller CPU/Gemini-only image).
ARG INTEL_GPU=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
 && if [ "$INTEL_GPU" = "1" ]; then \
        ( sed -i 's/Components: main/Components: main contrib non-free non-free-firmware/' \
              /etc/apt/sources.list.d/debian.sources \
          && apt-get update \
          && apt-get install -y --no-install-recommends intel-opencl-icd ) \
        || echo "WARNING: intel-opencl-icd unavailable; OPENVINO_DEVICE=GPU won't work, CPU still does"; \
    fi \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Local AI models download here on first use (AI_PROVIDER=local). Mount a
# volume at this path to persist them across container recreates.
ENV LOCAL_MODEL_DIR=/home/enhancer/.cache/unifi-protect-face

# Create the model dir owned by the runtime user BEFORE the volume mounts, so an
# empty named volume inherits enhancer ownership (Docker copies the dir's owner
# to the fresh volume) — otherwise the volume is root-owned and uid 1000 can't
# write to it.
RUN useradd --create-home --uid 1000 enhancer \
 && mkdir -p "$LOCAL_MODEL_DIR" \
 && chown -R enhancer:enhancer /app /home/enhancer/.cache
USER enhancer

EXPOSE 8080

CMD ["python", "-m", "app.main"]
