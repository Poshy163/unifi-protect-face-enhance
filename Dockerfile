# Pinned to bookworm on purpose: intel-opencl-icd (the Intel GPU OpenCL runtime)
# is NOT packaged in Debian trixie, which the unpinned python:3.12-slim now
# points at. Bookworm ships it, so OPENVINO_DEVICE=GPU works out of the box.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime libs for opencv-python-headless / openvino (local AI backend).
# INTEL_GPU=1 (default) also installs the Intel NEO OpenCL runtime so OpenVINO
# can target the Intel iGPU (AI_PROVIDER=local + OPENVINO_DEVICE=GPU):
#   * ocl-icd-libopencl1 — the OpenCL ICD loader (provides libOpenCL.so.1, which
#     the OpenVINO GPU plugin links against).
#   * intel-opencl-icd   — Intel's OpenCL driver (in bookworm main).
# We enable contrib/non-free defensively (covering both the classic and deb822
# sources layouts) in case a dependency needs it. Intel GPUs are x86 only, so
# this is skipped on non-amd64. The install is intentionally fatal on amd64: a
# missing runtime should fail the build loudly, not silently ship a GPU image
# that can't load the plugin at runtime. --build-arg INTEL_GPU=0 skips it all.
ARG INTEL_GPU=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
 && if [ "$INTEL_GPU" = "1" ] && [ "$(dpkg --print-architecture)" = "amd64" ]; then \
        { sed -i 's/ main/ main contrib non-free non-free-firmware/' /etc/apt/sources.list 2>/dev/null || true; } \
     && { sed -i 's/Components: main/Components: main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true; } \
     && apt-get update \
     && apt-get install -y --no-install-recommends ocl-icd-libopencl1 intel-opencl-icd; \
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
