"""On-device face matching via OpenVINO + ArcFace embeddings.

A drop-in alternative to :mod:`app.gemini_client` for the webapp's AI
endpoints. Instead of sending every face to a cloud VLM and paying per call,
this embeds each face crop with an ArcFace model — the same embedding-based
approach Frigate / CompreFace / Scrypted use — and matches by cosine
similarity. It's free, fast (~5-15 ms/face on CPU), private, and for the
"is this the same person?" task it's *more* accurate than VLM reasoning.

Inference runs on the OpenVINO runtime, so it uses your CPU by default or the
Intel iGPU via ``OPENVINO_DEVICE=GPU``. (Raptor Lake / 13th-gen has no NPU, so
those are the two targets.)

Public surface mirrors :class:`app.gemini_client.GeminiClient` so the webapp
can use either backend interchangeably:

  * ``suggest(query_image, identities, cache_key)``
        -> ``{"identityId", "name", "confidence", "reason"}``
  * ``pick_best_avatar(candidate_images)``
        -> ``{"index", "reason"}``

Matching uses an embedding model. The "best avatar" pick is a pure image-quality
heuristic (sharpness + exposure + contrast + size) — embeddings can't rate crop
quality, and this needs no extra model.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

# Lazy/optional imports — the webapp still imports this module (and
# ``is_available()`` returns False) when these aren't installed.
try:
    import cv2
    import numpy as np
    import openvino as ov

    _OV_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    cv2 = None  # type: ignore
    np = None  # type: ignore
    ov = None  # type: ignore
    _OV_AVAILABLE = False


# InsightFace model packs published on the project's GitHub releases. Each zip
# bundles a detector + an ArcFace recognition .onnx; we only need the latter and
# OpenVINO reads .onnx natively (no conversion step). buffalo_l is the accurate
# default (ResNet50, ~166 MB); buffalo_s is a lighter MobileFaceNet (~13 MB).
_PACKS: dict[str, tuple[str, str]] = {
    "buffalo_l": (
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "w600k_r50.onnx",
    ),
    "buffalo_s": (
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
        "w600k_mbf.onnx",
    ),
}

_DEFAULT_PACK = "buffalo_l"


def _model_dir() -> Path:
    d = os.getenv("LOCAL_MODEL_DIR", "").strip()
    if d:
        return Path(d)
    return Path.home() / ".cache" / "unifi-protect-face"


def _ensure_model() -> str:
    """Return a path to a usable model file (.onnx or .xml), downloading the
    chosen InsightFace pack on first use. Override with LOCAL_FACE_MODEL to
    point at any ArcFace .onnx / OpenVINO .xml you already have."""
    explicit = os.getenv("LOCAL_FACE_MODEL", "").strip()
    if explicit:
        if not Path(explicit).exists():
            raise FileNotFoundError(f"LOCAL_FACE_MODEL not found: {explicit}")
        return explicit

    pack = (os.getenv("LOCAL_FACE_PACK", _DEFAULT_PACK) or _DEFAULT_PACK).strip()
    if pack not in _PACKS:
        raise ValueError(
            f"unknown LOCAL_FACE_PACK '{pack}' (known: {', '.join(_PACKS)})"
        )
    url, member = _PACKS[pack]

    dest_dir = _model_dir() / pack
    onnx_path = dest_dir / member
    if onnx_path.exists():
        return str(onnx_path)

    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{pack}.zip"
    # Download then extract just the recognition model. Kept in stdlib so the
    # image needs no extra deps; first run pays a one-time download.
    urllib.request.urlretrieve(url, zip_path)  # noqa: S310 - pinned GitHub URL
    try:
        with zipfile.ZipFile(zip_path) as zf:
            name = next((n for n in zf.namelist() if n.endswith(member)), None)
            if name is None:
                raise RuntimeError(f"{member} not found inside {url}")
            with zf.open(name) as src, open(onnx_path, "wb") as dst:
                dst.write(src.read())
    finally:
        try:
            zip_path.unlink()
        except OSError:
            pass
    return str(onnx_path)


def _envf(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


class LocalClient:
    def __init__(self) -> None:
        if not _OV_AVAILABLE:
            raise RuntimeError(
                "openvino / opencv / numpy are not installed (pip install "
                "openvino opencv-python-headless numpy)"
            )
        self._device = (os.getenv("OPENVINO_DEVICE", "CPU") or "CPU").strip()
        self._pack = (os.getenv("LOCAL_FACE_PACK", _DEFAULT_PACK) or _DEFAULT_PACK).strip()
        # Cosine-similarity calibration. Below `unknown` -> UNKNOWN; at/above
        # `strong` -> ~1.0 confidence (so the UI auto-checks confident matches).
        self._sim_unknown = _envf("LOCAL_SIM_UNKNOWN", 0.30)
        self._sim_strong = _envf("LOCAL_SIM_STRONG", 0.55)

        # Loaded lazily on first use so app startup never blocks on a download.
        self._compiled = None
        self._in_size = (112, 112)  # (W, H), ArcFace default
        self._load_lock = threading.Lock()
        self._load_error: str | None = None
        self._bench: dict[str, Any] | None = None  # last benchmark result

        # Caches. Embeddings are keyed by image content hash (so each reference
        # face is embedded once across a whole batch). Results mirror the Gemini
        # client's per-(group, identity-set) cache.
        self._emb_cache: dict[str, "np.ndarray"] = {}
        self._result_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._cache_lock = threading.Lock()

    # Exposed for the status endpoint; matches GeminiClient's `_model` attr.
    @property
    def _model(self) -> str:
        return f"{self._pack}@{self._device}"

    def info(self) -> dict[str, Any]:
        # Surface what OpenVINO actually sees so you can confirm the iGPU is
        # detected (look for "GPU" in availableDevices before setting
        # OPENVINO_DEVICE=GPU).
        devices: list[str] = []
        try:
            devices = list(ov.Core().available_devices)
        except Exception:
            pass
        return {
            "backend": "openvino",
            "device": self._device,
            "availableDevices": devices,
            "pack": self._pack,
            "ready": self._compiled is not None,
            "loadError": self._load_error,
            "bench": self._bench,
        }

    # --- inference -------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._compiled is not None:
            return
        with self._load_lock:
            if self._compiled is not None:
                return
            try:
                model_path = _ensure_model()
                core = ov.Core()
                model = core.read_model(model_path)
                self._compiled = core.compile_model(model, self._device)
                # Pull the spatial input size if the model declares it static.
                try:
                    shape = self._compiled.input(0).get_partial_shape()
                    h = shape[2].get_length() if shape[2].is_static else 112
                    w = shape[3].get_length() if shape[3].is_static else 112
                    self._in_size = (int(w), int(h))
                except Exception:
                    self._in_size = (112, 112)
            except Exception as e:
                self._load_error = f"{type(e).__name__}: {e}"
                raise

    def _prep(self, jpeg: bytes) -> "np.ndarray":
        arr = np.frombuffer(jpeg, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise ValueError("could not decode image")
        img = cv2.resize(img, self._in_size, interpolation=cv2.INTER_LINEAR)
        # ArcFace expects RGB, mean/std 127.5, NCHW.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 127.5
        return np.ascontiguousarray(img.transpose(2, 0, 1)[None])

    def _infer(self, blob: "np.ndarray") -> "np.ndarray":
        # A fresh infer request per call so the threaded batch matcher
        # (AI_BATCH_WORKERS) can run embeddings concurrently and safely — the
        # shared CompiledModel.__call__ request is not thread-safe.
        req = self._compiled.create_infer_request()
        req.infer({0: blob})
        emb = req.get_output_tensor(0).data[0].astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-9  # L2-normalise for cosine via dot
        return emb

    def _embed(self, jpeg: bytes) -> "np.ndarray":
        key = hashlib.sha1(jpeg).hexdigest()
        with self._cache_lock:
            hit = self._emb_cache.get(key)
        if hit is not None:
            return hit
        self._ensure_loaded()
        emb = self._infer(self._prep(jpeg))
        with self._cache_lock:
            self._emb_cache[key] = emb
        return emb

    def benchmark(self, runs: int = 30) -> dict[str, Any]:
        """Time pure embedding inference on the active device, so CPU and GPU
        can be compared. Uses a synthetic face-sized image and bypasses the
        embedding cache (it measures the model + device, not a cache hit).
        The result is cached and echoed by ``info()`` / ``/api/ai/status``."""
        self._ensure_loaded()
        rng = np.random.default_rng(0)
        h, w = self._in_size[1], self._in_size[0]
        synthetic = (rng.random((h, w, 3)) * 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", synthetic)
        if not ok:
            raise RuntimeError("could not encode benchmark image")
        blob = self._prep(buf.tobytes())
        for _ in range(3):  # warm up — first infer uploads kernels / weights
            self._infer(blob)
        t0 = time.perf_counter()
        for _ in range(max(1, runs)):
            self._infer(blob)
        dt = time.perf_counter() - t0
        self._bench = {
            "device": self._device,
            "runs": runs,
            "msPerFace": round(dt / runs * 1000.0, 2),
            "facesPerSec": round(runs / dt, 1) if dt > 0 else None,
        }
        return self._bench

    def _confidence(self, sim: float) -> float:
        lo, hi = self._sim_unknown, self._sim_strong
        if hi <= lo:
            return 1.0 if sim >= hi else 0.0
        return float(max(0.0, min(1.0, (sim - lo) / (hi - lo))))

    def suggest(self, query_image: bytes, identities: list[dict],
                cache_key: str | None = None) -> dict[str, Any]:
        """Match `query_image` against `identities` by embedding cosine
        similarity. Same return schema as GeminiClient.suggest."""
        if cache_key is not None:
            fp = ",".join(sorted(i["id"] for i in identities))
            with self._cache_lock:
                hit = self._result_cache.get((cache_key, fp))
            if hit is not None:
                return hit

        q = self._embed(query_image)
        best_sim = -1.0
        best: dict | None = None
        for ident in identities:
            img = ident.get("image")
            if not img:
                continue
            try:
                sim = float(np.dot(q, self._embed(img)))
            except Exception:
                continue
            if sim > best_sim:
                best_sim = sim
                best = ident

        if best is None or best_sim < self._sim_unknown:
            result = {
                "identityId": None,
                "name": None,
                "confidence": self._confidence(best_sim) if best is not None else 0.0,
                "reason": (
                    f"no match (best cosine {best_sim:.2f} < {self._sim_unknown:.2f})"
                    if best is not None else "no comparable identities"
                ),
            }
        else:
            result = {
                "identityId": best["id"],
                "name": best["name"],
                "confidence": self._confidence(best_sim),
                "reason": f"ArcFace cosine {best_sim:.2f} → {best['name']}",
            }

        if cache_key is not None:
            with self._cache_lock:
                self._result_cache[(cache_key, fp)] = result
        return result

    def pick_best_avatar(self, candidate_images: list[bytes]) -> dict[str, Any]:
        """Pick the clearest crop as a reference photo. Pure image-quality
        heuristic — sharpness (Laplacian variance), even exposure, contrast,
        and resolution, each min-max normalised across the candidates."""
        if not candidate_images:
            return {"index": 0, "reason": "no candidates"}
        if len(candidate_images) == 1:
            return {"index": 0, "reason": "only one candidate"}

        sharp: list[float] = []
        expo: list[float] = []
        contrast: list[float] = []
        size: list[float] = []
        for jpeg in candidate_images:
            try:
                arr = np.frombuffer(jpeg, np.uint8)
                gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise ValueError
                sharp.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                # Closer to mid-grey (128) = better exposed; invert so higher=better.
                expo.append(-abs(float(gray.mean()) - 128.0))
                contrast.append(float(gray.std()))
                size.append(float(gray.shape[0] * gray.shape[1]))
            except Exception:
                sharp.append(0.0); expo.append(-128.0); contrast.append(0.0); size.append(0.0)

        def norm(xs: list[float]) -> list[float]:
            lo, hi = min(xs), max(xs)
            if hi - lo < 1e-9:
                return [0.0 for _ in xs]
            return [(x - lo) / (hi - lo) for x in xs]

        ns, ne, nc, nz = norm(sharp), norm(expo), norm(contrast), norm(size)
        scores = [
            0.50 * ns[i] + 0.20 * nc[i] + 0.15 * ne[i] + 0.15 * nz[i]
            for i in range(len(candidate_images))
        ]
        idx = max(range(len(scores)), key=lambda i: scores[i])
        return {
            "index": idx,
            "reason": (
                f"sharpest/best-exposed (laplacian var {sharp[idx]:.0f}, "
                f"contrast {contrast[idx]:.0f})"
            ),
        }


def build_from_env() -> LocalClient | None:
    """Return a LocalClient if the OpenVINO stack is importable, else None."""
    if not _OV_AVAILABLE:
        return None
    return LocalClient()


def is_available() -> bool:
    return _OV_AVAILABLE
