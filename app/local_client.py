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

Matching pipeline (per face crop): a SCRFD face detector (the one bundled inside
the InsightFace pack itself) finds the face and its 5 keypoints, the crop is
warped to ArcFace's canonical 112x112 layout via a least-squares similarity fit,
then the recognition model embeds it. The alignment step is critical — ArcFace
is trained on landmark-aligned faces, so feeding raw, unaligned crops badly hurts
accuracy (this is the step Frigate / CompreFace / Scrypted also do). SCRFD is far
stronger than the old standalone YuNet on off-angle / profile / partial / small
faces, which is exactly when matching used to silently fall back. If no face is
detected in a crop, it still falls back to a plain resize (lower accuracy).

The "best avatar" pick is a pure image-quality heuristic (sharpness + exposure +
contrast + size) — embeddings can't rate crop quality, and this needs no model.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
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


# InsightFace packs published on GitHub releases. Each zip bundles a SCRFD
# detector *and* an ArcFace recognition .onnx; we extract and use both (OpenVINO
# reads .onnx natively). The detector emits a bbox + 5 keypoints per face — what
# we warp to ArcFace's canonical layout — and is much stronger than the old
# standalone YuNet on off-angle / profile / partial faces. Ordered weakest→strongest:
#   buffalo_s  — MobileFaceNet (~13 MB)  + SCRFD-500M, fastest, least accurate
#   buffalo_l  — ResNet50 / WebFace600K (~166 MB) + SCRFD-10G, strong default
#   antelopev2 — ResNet100 / Glint360K (~260 MB) + SCRFD-10G, most accurate, ~2x slower
# Tuple: (zip url, recognition member, detector member).
_PACKS: dict[str, tuple[str, str, str]] = {
    "buffalo_s": (
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
        "w600k_mbf.onnx",
        "det_500m.onnx",
    ),
    "buffalo_l": (
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "w600k_r50.onnx",
        "det_10g.onnx",
    ),
    "antelopev2": (
        "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
        "glintr100.onnx",
        "scrfd_10g_bnkps.onnx",
    ),
}

_DEFAULT_PACK = "buffalo_l"

# Canonical 5-point destination for ArcFace alignment on a 112x112 crop
# (left eye, right eye, nose, left mouth, right mouth, in image coords). SCRFD's
# keypoint order lines up with this directly.
_ARCFACE_DST = [
    (38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366),
    (41.5493, 92.3655), (70.7299, 92.2041),
]


def _umeyama(src: "np.ndarray", dst: "np.ndarray") -> "np.ndarray | None":
    """Least-squares similarity transform (rotation + uniform scale + translation)
    mapping ``src`` onto ``dst`` (Umeyama 1991) — a 2x3 affine matrix.

    This is what InsightFace's canonical ``norm_crop`` uses: a least-squares fit
    over *all* 5 landmarks. The previous code used ``cv2.estimateAffinePartial2D``
    with LMEDS, an outlier-rejecting estimator that — with only 5 points — can
    throw away a *good* landmark and warp the face poorly. Returns None only if
    the points are degenerate (all coincident)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    num, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = dst_demean.T @ src_demean / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.float64)
    U, S, Vt = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return None
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            T[:dim, :dim] = U @ Vt
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ Vt
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ Vt
    var = src_demean.var(axis=0).sum()
    if var <= 1e-12:
        return None
    scale = 1.0 / var * (S @ d)
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean)
    T[:dim, :dim] *= scale
    return T[:dim].astype(np.float32)


def _writable_dir(path: Path) -> Path:
    """Return `path` if it can be created and written, else a temp-dir fallback.

    Guards against a model volume that's mounted root-owned while we run as a
    non-root user — without this the first download raises PermissionError and
    the AI call 500s. The fallback works but isn't persistent (re-downloads on
    restart), so the real fix is fixing the mount's ownership."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.touch()
        probe.unlink()
        return path
    except Exception:
        fallback = Path(tempfile.gettempdir()) / "unifi-protect-face"
        fallback.mkdir(parents=True, exist_ok=True)
        print(
            f"[local_client] WARNING: {path} not writable — caching models in "
            f"{fallback} instead (not persistent; fix the volume ownership).",
            flush=True,
        )
        return fallback


def _model_dir() -> Path:
    d = os.getenv("LOCAL_MODEL_DIR", "").strip()
    base = Path(d) if d else (Path.home() / ".cache" / "unifi-protect-face")
    return _writable_dir(base)


def _pack_paths(pack: str) -> tuple[Path, Path]:
    """(recognition path, detector path) for a pack inside the model cache."""
    _, recog_member, det_member = _PACKS[pack]
    dest_dir = _model_dir() / pack
    return dest_dir / recog_member, dest_dir / det_member


def _ensure_pack(pack: str) -> None:
    """Download the pack zip once (if needed) and extract both the recognition
    and detector models. No-op once both already exist on disk.

    Kept in stdlib so the image needs no extra deps; first run pays a one-time
    download. Installs that already cached only the recognition model from an
    older version re-fetch the zip once to pull the SCRFD detector out of it."""
    if pack not in _PACKS:
        raise ValueError(
            f"unknown LOCAL_FACE_PACK '{pack}' (known: {', '.join(_PACKS)})"
        )
    url, recog_member, det_member = _PACKS[pack]
    recog_path, det_path = _pack_paths(pack)
    if recog_path.exists() and det_path.exists():
        return

    dest_dir = recog_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{pack}.zip"
    urllib.request.urlretrieve(url, zip_path)  # noqa: S310 - pinned GitHub URL
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member, out in ((recog_member, recog_path), (det_member, det_path)):
                if out.exists():
                    continue
                name = next((n for n in zf.namelist() if n.endswith(member)), None)
                if name is None:
                    raise RuntimeError(f"{member} not found inside {url}")
                with zf.open(name) as src, open(out, "wb") as dst:
                    dst.write(src.read())
    finally:
        try:
            zip_path.unlink()
        except OSError:
            pass


def _ensure_model() -> str:
    """Return a path to a usable recognition model (.onnx or .xml), downloading
    the chosen InsightFace pack on first use. Override with LOCAL_FACE_MODEL to
    point at any ArcFace .onnx / OpenVINO .xml you already have."""
    explicit = os.getenv("LOCAL_FACE_MODEL", "").strip()
    if explicit:
        if not Path(explicit).exists():
            raise FileNotFoundError(f"LOCAL_FACE_MODEL not found: {explicit}")
        return explicit
    pack = (os.getenv("LOCAL_FACE_PACK", _DEFAULT_PACK) or _DEFAULT_PACK).strip()
    _ensure_pack(pack)
    return str(_pack_paths(pack)[0])


def _ensure_detector_model() -> str:
    """Return a path to the pack's SCRFD detector .onnx, downloading the pack on
    first use. Override with LOCAL_DETECT_MODEL to point at any SCRFD-family
    (.onnx with 5-keypoint outputs) detector you already have."""
    explicit = os.getenv("LOCAL_DETECT_MODEL", "").strip()
    if explicit:
        if not Path(explicit).exists():
            raise FileNotFoundError(f"LOCAL_DETECT_MODEL not found: {explicit}")
        return explicit
    pack = (os.getenv("LOCAL_FACE_PACK", _DEFAULT_PACK) or _DEFAULT_PACK).strip()
    _ensure_pack(pack)
    return str(_pack_paths(pack)[1])


def _envf(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _envb(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


class _Scrfd:
    """Minimal SCRFD face detector on the OpenVINO runtime.

    Decodes the standard 3-stride (8/16/32), 2-anchor, 5-keypoint ("bnkps")
    outputs that every detector in :data:`_PACKS` produces. Outputs are mapped to
    strides by shape (not by index/name), so it's robust to export ordering.
    Thread-safe: a fresh infer request per call and read-only anchor caches, so
    the batch matcher can detect concurrently."""

    _STRIDES = (8, 16, 32)
    _NUM_ANCHORS = 2  # all SCRFD *_bnkps / det_* models use 2 anchors per cell

    def __init__(self, compiled: Any, input_size: int = 640,
                 nms_thresh: float = 0.4) -> None:
        self._compiled = compiled
        self._size = int(input_size)
        self._nms_thresh = float(nms_thresh)
        self._n_out = len(compiled.outputs)
        # Precompute anchor centres per stride for the fixed square input. Read
        # only after __init__, so no locking is needed on the hot path.
        self._centers: dict[int, "np.ndarray"] = {}
        self._stride_for_n: dict[int, int] = {}
        for s in self._STRIDES:
            h = self._size // s
            w = self._size // s
            ac = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            ac = (ac * s).reshape(-1, 2)
            ac = np.stack([ac] * self._NUM_ANCHORS, axis=1).reshape(-1, 2)
            self._centers[s] = ac
            self._stride_for_n[ac.shape[0]] = s

    @staticmethod
    def _distance2bbox(points: "np.ndarray", distance: "np.ndarray") -> "np.ndarray":
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def _distance2kps(points: "np.ndarray", distance: "np.ndarray") -> "np.ndarray":
        preds = []
        for i in range(0, distance.shape[1], 2):
            preds.append(points[:, 0] + distance[:, i])
            preds.append(points[:, 1] + distance[:, i + 1])
        return np.stack(preds, axis=-1)

    def _nms(self, dets: "np.ndarray") -> list[int]:
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= self._nms_thresh)[0] + 1]
        return keep

    def detect(self, img_bgr: "np.ndarray",
               score_thresh: float) -> "tuple[np.ndarray, np.ndarray]":
        """Return (bboxes Nx4, keypoints Nx5x2) in the input image's coordinates."""
        sz = self._size
        h0, w0 = img_bgr.shape[:2]
        # Letterbox the crop into a square sz x sz, preserving aspect ratio.
        im_ratio = h0 / float(w0)
        if im_ratio > 1.0:
            new_h, new_w = sz, int(round(sz / im_ratio))
        else:
            new_w, new_h = sz, int(round(sz * im_ratio))
        new_w = max(1, min(new_w, sz))
        new_h = max(1, min(new_h, sz))
        det_scale = new_h / float(h0)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((sz, sz, 3), dtype=np.uint8)
        canvas[:new_h, :new_w, :] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0  # SCRFD mean/std
        blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])

        req = self._compiled.create_infer_request()
        req.infer({0: blob})
        outs = [np.asarray(req.get_output_tensor(i).data) for i in range(self._n_out)]

        # Map each output to a stride by its row count; classify by last dim:
        # 1 -> score, 4 -> bbox distances, 10 -> keypoint distances.
        scores_by_n: dict[int, "np.ndarray"] = {}
        bbox_by_n: dict[int, "np.ndarray"] = {}
        kps_by_n: dict[int, "np.ndarray"] = {}
        for a in outs:
            if a.ndim == 3:
                a = a[0]
            elif a.ndim == 1:
                a = a.reshape(-1, 1)
            n, c = a.shape[0], a.shape[-1]
            if c == 1:
                scores_by_n[n] = a.reshape(-1)
            elif c == 4:
                bbox_by_n[n] = a
            elif c == 10:
                kps_by_n[n] = a

        all_scores, all_bboxes, all_kps = [], [], []
        for n, scores in scores_by_n.items():
            stride = self._stride_for_n.get(n)
            if stride is None or n not in bbox_by_n or n not in kps_by_n:
                continue
            keep = scores >= score_thresh
            if not keep.any():
                continue
            centers = self._centers[stride]
            bboxes = self._distance2bbox(centers, bbox_by_n[n] * stride)
            kpss = self._distance2kps(centers, kps_by_n[n] * stride)
            all_scores.append(scores[keep])
            all_bboxes.append(bboxes[keep])
            all_kps.append(kpss[keep])

        if not all_scores:
            return np.zeros((0, 4), np.float32), np.zeros((0, 5, 2), np.float32)
        scores = np.concatenate(all_scores)
        bboxes = np.concatenate(all_bboxes) / det_scale
        kpss = np.concatenate(all_kps).reshape(-1, 5, 2) / det_scale
        dets = np.hstack([bboxes, scores[:, None]]).astype(np.float32)
        keep = self._nms(dets)
        return dets[keep, :4], kpss[keep].astype(np.float32)


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
        # With alignment + galleries, true matches sit well above 0.4 and
        # non-matches below, so 0.4 is a sane floor (was 0.3).
        self._sim_unknown = _envf("LOCAL_SIM_UNKNOWN", 0.40)
        self._sim_strong = _envf("LOCAL_SIM_STRONG", 0.55)
        # Identity score = mean of the top-K query×gallery cosine pairs.
        self._gallery_topk = max(1, int(_envf("LOCAL_GALLERY_TOPK", 3)))

        # Loaded lazily on first use so app startup never blocks on a download.
        self._compiled = None
        self._in_size = (112, 112)  # (W, H), ArcFace default
        self._load_lock = threading.Lock()
        self._load_error: str | None = None
        self._bench: dict[str, Any] | None = None  # last benchmark result

        # Face detection + alignment (critical for ArcFace accuracy). Loaded
        # lazily too; if it fails it self-disables and matching falls back to a
        # plain resize (lower accuracy) rather than erroring.
        self._align = _envb("LOCAL_ALIGN", True)
        self._detect_score = _envf("LOCAL_DETECT_SCORE", 0.5)
        # SCRFD square input, rounded to a multiple of 32. Bigger = better recall
        # on small / off-centre faces, slightly slower. 640 = the 10G train size.
        size = int(_envf("LOCAL_DETECT_SIZE", 640))
        self._detect_size = max(160, (size // 32) * 32)
        self._detect_nms = _envf("LOCAL_DETECT_NMS", 0.4)
        # Upscale crops whose short side is below this before detecting, so tiny
        # Protect thumbnails carry enough detail to detect. 0 disables.
        self._detect_min = int(_envf("LOCAL_DETECT_MIN_SIZE", 160))
        self._detector = None
        self._det_lock = threading.Lock()      # guards lazy load
        self._detector_error: str | None = None
        # Diagnostics: how many embeds aligned vs. fell back to a plain resize.
        # A high fallback share means the detector is missing faces (raise
        # LOCAL_DETECT_SIZE / lower LOCAL_DETECT_SCORE, or check detectorError).
        self._det_stats = {"aligned": 0, "fallback": 0}
        self._stats_lock = threading.Lock()

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
            "align": self._align,
            "detector": "scrfd",
            "detectSize": self._detect_size,
            "detectorReady": self._detector is not None,
            "detectorError": self._detector_error,
            "detectStats": self._detect_stats_snapshot(),
        }

    def _detect_stats_snapshot(self) -> dict[str, Any]:
        with self._stats_lock:
            aligned = self._det_stats.get("aligned", 0)
            fallback = self._det_stats.get("fallback", 0)
        total = aligned + fallback
        return {
            "aligned": aligned,
            "fallback": fallback,
            "alignedPct": round(100.0 * aligned / total, 1) if total else None,
        }

    def _bump(self, key: str) -> None:
        with self._stats_lock:
            self._det_stats[key] = self._det_stats.get(key, 0) + 1

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

    def _ensure_detector(self) -> None:
        if self._detector is not None or not self._align:
            return
        with self._det_lock:
            if self._detector is not None or not self._align:
                return
            try:
                path = _ensure_detector_model()
                core = ov.Core()
                model = core.read_model(path)
                # Force a static square input so anchor decoding is deterministic
                # (SCRFD onnx ships with a dynamic spatial shape).
                try:
                    model.reshape([1, 3, self._detect_size, self._detect_size])
                except Exception:
                    pass
                compiled = core.compile_model(model, self._device)
                # Use the model's *actual* input size for the letterbox + anchors,
                # in case reshape was a no-op on an already-static export.
                size = self._detect_size
                try:
                    ps = compiled.input(0).get_partial_shape()
                    if ps[2].is_static and ps[3].is_static:
                        size = int(ps[2].get_length())
                except Exception:
                    pass
                self._detector = _Scrfd(compiled, size, self._detect_nms)
            except Exception as e:
                self._detector_error = f"{type(e).__name__}: {e}"
                self._align = False  # degrade to resize; stop retrying
                print(
                    f"[local_client] WARNING: face detector unavailable "
                    f"({self._detector_error}); matching WITHOUT alignment "
                    f"(lower accuracy). Set LOCAL_DETECT_MODEL or LOCAL_ALIGN=false.",
                    flush=True,
                )

    def _to_blob(self, img_bgr: "np.ndarray") -> "np.ndarray":
        """112x112 (or model in_size) BGR -> normalised NCHW blob for ArcFace."""
        if (img_bgr.shape[1], img_bgr.shape[0]) != self._in_size:
            img_bgr = cv2.resize(img_bgr, self._in_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 127.5  # ArcFace mean/std
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])

    def _prep(self, jpeg: bytes) -> "np.ndarray":
        """Fallback path: resize the whole crop (no alignment)."""
        arr = np.frombuffer(jpeg, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise ValueError("could not decode image")
        return self._to_blob(img)

    def _prep_aligned(self, jpeg: bytes) -> "np.ndarray | None":
        """Detect the face, warp its 5 keypoints to ArcFace's canonical layout,
        and return the blob. None if no face is found (caller falls back)."""
        self._ensure_detector()
        if self._detector is None:
            return None
        arr = np.frombuffer(jpeg, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            return None
        # Upscale tiny crops so small / edge faces carry enough detail to detect.
        h, w = img.shape[:2]
        short = min(h, w)
        if 0 < short < self._detect_min:
            s = self._detect_min / float(short)
            img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                             interpolation=cv2.INTER_CUBIC)
        try:
            bboxes, kpss = self._detector.detect(img, self._detect_score)
        except Exception:
            return None
        if len(bboxes) == 0:
            return None
        # Largest detected face (the subject of a Protect crop).
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        lmk = kpss[int(np.argmax(areas))].astype(np.float32)
        M = _umeyama(lmk, np.array(_ARCFACE_DST, dtype=np.float32))
        if M is None:
            return None
        aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0)
        return self._to_blob(aligned)

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
        blob = self._prep_aligned(jpeg) if self._align else None
        if blob is None:
            blob = self._prep(jpeg)  # no face detected → resize fallback
            self._bump("fallback")
        else:
            self._bump("aligned")
        emb = self._infer(blob)
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

    def _aggregate(self, sims: list[float]) -> float:
        """Identity score from a bag of query×gallery cosine similarities: the
        mean of the top-K. Top-K (not plain max) resists a single lucky pair;
        not the global mean, so a few off-angle gallery shots don't drag a true
        match down."""
        sims.sort(reverse=True)
        k = min(len(sims), self._gallery_topk)
        return sum(sims[:k]) / k if k else -1.0

    def suggest(self, query_image: bytes, identities: list[dict],
                cache_key: str | None = None,
                query_images: list[bytes] | None = None) -> dict[str, Any]:
        """Match a query face against identities by embedding cosine similarity.

        Multi-sample aware: each identity may carry ``images`` (a gallery of
        sample crops) and the query may carry ``query_images`` (several crops of
        the same cluster). The identity score is the mean of the top-K cosine
        similarities over all query×gallery pairs — far more robust than a single
        reference photo. Falls back to the single ``image``/``query_image`` when
        lists aren't supplied. Same return schema as GeminiClient.suggest."""
        if cache_key is not None:
            fp = ",".join(sorted(i["id"] for i in identities))
            with self._cache_lock:
                hit = self._result_cache.get((cache_key, fp))
            if hit is not None:
                return hit

        q_sources = query_images or ([query_image] if query_image else [])
        q_embs: list[np.ndarray] = []
        for qi in q_sources:
            try:
                q_embs.append(self._embed(qi))
            except Exception:
                continue

        best_score = -1.0
        best: dict | None = None
        best_pairs = 0
        if q_embs:
            for ident in identities:
                g_sources = ident.get("images") or (
                    [ident["image"]] if ident.get("image") else [])
                sims: list[float] = []
                for gi in g_sources:
                    try:
                        ge = self._embed(gi)
                    except Exception:
                        continue
                    sims.extend(float(np.dot(qe, ge)) for qe in q_embs)
                if not sims:
                    continue
                score = self._aggregate(sims)
                if score > best_score:
                    best_score, best, best_pairs = score, ident, len(sims)

        if not q_embs:
            result = {"identityId": None, "name": None, "confidence": 0.0,
                      "reason": "could not embed query face"}
        elif best is None or best_score < self._sim_unknown:
            result = {
                "identityId": None,
                "name": None,
                "confidence": self._confidence(best_score) if best is not None else 0.0,
                "reason": (
                    f"no match (best cosine {best_score:.2f} < {self._sim_unknown:.2f})"
                    if best is not None else "no comparable identities"
                ),
            }
        else:
            k = min(best_pairs, self._gallery_topk)
            result = {
                "identityId": best["id"],
                "name": best["name"],
                "confidence": self._confidence(best_score),
                "reason": (f"ArcFace cosine {best_score:.2f} "
                           f"(top-{k} of {best_pairs} samples) → {best['name']}"),
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
