"""FastAPI webapp for fast face-group triage.

Endpoints proxy the private Protect API via ProtectClient. The frontend is
a single static page served at /. No CORS — same-origin only.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import local_client
from .enhancer_status import STATUS
from .gemini_client import build_from_env as build_gemini, is_available as gemini_available
from .protect_client import BASE_PROTECT, ProtectClient
from .version import __build_date__, __version__

STATIC_DIR = Path(__file__).parent / "static"

_AI_UNCONFIGURED = (
    "AI matching is not configured. Set AI_PROVIDER=local for on-device "
    "OpenVINO matching, or GEMINI_API_KEY for the Gemini cloud backend."
)


def _gallery_min_conf() -> float:
    """Drop gallery detections below this `matchedGroupConfidence`.

    Low-confidence detections are the ones Protect most often *misattributes* to
    the wrong person; one wrong face in a gallery causes confident false matches.
    Default 0 (off) keeps every confidence; raise to harden galleries if you see
    cross-person matches.

    Scale: `matchedGroupConfidence` is an integer PERCENTAGE (0–100), not a
    0–1 fraction — on a live instance it spans roughly 77–95, so a meaningful
    floor is ~80. This was previously documented as 0–1, which made every
    setting in that range a no-op. Values ≤ 1 are therefore read as the old
    fraction and scaled, so an existing `0.6` now means what it was meant to.
    """
    try:
        floor = max(0.0, float(os.getenv("LOCAL_GALLERY_MIN_CONF", "0") or 0))
    except ValueError:
        return 0.0
    return floor * 100 if 0 < floor <= 1.0 else floor


def _ai_batch_max() -> int:
    """Hard cap on how many faces one ⚡ Auto-match run sends to Gemini.

    Keeps a single click from quietly firing hundreds of billable calls.
    Tunable via AI_BATCH_MAX; 0 disables the cap (use with care)."""
    try:
        return max(0, int(os.getenv("AI_BATCH_MAX", "50") or 50))
    except ValueError:
        return 50


def _build_ai_provider() -> tuple[Any, str]:
    """Pick the backend for the AI suggest / best-avatar endpoints.

    AI_PROVIDER selects it:
      * ``local``  — on-device OpenVINO + ArcFace embeddings (free, private)
      * ``gemini`` — Google Gemini cloud VLM (per-call cost)
      * ``auto`` (default) — keeps existing cloud setups working: uses Gemini
        when GEMINI_API_KEY is set, otherwise falls back to the local model.

    Returns ``(provider_or_None, kind)``. Both backends expose the same
    ``suggest`` / ``pick_best_avatar`` interface so callers don't branch.
    """
    choice = (os.getenv("AI_PROVIDER", "auto") or "auto").strip().lower()
    if choice == "gemini":
        return build_gemini(), "gemini"
    if choice == "local":
        return local_client.build_from_env(), "local"
    # auto: prefer the already-configured cloud key for backward compatibility.
    if os.getenv("GEMINI_API_KEY", "").strip():
        return build_gemini(), "gemini"
    return local_client.build_from_env(), "local"


def is_named_identity(g: dict) -> bool:
    """A group is a 'named identity' if it has a user-given name and isn't degraded.

    Don't filter by the `face_*` id prefix — Protect keeps the original
    auto-cluster id when you rename a group, so some real identities (created
    early in face-recognition history) have ids like face_825 ("Carolyn Dale").
    Trust the name field instead.
    """
    if g.get("isDegraded"):
        return False
    return bool(g.get("name") or g.get("matchedName"))


def group_summary(g: dict) -> dict:
    return {
        "id": g.get("id"),
        "name": g.get("name") or g.get("matchedName"),
        "matchedName": g.get("matchedName"),
        "detectionsCount": g.get("detectionsCount", 0),
        "firstDetectedAt": g.get("firstDetectedAt"),
        "lastDetectedAt": g.get("lastDetectedAt"),
        "isDegraded": bool(g.get("isDegraded")),
        "metadata": g.get("metadata") or {},
    }


class MergeBody(BaseModel):
    fromGroupIds: list[str] = Field(min_length=1)
    toGroupId: str


class RenameBody(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class RetroactiveBody(BaseModel):
    numberOfEvents: int = Field(ge=1, le=100000)
    allCameras: bool = True
    cameraIds: list[str] = []


class DeleteBody(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=500)


class ReassignBody(BaseModel):
    detectionIds: list[str] = Field(min_length=1, max_length=500)
    # Target group id. Use null to unassign (the detections leave the current
    # identity; Protect will re-cluster them on its own).
    groupId: str | None = None


class BestAvatarApplyBody(BaseModel):
    """Apply a candidate enhanced image as the identity's reference avatar."""
    enhancedImageId: str = Field(min_length=1)


class BatchSuggestBody(BaseModel):
    """Group ids (unnamed clusters) to match against known identities in one go."""
    groupIds: list[str] = Field(min_length=1, max_length=1000)


def create_app(client: ProtectClient) -> FastAPI:
    app = FastAPI(title="Face Triage", docs_url="/api/docs", redoc_url=None)
    ai, ai_kind = _build_ai_provider()  # None if the chosen backend is unavailable

    # In-memory cache for reference identity images used by AI suggest.
    # Saves 14+ HTTP round-trips per suggestion after the first call.
    _ref_image_cache: dict[str, tuple[float, bytes]] = {}
    _ref_images_cache: dict[tuple[str, int], tuple[float, list[bytes]]] = {}
    _ref_image_lock = threading.Lock()
    _REF_IMAGE_TTL = 30 * 60  # 30 minutes

    def get_reference_image(group_id: str) -> bytes | None:
        """Return the best face image for an identity (enhanced if possible)."""
        now = time.time()
        with _ref_image_lock:
            cached = _ref_image_cache.get(group_id)
            if cached and now - cached[0] < _REF_IMAGE_TTL:
                return cached[1]
        eid = client.find_enhanced_id_for_group(group_id)
        if eid:
            r = client.get(f"{BASE_PROTECT}/thumbnails/enhanced/{eid}")
        else:
            r = client.get(f"{BASE_PROTECT}/recognition/face/groups/{group_id}/image")
        if r.status_code != 200:
            return None
        with _ref_image_lock:
            _ref_image_cache[group_id] = (time.time(), r.content)
        return r.content

    def invalidate_ref_image(group_id: str) -> None:
        with _ref_image_lock:
            _ref_image_cache.pop(group_id, None)
            for key in [k for k in _ref_images_cache if k[0] == group_id]:
                _ref_images_cache.pop(key, None)

    def get_reference_images(group_id: str, k: int) -> list[bytes]:
        """Up to k face crops for a group, for multi-sample local matching.

        The cover image plus the k-best enhanced detection crops (ranked by
        matchedGroupConfidence) — a small gallery that's far more robust than a
        single reference photo. Cached per (group, k)."""
        if k <= 1:
            img = get_reference_image(group_id)
            return [img] if img else []
        now = time.time()
        with _ref_image_lock:
            cached = _ref_images_cache.get((group_id, k))
            if cached and now - cached[0] < _REF_IMAGE_TTL:
                return cached[1]

        imgs: list[bytes] = []
        cover = get_reference_image(group_id)
        if cover:
            imgs.append(cover)
        try:
            dets: list[dict] = []
            for page in range(1, 3):
                r = client.get(
                    f"{BASE_PROTECT}/recognition/face/groups/{group_id}/detections",
                    params={"page": page, "pageSize": 100},
                )
                if r.status_code != 200:
                    break
                batch = r.json().get("detections", [])
                if not batch:
                    break
                dets.extend(batch)
                if len(batch) < 100:
                    break
            enhanced = [d for d in dets if d.get("enhancedImageId")]
            enhanced.sort(key=lambda d: d.get("matchedGroupConfidence") or 0, reverse=True)
            # Precision over variety: take the HIGHEST-confidence crops. Low
            # matchedGroupConfidence detections are the ones Protect most often
            # misattributes to the wrong person, and a single impostor in the
            # gallery causes confident cross-person matches. An optional floor
            # (LOCAL_GALLERY_MIN_CONF) hardens this further. Good query alignment
            # (SCRFD) handles off-angle queries, so we don't need off-angle
            # gallery samples to get recall.
            floor = _gallery_min_conf()
            if floor > 0:
                enhanced = [d for d in enhanced
                            if (d.get("matchedGroupConfidence") or 0) >= floor]
            candidates = enhanced[: k * 2]  # over-fetch a little; some may fail
            with ThreadPoolExecutor(max_workers=8) as ex:
                fetched = list(ex.map(
                    lambda d: client.get(
                        f"{BASE_PROTECT}/thumbnails/enhanced/{d['enhancedImageId']}"),
                    candidates,
                ))
            for resp in fetched:
                if len(imgs) >= k:
                    break
                if resp.status_code == 200 and resp.content:
                    imgs.append(resp.content)
        except Exception:
            pass

        imgs = imgs[:k]
        with _ref_image_lock:
            _ref_images_cache[(group_id, k)] = (time.time(), imgs)
        return imgs

    # How many sample crops per face to use for local (gallery) matching. 1
    # restores single-photo behaviour. Only the local backend uses galleries;
    # Gemini stays single-image to keep its per-call cost bounded.
    def _gallery_samples() -> int:
        if ai_kind != "local":
            return 1
        try:
            return max(1, int(os.getenv("AI_GALLERY_SAMPLES", "5") or 5))
        except ValueError:
            return 5

    @app.get("/")
    def root() -> FileResponse:
        index = STATIC_DIR / "index.html"
        if not index.exists():
            raise HTTPException(500, "frontend missing")
        return FileResponse(index)

    @app.get("/favicon.svg", include_in_schema=False)
    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        """Serve the app icon. Both paths return the same SVG — modern
        browsers honour SVG favicons, and routing /favicon.ico here keeps
        the logs clean of 404s from browsers that probe it by default."""
        return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/version")
    def version() -> dict:
        return {"version": __version__, "buildDate": __build_date__}

    @app.get("/api/identities")
    def list_identities() -> list[dict]:
        groups = client.list_face_groups()
        named = [group_summary(g) for g in groups if is_named_identity(g)]
        named.sort(key=lambda g: g.get("lastDetectedAt") or 0, reverse=True)
        return named

    @app.get("/api/unenrolled")
    def list_unenrolled(
        include_degraded: bool = Query(True),
        min_detections: int = Query(1, ge=0),
        offset: int = Query(0, ge=0),
        limit: int = Query(60, ge=1, le=500),
    ) -> dict:
        """Every unnamed face cluster, sorted by most-recent first.

        Two listings make up the set, because Protect splits them:
        /recognition/face/groups returns known + unknown clusters but no
        degraded ones, and `?labels=groupType:degraded` returns those.

        `include_degraded=true` (default) keeps Protect's "low quality" face
        clusters. Set false to hide single-detection junk crops — which now
        also skips the second (3-page) fetch entirely.
        """
        groups = client.list_face_groups()
        if include_degraded:
            groups = groups + client.list_degraded_face_groups()

        unnamed = []
        for g in groups:
            if is_named_identity(g):
                continue
            if g.get("isDegraded") and not include_degraded:
                continue
            if (g.get("detectionsCount") or 0) < min_detections:
                continue
            unnamed.append(group_summary(g))

        unnamed.sort(key=lambda g: g.get("lastDetectedAt") or 0, reverse=True)
        total = len(unnamed)
        page = unnamed[offset:offset + limit]
        return {"total": total, "offset": offset, "limit": limit, "items": page}

    @app.get("/api/groups/{group_id}/avatar")
    def avatar(group_id: str, enhanced: bool = Query(False)) -> Response:
        """Group cover image — the SAME reference photo Protect shows in its
        own UI, so editing a person's photo there (or via 🎯 Best avatar here)
        is mirrored 1:1. Pass ?enhanced=true to instead serve a sharpened face
        crop picked from the group's enhanced detections.

        Cache-Control is deliberately short and NOT `immutable`: the avatar is
        keyed by a mutable group id, so a cover photo changed anywhere outside
        this app must be allowed to propagate. (Detection thumbnails below are
        content-addressed and stay immutable.)"""
        if enhanced:
            eid = client.find_enhanced_id_for_group(group_id)
            if eid:
                r = client.get(f"{BASE_PROTECT}/thumbnails/enhanced/{eid}")
                if r.status_code == 200:
                    return Response(
                        content=r.content,
                        media_type=r.headers.get("content-type", "image/jpeg"),
                        headers={"Cache-Control": "private, max-age=60"},
                    )
        # Default: the raw group reference image (Protect's cover photo).
        r = client.get(f"{BASE_PROTECT}/recognition/face/groups/{group_id}/image")
        if r.status_code != 200:
            raise HTTPException(r.status_code, "avatar fetch failed")
        return Response(
            content=r.content,
            media_type=r.headers.get("content-type", "image/jpeg"),
            headers={"Cache-Control": "private, max-age=60"},
        )

    @app.get("/api/groups/{group_id}/detections")
    def detections(group_id: str, page: int = 1, pageSize: int = 20) -> Any:
        r = client.get(
            f"{BASE_PROTECT}/recognition/face/groups/{group_id}/detections",
            params={"page": page, "pageSize": pageSize},
        )
        if r.status_code != 200:
            raise HTTPException(r.status_code, r.text[:300])
        return JSONResponse(r.json())

    @app.post("/api/merge")
    def merge(body: MergeBody) -> dict:
        try:
            result = client.merge_groups(body.fromGroupIds, body.toGroupId)
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        client.invalidate_groups_cache()
        for gid in body.fromGroupIds + [body.toGroupId]:
            invalidate_ref_image(gid)
        return {"ok": True, "merged": len(body.fromGroupIds), "result": result}

    @app.patch("/api/identities/{group_id}")
    def rename_identity(group_id: str, body: RenameBody) -> dict:
        """Rename an existing identity, OR convert an unnamed face_NNNN cluster
        into a named identity (Protect treats both as the same operation)."""
        try:
            result = client.rename_group(group_id, body.name)
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        client.invalidate_groups_cache()
        invalidate_ref_image(group_id)
        return group_summary(result)

    @app.post("/api/groups/delete")
    def delete_groups(body: DeleteBody) -> dict:
        """Bulk-delete face groups (the cluster + its detections in Protect)."""
        deleted, failed = 0, []
        for gid in body.ids:
            try:
                client.delete_group(gid)
                deleted += 1
                invalidate_ref_image(gid)
            except Exception as e:
                failed.append({"id": gid, "error": str(e)[:200]})
        client.invalidate_groups_cache()
        return {"ok": True, "deleted": deleted, "failed": failed}

    @app.post("/api/retroactive/start")
    def retroactive_start(body: RetroactiveBody) -> dict:
        try:
            result = client.start_retroactive(
                body.numberOfEvents, body.allCameras, body.cameraIds
            )
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "result": result}

    @app.post("/api/retroactive/cancel")
    def retroactive_cancel() -> dict:
        client.cancel_retroactive()
        return {"ok": True}

    @app.get("/api/retroactive/status")
    def retroactive_status() -> dict:
        ai = client.get_aiprocessor()
        if not ai:
            return {"state": "unknown"}
        rp = ai.get("retroactiveProcessing") or {}
        stats = ai.get("taskStatistics") or {}
        return {
            "state": rp.get("state"),
            "startedAt": rp.get("startedAt"),
            "estimatedTimeOfCompletion": rp.get("estimatedTimeOfCompletion"),
            "numberOfEvents": rp.get("numberOfEvents"),
            "tasksInQueue": stats.get("tasksInQueue"),
            "recentProcessedFaceEnhanceTasks": stats.get("recentProcessedFaceEnhanceTasks"),
        }

    # === Enhancer status (the background loop) ============================
    @app.get("/api/enhancer/status")
    def enhancer_status() -> dict:
        snap = STATUS.snapshot()
        # Splice in the AI Key queue for a one-stop status view.
        ai = client.get_aiprocessor()
        if ai:
            stats = ai.get("taskStatistics") or {}
            rp = ai.get("retroactiveProcessing") or {}
            snap["aiKey"] = {
                "tasksInQueue": stats.get("tasksInQueue"),
                "recentProcessedFaceEnhanceTasks": stats.get("recentProcessedFaceEnhanceTasks"),
                "recentProcessedRAMTasks": stats.get("recentProcessedRAMTasks"),
                "recentProcessedSTTTasks": stats.get("recentProcessedSTTTasks"),
                "retroactiveState": rp.get("state"),
                "retroactiveEta": rp.get("estimatedTimeOfCompletion"),
            }
        return snap

    @app.post("/api/enhancer/run-now")
    def enhancer_run_now() -> dict:
        STATUS.request_run_now()
        return {"ok": True}

    # === Per-identity detection management ================================
    @app.get("/api/identities/{group_id}/detections")
    def identity_detections(group_id: str, page: int = 1, pageSize: int = 50) -> Any:
        r = client.get(
            f"{BASE_PROTECT}/recognition/face/groups/{group_id}/detections",
            params={"page": page, "pageSize": pageSize},
        )
        if r.status_code != 200:
            raise HTTPException(r.status_code, r.text[:300])
        return JSONResponse(r.json())

    @app.get("/api/thumbnails/{thumb_id}")
    def thumbnail(thumb_id: str, enhanced: bool = Query(False)) -> Response:
        """Proxy face-crop thumbnails.

        - `?enhanced=false` (default) fetches the raw face crop via
          /thumbnails/{id}. Use this with `detection.thumbnailId`.
        - `?enhanced=true` fetches the enhanced version via
          /thumbnails/enhanced/{id}. Use this with `detection.enhancedImageId`.
        """
        path = f"{BASE_PROTECT}/thumbnails/{'enhanced/' if enhanced else ''}{thumb_id}"
        r = client.get(path)
        if r.status_code != 200:
            raise HTTPException(r.status_code, "thumbnail fetch failed")
        return Response(
            content=r.content,
            media_type=r.headers.get("content-type", "image/jpeg"),
            headers={"Cache-Control": "private, max-age=3600, immutable"},
        )

    @app.get("/api/identities/{group_id}/best-avatar-candidates")
    def best_avatar_candidates(group_id: str, max_candidates: int = Query(8, ge=1, le=20)) -> dict:
        """Rank an identity's detections and return the top candidates for use
        as the cluster reference image. Sorts by matchedGroupConfidence desc
        with enhanced images preferred. The frontend shows these as a picker."""
        # Walk a few pages — most identities have plenty of detections; we want
        # quality not quantity.
        all_dets: list[dict] = []
        for page in range(1, 4):
            r = client.get(
                f"{BASE_PROTECT}/recognition/face/groups/{group_id}/detections",
                params={"page": page, "pageSize": 100},
            )
            if r.status_code != 200:
                break
            batch = r.json().get("detections", [])
            if not batch:
                break
            all_dets.extend(batch)
            if len(batch) < 100:
                break

        if not all_dets:
            raise HTTPException(404, "no detections for this identity")

        # Rank: (has_enhanced, matchedGroupConfidence, recency)
        def score(d: dict) -> tuple:
            return (
                1 if d.get("enhancedImageId") else 0,
                d.get("matchedGroupConfidence") or 0,
                d.get("detectedAt") or 0,
            )

        all_dets.sort(key=score, reverse=True)
        top = all_dets[:max_candidates]

        return {
            "candidates": [
                {
                    "detectionId": d["id"],
                    "enhancedImageId": d.get("enhancedImageId"),
                    "thumbnailId": d.get("thumbnailId"),
                    "matchedGroupConfidence": d.get("matchedGroupConfidence"),
                    "detectedAt": d.get("detectedAt"),
                }
                for d in top
            ],
            "totalChecked": len(all_dets),
        }

    @app.post("/api/identities/{group_id}/best-avatar")
    def apply_best_avatar(group_id: str, body: BestAvatarApplyBody) -> dict:
        """Replace the identity's reference image with the chosen enhanced crop."""
        r = client.get(f"{BASE_PROTECT}/thumbnails/enhanced/{body.enhancedImageId}")
        if r.status_code != 200:
            raise HTTPException(404, "could not fetch chosen image")
        try:
            client.upload_group_image(group_id, r.content)
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        client.invalidate_groups_cache()
        invalidate_ref_image(group_id)
        return {"ok": True, "enhancedImageId": body.enhancedImageId}

    def _ai_pick_best(group_id: str, max_candidates: int,
                      only_eids: list[str] | None = None) -> dict:
        """Internal helper. Returns {enhancedImageId, detectionId, reason,
        candidatesConsidered}. Does NOT upload — caller decides.

        When `only_eids` is provided, the pick is restricted to those exact
        enhancedImageIds. This is what the frontend passes so the picked tile
        is guaranteed to be one of the displayed candidates (avoids the bug
        where AI picked an id not in the visible grid → no highlight).
        """
        if ai is None:
            raise HTTPException(503, _AI_UNCONFIGURED)

        all_dets: list[dict] = []
        for page in range(1, 4):
            r = client.get(
                f"{BASE_PROTECT}/recognition/face/groups/{group_id}/detections",
                params={"page": page, "pageSize": 100},
            )
            if r.status_code != 200:
                break
            batch = r.json().get("detections", [])
            if not batch:
                break
            all_dets.extend(batch)
            if len(batch) < 100:
                break
        if not all_dets:
            raise HTTPException(404, "no detections for this identity")

        enhanced = [d for d in all_dets if d.get("enhancedImageId")]
        if only_eids:
            eid_set = set(only_eids)
            enhanced = [d for d in enhanced if d.get("enhancedImageId") in eid_set]
        if not enhanced:
            raise HTTPException(400, "no enhanced detections yet — run the enhancer first")
        enhanced.sort(key=lambda d: d.get("matchedGroupConfidence") or 0, reverse=True)
        top = enhanced[:max_candidates]

        def fetch_one(d: dict) -> bytes | None:
            r = client.get(f"{BASE_PROTECT}/thumbnails/enhanced/{d['enhancedImageId']}")
            return r.content if r.status_code == 200 else None

        with ThreadPoolExecutor(max_workers=16) as ex:
            results = list(ex.map(fetch_one, top))
        pairs = [(d, img) for d, img in zip(top, results) if img]
        if not pairs:
            raise HTTPException(500, "could not fetch any candidate images")

        try:
            choice = ai.pick_best_avatar([img for _, img in pairs])
        except Exception as e:
            raise HTTPException(500, f"AI call failed: {type(e).__name__}: {e}")

        chosen_det, _ = pairs[choice["index"]]
        return {
            "enhancedImageId": chosen_det["enhancedImageId"],
            "detectionId": chosen_det["id"],
            "candidatesConsidered": len(pairs),
            "reason": choice["reason"],
        }

    @app.get("/api/identities/{group_id}/best-avatar-ai-suggest")
    def suggest_best_avatar_ai(
        group_id: str,
        max_candidates: int = Query(12, ge=2, le=20),
        eids: str = Query("", description="Comma-separated enhancedImageIds "
                                          "the AI must pick from. The UI sends "
                                          "the displayed tiles so the picked "
                                          "id always matches a visible tile."),
    ) -> dict:
        """Same picker as the apply endpoint but returns the choice WITHOUT
        uploading. Lets the UI preview before the user commits."""
        only_eids = [e for e in (eids or "").split(",") if e] or None
        return _ai_pick_best(group_id, max_candidates, only_eids=only_eids)

    @app.post("/api/identities/{group_id}/best-avatar-ai")
    def apply_best_avatar_ai(group_id: str, max_candidates: int = Query(12, ge=2, le=20)) -> dict:
        """Pick + upload in one call. Kept for back-compat / power use."""
        choice = _ai_pick_best(group_id, max_candidates)
        r = client.get(f"{BASE_PROTECT}/thumbnails/enhanced/{choice['enhancedImageId']}")
        if r.status_code != 200:
            raise HTTPException(404, "could not fetch chosen image")
        try:
            client.upload_group_image(group_id, r.content)
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        client.invalidate_groups_cache()
        invalidate_ref_image(group_id)
        return {"ok": True, **choice}

    @app.post("/api/detections/reassign")
    def reassign_detections(body: ReassignBody) -> dict:
        """Move detections to another identity, or pass groupId=null to
        unassign them entirely (Protect will re-cluster)."""
        api_body: dict = {"objectIds": body.detectionIds, "groupId": body.groupId}
        r = client.post(
            f"{BASE_PROTECT}/recognition/face/assign-group",
            json=api_body,
        )
        if r.status_code >= 400:
            raise HTTPException(400, f"reassign failed HTTP {r.status_code}: {r.text[:300]}")
        client.invalidate_groups_cache()
        return {
            "ok": True,
            "moved": len(body.detectionIds),
            "to": body.groupId,
        }

    # === AI suggestion (local OpenVINO or Gemini) ========================
    @app.get("/api/ai/status")
    def ai_status(bench: bool = False) -> dict:
        sdk_installed = (
            gemini_available() if ai_kind == "gemini" else local_client.is_available()
        )
        out = {
            "available": ai is not None,
            "provider": ai_kind,
            "sdk_installed": sdk_installed,
            "model": getattr(ai, "_model", None) if ai is not None else None,
            # Max faces a single Auto-match run processes, so the UI can warn +
            # cap before spending the user's cloud quota (harmless cap locally).
            "batchMax": _ai_batch_max(),
        }
        # ?bench=true runs a quick on-device inference benchmark (local backend
        # only) so you can compare CPU vs GPU; result is echoed in info().
        if bench and ai is not None and hasattr(ai, "benchmark"):
            try:
                ai.benchmark()
            except Exception as e:
                out["benchError"] = f"{type(e).__name__}: {e}"
        if ai is not None and hasattr(ai, "info"):
            out.update(ai.info())
        return out

    @app.get("/api/ai/suggest")
    def ai_suggest(groupId: str) -> dict:
        """Suggest which known identity (if any) matches an unnamed group."""
        if ai is None:
            raise HTTPException(503, _AI_UNCONFIGURED)

        groups = client.list_face_groups()
        identities_raw = [g for g in groups if is_named_identity(g)]
        if not identities_raw:
            raise HTTPException(400, "no known identities to compare against")

        # Pull every gallery in parallel — this is the dominant latency. k=1 for
        # Gemini (single photo); several for local (multi-sample matching).
        k = _gallery_samples()
        with ThreadPoolExecutor(max_workers=16) as ex:
            fut_q = ex.submit(get_reference_images, groupId, k)
            futs = {ex.submit(get_reference_images, g["id"], k): g for g in identities_raw}
            query_imgs = fut_q.result()
            identities: list[dict] = []
            for fut, g in futs.items():
                imgs = fut.result()
                if imgs:
                    identities.append({
                        "id": g["id"],
                        "name": g.get("name") or g.get("matchedName") or g["id"],
                        "image": imgs[0],   # single photo for Gemini
                        "images": imgs,     # gallery for local
                    })

        if not query_imgs:
            raise HTTPException(404, "could not fetch query face")

        try:
            if ai_kind == "local":
                result = ai.suggest(query_imgs[0], identities, cache_key=groupId,
                                    query_images=query_imgs)
            else:
                result = ai.suggest(query_imgs[0], identities, cache_key=groupId)
        except Exception as e:
            raise HTTPException(500, f"AI call failed: {type(e).__name__}: {e}")
        return result

    @app.post("/api/ai/suggest-batch")
    def ai_suggest_batch(body: BatchSuggestBody) -> dict:
        """Match MANY unnamed groups against the known identities in one shot.

        This is what powers the fast "Auto-match" flow: instead of the
        one-card-at-a-time Y/N walk, we fetch every identity reference image
        ONCE, then fan the per-face match calls out across a thread pool so
        the wall-clock time is roughly one call deep instead of N calls deep.
        Returns a per-group result the UI renders as a bulk review list — the
        user still confirms before anything merges.
        """
        if ai is None:
            raise HTTPException(503, _AI_UNCONFIGURED)

        # Cap the run so one click can't fire hundreds of billable calls.
        requested = len(body.groupIds)
        cap = _ai_batch_max()
        group_ids = body.groupIds[:cap] if cap else body.groupIds

        groups = client.list_face_groups()
        identities_raw = [g for g in groups if is_named_identity(g)]
        if not identities_raw:
            raise HTTPException(400, "no known identities to compare against")

        # Fetch the identity galleries once (cached across the whole batch).
        k = _gallery_samples()
        with ThreadPoolExecutor(max_workers=16) as ex:
            sample_lists = list(ex.map(
                lambda g: get_reference_images(g["id"], k), identities_raw))
        identities = [
            {
                "id": g["id"],
                "name": g.get("name") or g.get("matchedName") or g["id"],
                "image": imgs[0],
                "images": imgs,
            }
            for g, imgs in zip(identities_raw, sample_lists) if imgs
        ]
        if not identities:
            raise HTTPException(404, "could not fetch any identity reference images")

        def work(gid: str) -> dict:
            try:
                q_imgs = get_reference_images(gid, k)
                if not q_imgs:
                    return {"groupId": gid, "error": "no query image"}
                if ai_kind == "local":
                    res = ai.suggest(q_imgs[0], identities, cache_key=gid,
                                     query_images=q_imgs)
                else:
                    res = ai.suggest(q_imgs[0], identities, cache_key=gid)
                return {"groupId": gid, **res}
            except Exception as e:
                return {"groupId": gid, "error": f"{type(e).__name__}: {e}"}

        workers = max(1, int(os.getenv("AI_BATCH_WORKERS", "6") or 6))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(work, group_ids))
        return {
            "results": results,
            "identitiesConsidered": len(identities),
            "requested": requested,
            "processed": len(group_ids),
            "capped": len(group_ids) < requested,
        }

    return app


def run_webapp(client: ProtectClient, host: str, port: int) -> None:
    import uvicorn
    app = create_app(client)
    uvicorn.run(app, host=host, port=port, log_level=os.getenv("WEBAPP_LOG_LEVEL", "info"))
