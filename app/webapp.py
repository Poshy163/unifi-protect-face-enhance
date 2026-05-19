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

from .enhancer_status import STATUS
from .gemini_client import build_from_env as build_gemini, is_available as gemini_available
from .protect_client import BASE_PROTECT, ProtectClient
from .version import __build_date__, __version__

STATIC_DIR = Path(__file__).parent / "static"


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


def create_app(client: ProtectClient) -> FastAPI:
    app = FastAPI(title="Face Triage", docs_url="/api/docs", redoc_url=None)
    gemini = build_gemini()  # None if no GEMINI_API_KEY or SDK missing

    # In-memory cache for reference identity images used by Gemini suggest.
    # Saves 14+ HTTP round-trips per suggestion after the first call.
    _ref_image_cache: dict[str, tuple[float, bytes]] = {}
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

    @app.get("/")
    def root() -> FileResponse:
        index = STATIC_DIR / "index.html"
        if not index.exists():
            raise HTTPException(500, "frontend missing")
        return FileResponse(index)

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
        include_phantoms: bool = Query(True),
        min_detections: int = Query(1, ge=0),
        offset: int = Query(0, ge=0),
        limit: int = Query(60, ge=1, le=500),
    ) -> dict:
        """Every unnamed face cluster, sorted by most-recent first.

        - Pulls listed groups from /recognition/face/groups.
        - When `include_phantoms` is true (default), ALSO harvests group IDs
          referenced by recent face events but missing from the listing —
          that's where freshly-created clusters and the bulk of degraded
          singletons live. Protect's UI shows these; we missed them before.
        - `include_degraded=true` (default) keeps Protect's "low quality"
          face clusters. Set false to hide single-detection junk crops.
        """
        groups = client.list_face_groups()
        unnamed = []
        for g in groups:
            if is_named_identity(g):
                continue
            if g.get("isDegraded") and not include_degraded:
                continue
            if (g.get("detectionsCount") or 0) < min_detections:
                continue
            unnamed.append(group_summary(g))

        if include_phantoms:
            for g in client.harvest_phantom_groups():
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
    def avatar(group_id: str, enhanced: bool = Query(True)) -> Response:
        """Group cover image. By default we serve an *enhanced* face crop from
        one of the group's detections (much clearer than the raw avatar);
        pass ?enhanced=false to force the raw group image."""
        if enhanced:
            eid = client.find_enhanced_id_for_group(group_id)
            if eid:
                r = client.get(f"{BASE_PROTECT}/thumbnails/enhanced/{eid}")
                if r.status_code == 200:
                    return Response(
                        content=r.content,
                        media_type=r.headers.get("content-type", "image/jpeg"),
                        headers={"Cache-Control": "private, max-age=3600, immutable"},
                    )
        # Fall back to the raw group avatar.
        r = client.get(f"{BASE_PROTECT}/recognition/face/groups/{group_id}/image")
        if r.status_code != 200:
            raise HTTPException(r.status_code, "avatar fetch failed")
        return Response(
            content=r.content,
            media_type=r.headers.get("content-type", "image/jpeg"),
            headers={"Cache-Control": "private, max-age=3600, immutable"},
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
        client.invalidate_phantom_cache()
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
        client.invalidate_phantom_cache()
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
        client.invalidate_phantom_cache()
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

    def _gemini_pick_best(group_id: str, max_candidates: int) -> dict:
        """Internal helper. Returns {enhancedImageId, detectionId, reason,
        candidatesConsidered}. Does NOT upload — caller decides."""
        if gemini is None:
            raise HTTPException(503, "Gemini is not configured. Set GEMINI_API_KEY.")

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
            choice = gemini.pick_best_avatar([img for _, img in pairs])
        except Exception as e:
            raise HTTPException(500, f"Gemini call failed: {type(e).__name__}: {e}")

        chosen_det, _ = pairs[choice["index"]]
        return {
            "enhancedImageId": chosen_det["enhancedImageId"],
            "detectionId": chosen_det["id"],
            "candidatesConsidered": len(pairs),
            "reason": choice["reason"],
        }

    @app.get("/api/identities/{group_id}/best-avatar-ai-suggest")
    def suggest_best_avatar_ai(group_id: str, max_candidates: int = Query(12, ge=2, le=20)) -> dict:
        """Same picker as the apply endpoint but returns the choice WITHOUT
        uploading. Lets the UI preview before the user commits."""
        return _gemini_pick_best(group_id, max_candidates)

    @app.post("/api/identities/{group_id}/best-avatar-ai")
    def apply_best_avatar_ai(group_id: str, max_candidates: int = Query(12, ge=2, le=20)) -> dict:
        """Pick + upload in one call. Kept for back-compat / power use."""
        choice = _gemini_pick_best(group_id, max_candidates)
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
        client.invalidate_phantom_cache()
        return {
            "ok": True,
            "moved": len(body.detectionIds),
            "to": body.groupId,
        }

    # === AI suggestion (Gemini) ==========================================
    @app.get("/api/ai/status")
    def ai_status() -> dict:
        return {
            "available": gemini is not None,
            "sdk_installed": gemini_available(),
        }

    @app.get("/api/ai/suggest")
    def ai_suggest(groupId: str) -> dict:
        """Ask Gemini which known identity (if any) matches an unnamed group."""
        if gemini is None:
            raise HTTPException(503, "Gemini is not configured. Set GEMINI_API_KEY.")

        groups = client.list_face_groups()
        identities_raw = [g for g in groups if is_named_identity(g)]
        if not identities_raw:
            raise HTTPException(400, "no known identities to compare against")

        # Pull every image in parallel — this is the dominant latency.
        def fetch_query():
            return get_reference_image(groupId)
        with ThreadPoolExecutor(max_workers=16) as ex:
            fut_q = ex.submit(fetch_query)
            futs = {ex.submit(get_reference_image, g["id"]): g for g in identities_raw}
            query_image = fut_q.result()
            identities: list[dict] = []
            for fut, g in futs.items():
                img = fut.result()
                if img:
                    identities.append({
                        "id": g["id"],
                        "name": g.get("name") or g.get("matchedName") or g["id"],
                        "image": img,
                    })

        if not query_image:
            raise HTTPException(404, "could not fetch query face")

        try:
            result = gemini.suggest(query_image, identities, cache_key=groupId)
        except Exception as e:
            raise HTTPException(500, f"Gemini call failed: {type(e).__name__}: {e}")
        return result

    return app


def run_webapp(client: ProtectClient, host: str, port: int) -> None:
    import uvicorn
    app = create_app(client)
    uvicorn.run(app, host=host, port=port, log_level=os.getenv("WEBAPP_LOG_LEVEL", "info"))
