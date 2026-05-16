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

from .protect_client import BASE_PROTECT, ProtectClient

STATIC_DIR = Path(__file__).parent / "static"


def is_named_identity(g: dict) -> bool:
    """A group is a 'named identity' if it has a user-given name and isn't degraded.
    Auto-cluster groups have id like face_NNNN with name=None.
    Degraded groups have id starting face_degraded_.
    """
    if g.get("isDegraded"):
        return False
    if str(g.get("id", "")).startswith("face_"):
        return False
    return bool(g.get("name"))


def group_summary(g: dict) -> dict:
    return {
        "id": g.get("id"),
        "name": g.get("name"),
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


def create_app(client: ProtectClient) -> FastAPI:
    app = FastAPI(title="Face Triage", docs_url="/api/docs", redoc_url=None)

    @app.get("/")
    def root() -> FileResponse:
        index = STATIC_DIR / "index.html"
        if not index.exists():
            raise HTTPException(500, "frontend missing")
        return FileResponse(index)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/identities")
    def list_identities() -> list[dict]:
        groups = client.list_face_groups()
        named = [group_summary(g) for g in groups if is_named_identity(g)]
        named.sort(key=lambda g: g.get("lastDetectedAt") or 0, reverse=True)
        return named

    @app.get("/api/unenrolled")
    def list_unenrolled(
        include_degraded: bool = Query(False),
        min_detections: int = Query(1, ge=0),
        offset: int = Query(0, ge=0),
        limit: int = Query(60, ge=1, le=500),
    ) -> dict:
        """List unnamed face groups (auto-clusters), sorted by most-recent first."""
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
        unnamed.sort(key=lambda g: g.get("lastDetectedAt") or 0, reverse=True)
        total = len(unnamed)
        page = unnamed[offset:offset + limit]
        return {"total": total, "offset": offset, "limit": limit, "items": page}

    @app.get("/api/groups/{group_id}/avatar")
    def avatar(group_id: str) -> Response:
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
        return {"ok": True, "merged": len(body.fromGroupIds), "result": result}

    @app.patch("/api/identities/{group_id}")
    def rename_identity(group_id: str, body: RenameBody) -> dict:
        """Rename an existing identity, OR convert an unnamed face_NNNN cluster
        into a named identity (Protect treats both as the same operation)."""
        try:
            result = client.rename_group(group_id, body.name)
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return group_summary(result)

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

    return app


def run_webapp(client: ProtectClient, host: str, port: int) -> None:
    import uvicorn
    app = create_app(client)
    uvicorn.run(app, host=host, port=port, log_level=os.getenv("WEBAPP_LOG_LEVEL", "info"))
