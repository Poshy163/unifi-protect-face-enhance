"""Thread-safe HTTP client for the UniFi Protect private API.

Handles login, CSRF extraction, and automatic re-authentication on 401.
Used by the webapp; the long-running enhancer keeps its own session.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from typing import Any

import requests
import urllib3
from requests.adapters import HTTPAdapter

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_PROTECT = "/proxy/protect/api"


class ProtectClient:
    def __init__(self, host: str, username: str, password: str,
                 groups_cache_ttl: float = 5.0, pool_size: int = 50):
        self.host = host
        self.username = username
        self.password = password
        self.pool_size = pool_size
        self._session: requests.Session | None = None
        self._lock = threading.RLock()
        # Cached list_face_groups output. The webapp hits this constantly —
        # caching turns ~1-2s of pagination into a dict lookup.
        self._groups_cache_ttl = groups_cache_ttl
        self._groups_cache_at = 0.0
        self._groups_cache_data: list[dict] | None = None
        self._groups_cache_lock = threading.Lock()

    def _login_locked(self) -> None:
        session = requests.Session()
        session.headers.update({"Accept": "application/json"})
        # Bigger urllib3 pool so dozens of concurrent avatar fetches don't queue.
        adapter = HTTPAdapter(pool_connections=self.pool_size,
                              pool_maxsize=self.pool_size)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        url = f"https://{self.host}/api/auth/login"
        payload = {"username": self.username, "password": self.password, "rememberMe": True}
        resp = session.post(url, json=payload, verify=False, timeout=15)
        resp.raise_for_status()

        token = session.cookies.get("TOKEN")
        if token:
            parts = token.split(".")
            if len(parts) >= 2:
                pad = parts[1] + "=" * (4 - len(parts[1]) % 4)
                claims = json.loads(base64.b64decode(pad))
                csrf = claims.get("csrfToken")
                if csrf:
                    session.headers["x-csrf-token"] = csrf
        self._session = session

    def _ensure_session(self) -> requests.Session:
        if self._session is None:
            with self._lock:
                if self._session is None:
                    self._login_locked()
        assert self._session is not None
        return self._session

    def request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        """Send `method path` against the Protect API. Refreshes on 401."""
        if not path.startswith("http"):
            url = f"https://{self.host}{path}"
        else:
            url = path
        kwargs.setdefault("verify", False)
        kwargs.setdefault("timeout", 30)

        session = self._ensure_session()
        resp = session.request(method, url, **kwargs)
        if resp.status_code == 401:
            with self._lock:
                self._login_locked()
                session = self._session
            assert session is not None
            resp = session.request(method, url, **kwargs)
        return resp

    def get(self, path: str, **kwargs: Any) -> requests.Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> requests.Response:
        return self.request("POST", path, **kwargs)

    def patch(self, path: str, **kwargs: Any) -> requests.Response:
        return self.request("PATCH", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> requests.Response:
        return self.request("DELETE", path, **kwargs)

    def list_face_groups(self, page_size: int = 1000,
                         force_refresh: bool = False) -> list[dict]:
        """Paginate through every face group. TTL-cached so the webapp
        doesn't hammer the Protect API on every page load."""
        now = time.time()
        with self._groups_cache_lock:
            if (not force_refresh
                    and self._groups_cache_data is not None
                    and now - self._groups_cache_at < self._groups_cache_ttl):
                return self._groups_cache_data

        out: list[dict] = []
        page = 1
        while True:
            r = self.get(f"{BASE_PROTECT}/recognition/face/groups",
                         params={"page": page, "pageSize": page_size})
            r.raise_for_status()
            j = r.json()
            batch = j.get("groups", []) if isinstance(j, dict) else j
            if not batch:
                break
            out.extend(batch)
            has_next = isinstance(j, dict) and j.get("links", {}).get("next") is not None
            if not has_next:
                break
            page += 1
            time.sleep(0.05)

        with self._groups_cache_lock:
            self._groups_cache_data = out
            self._groups_cache_at = time.time()
        return out

    def invalidate_groups_cache(self) -> None:
        """Drop the cached groups list so the next call refetches."""
        with self._groups_cache_lock:
            self._groups_cache_data = None
            self._groups_cache_at = 0.0

    def invalidate_phantom_cache(self) -> None:
        """Drop the phantom-group cache so the next harvest re-walks events."""
        lock = getattr(self, "_phantom_lock", None)
        cache = getattr(self, "_phantom_cache", None)
        if lock is None or cache is None:
            return
        with lock:
            cache["at"] = 0.0
            cache["data"] = []

    def harvest_phantom_groups(self, window_hours: int = 168,
                               cache_ttl: float = 4.0) -> list[dict]:
        """Walk recent smartDetectZone events and return synthetic records
        for every face group referenced that isn't in /recognition/face/groups.

        Protect's listing endpoint only returns 'committed' face groups; new
        clusters and most face_degraded_* singletons don't appear there, even
        though their detections exist and can be fetched per-id. This bridges
        that gap so the UI shows every face Protect has seen.
        """
        cache_attr = "_phantom_cache"
        lock_attr = "_phantom_lock"
        if not hasattr(self, lock_attr):
            setattr(self, lock_attr, threading.Lock())
            setattr(self, cache_attr, {"at": 0.0, "data": []})
        lock = getattr(self, lock_attr)
        cache = getattr(self, cache_attr)
        now = time.time()
        with lock:
            if now - cache["at"] < cache_ttl:
                return cache["data"]

        listed = {g["id"] for g in self.list_face_groups()}
        end_ms = int(now * 1000)
        start_ms = end_ms - window_hours * 3600 * 1000

        # CRITICAL: Protect's /events endpoint defaults to ASCending order.
        # Without orderDirection=DESC, `limit=500` over a 7-day window grabs
        # the OLDEST 500 events and misses everything recent — meaning new
        # face clusters never appear in the triage UI. Verified against a
        # live Protect 7.1.55 instance: without DESC, harvest returned 0
        # phantoms because every event was 6+ days old.
        r = self.get(
            f"{BASE_PROTECT}/events",
            params={"start": start_ms, "end": end_ms,
                    "type": "smartDetectZone", "limit": 500,
                    "orderDirection": "DESC"},
            timeout=60,
        )
        if r.status_code != 200:
            return []

        # group_id -> {count, last_ts}
        seen: dict[str, dict] = {}
        for ev in r.json():
            ts = ev.get("start") or 0
            for t in ((ev.get("metadata") or {}).get("detectedThumbnails") or []):
                if t.get("type") != "face":
                    continue
                g = t.get("group") or {}
                gid = g.get("id")
                if not gid or gid in listed:
                    continue
                entry = seen.setdefault(gid, {"count": 0, "last_ts": 0,
                                              "first_ts": ts or 0, "matchedName": g.get("matchedName")})
                entry["count"] += 1
                if (ts or 0) > entry["last_ts"]:
                    entry["last_ts"] = ts or 0
                if (ts or 0) < entry["first_ts"] or entry["first_ts"] == 0:
                    entry["first_ts"] = ts or 0

        synthetic = []
        for gid, info in seen.items():
            synthetic.append({
                "id": gid,
                "name": None,
                "matchedName": info.get("matchedName") or "",
                "type": "face",
                "isDegraded": gid.startswith("face_degraded_"),
                "detectionsCount": info["count"],
                "firstDetectedAt": info["first_ts"] or None,
                "lastDetectedAt": info["last_ts"] or None,
                "metadata": {},
                "_phantom": True,
            })

        with lock:
            cache["at"] = time.time()
            cache["data"] = synthetic
        return synthetic

    def find_enhanced_id_for_group(self, group_id: str) -> str | None:
        """Return the first enhancedImageId in the group's detections, or None.

        Used to swap the group's reference avatar for an enhanced face crop.
        TTL-cached per group via _enhanced_id_cache.
        """
        cache_attr = "_enhanced_id_cache"
        lock_attr = "_enhanced_id_lock"
        if not hasattr(self, lock_attr):
            setattr(self, lock_attr, threading.Lock())
            setattr(self, cache_attr, {})  # group_id -> (timestamp, id-or-None)
        lock = getattr(self, lock_attr)
        cache = getattr(self, cache_attr)
        now = time.time()
        with lock:
            if group_id in cache:
                t, val = cache[group_id]
                if now - t < 600:  # 10 minutes
                    return val
        # Fetch outside the lock.
        try:
            r = self.get(
                f"{BASE_PROTECT}/recognition/face/groups/{group_id}/detections",
                params={"page": 1, "pageSize": 10},
            )
            if r.status_code == 200:
                dets = r.json().get("detections", [])
                eid = next((d.get("enhancedImageId") for d in dets if d.get("enhancedImageId")), None)
            else:
                eid = None
        except Exception:
            eid = None
        with lock:
            cache[group_id] = (time.time(), eid)
        return eid

    def merge_groups(self, from_group_ids: list[str], to_group_id: str) -> dict:
        """POST /recognition/v2/merge-group"""
        r = self.post(f"{BASE_PROTECT}/recognition/v2/merge-group",
                      json={"fromGroupIds": from_group_ids, "toGroupId": to_group_id})
        if r.status_code >= 400:
            raise RuntimeError(f"merge failed HTTP {r.status_code}: {r.text[:300]}")
        return r.json() if r.text else {}

    def delete_group(self, group_id: str) -> None:
        """DELETE /recognition/face/groups/{id} — removes the face group and
        its detections from Protect entirely."""
        r = self.delete(f"{BASE_PROTECT}/recognition/face/groups/{group_id}")
        if r.status_code >= 400 and r.status_code != 404:
            raise RuntimeError(f"delete failed HTTP {r.status_code}: {r.text[:300]}")

    def upload_group_image(self, group_id: str, image_bytes: bytes,
                           filename: str = "best.jpg",
                           mime_type: str = "image/jpeg") -> dict:
        """POST /recognition/face/groups/{id}/image (multipart) — sets the
        cluster's reference image. Used to replace a low-quality avatar with
        a clearer enhanced face crop."""
        files = {"file": (filename, image_bytes, mime_type)}
        # buildUrl auto-pulls cookies + csrf, but requests + Session does the
        # multipart encoding for us. We call .post directly without `json=` so
        # the multipart Content-Type is set correctly.
        url = f"https://{self.host}{BASE_PROTECT}/recognition/face/groups/{group_id}/image"
        session = self._ensure_session()
        r = session.post(url, files=files, verify=False, timeout=30)
        if r.status_code == 401:
            with self._lock:
                self._login_locked()
                session = self._session
            assert session is not None
            r = session.post(url, files=files, verify=False, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"avatar upload failed HTTP {r.status_code}: {r.text[:300]}")
        try:
            return r.json()
        except Exception:
            return {"ok": True}

    def rename_group(self, group_id: str, name: str) -> dict:
        """PATCH /recognition/face/groups/{id}  with {name}.

        Also serves as 'create identity' when applied to an unnamed face_NNNN
        group — the rename converts the auto-cluster into a named identity.
        """
        r = self.patch(f"{BASE_PROTECT}/recognition/face/groups/{group_id}",
                       json={"name": name, "matchedName": name})
        if r.status_code >= 400:
            raise RuntimeError(f"rename failed HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def start_retroactive(self, number_of_events: int, all_cameras: bool = True,
                          camera_ids: list[str] | None = None) -> dict:
        body = {
            "numberOfEvents": number_of_events,
            "allCameras": all_cameras,
            "cameraIds": camera_ids or [],
        }
        r = self.post(f"{BASE_PROTECT}/aiprocessors/retroactive-processing/start", json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"retroactive start failed HTTP {r.status_code}: {r.text[:300]}")
        return r.json() if r.text else {}

    def cancel_retroactive(self) -> None:
        self.post(f"{BASE_PROTECT}/aiprocessors/retroactive-processing/cancel")

    def get_aiprocessor(self) -> dict | None:
        r = self.get(f"{BASE_PROTECT}/aiprocessors")
        if r.status_code != 200:
            return None
        arr = r.json()
        return arr[0] if arr else None
