"""Thread-safe HTTP client for the UniFi Protect private API.

Handles login, CSRF extraction, and automatic re-authentication on 401.
Used by the webapp; the long-running enhancer keeps its own session.
"""
from __future__ import annotations

import base64
import json
import os
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
        # Cached list_face_groups output, keyed by the `labels` filter. The
        # webapp hits this constantly — caching turns ~1-2s of pagination into
        # a dict lookup.
        self._groups_cache_ttl = groups_cache_ttl
        # The degraded listing is ~2900 groups over 3 pages, so it gets a
        # longer TTL than the ~300-row default listing.
        try:
            self._labelled_cache_ttl = float(os.getenv("DEGRADED_CACHE_TTL", "45") or 45)
        except ValueError:
            self._labelled_cache_ttl = 45.0
        # labels-or-None -> (fetched_at, groups)
        self._groups_cache: dict[str | None, tuple[float, list[dict]]] = {}
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

    def list_face_groups(self, labels: str | None = None, page_size: int = 1000,
                         force_refresh: bool = False) -> list[dict]:
        """Paginate through every face group. TTL-cached so the webapp
        doesn't hammer the Protect API on every page load.

        `labels` filters server-side. Comma-joined values inside one param are
        ORed (`"groupType:known,groupType:degraded"`); the default listing
        (labels=None) returns known + unknown groups but NO degraded ones.

        Caller beware: this filter FAILS OPEN. A misspelled label returns
        HTTP 200 with the full unfiltered listing rather than an error, so
        never trust a 200 as proof the filter applied — see
        list_degraded_face_groups, which validates what came back.
        """
        cache_key = labels
        ttl = self._groups_cache_ttl if labels is None else self._labelled_cache_ttl
        now = time.time()
        with self._groups_cache_lock:
            hit = self._groups_cache.get(cache_key)
            if not force_refresh and hit is not None and now - hit[0] < ttl:
                return hit[1]

        params: dict[str, Any] = {"page": 1, "pageSize": page_size}
        if labels:
            params["labels"] = labels

        out: list[dict] = []
        page = 1
        while True:
            params["page"] = page
            r = self.get(f"{BASE_PROTECT}/recognition/face/groups", params=params)
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
            self._groups_cache[cache_key] = (time.time(), out)
        return out

    def invalidate_groups_cache(self) -> None:
        """Drop every cached groups listing so the next call refetches."""
        with self._groups_cache_lock:
            self._groups_cache.clear()

    def list_degraded_face_groups(self, force_refresh: bool = False) -> list[dict]:
        """Every 'degraded' (low-quality) face cluster, straight from the API.

        Protect's default groups listing omits degraded clusters entirely, so
        they used to be reconstructed by walking smartDetectZone events and
        synthesising records from the group ids found on face thumbnails. That
        is unnecessary: the listing endpoint accepts a `labels` filter, and
        `groupType:degraded` returns the complete set (~2900 here vs the ~830
        the event walk reached) as full-fidelity records — real
        detectionsCount, firstDetectedAt/lastDetectedAt, imagePath and, most
        importantly, an authoritative `isDegraded` instead of one inferred
        from the id prefix.

        The `labels` filter fails open, so validate rather than trust: if the
        server hands back rows that aren't degraded, the filter didn't apply
        and we must not pass them off as degraded.
        """
        groups = self.list_face_groups(labels="groupType:degraded",
                                       force_refresh=force_refresh)
        degraded = [g for g in groups if g.get("isDegraded")]
        if groups and not degraded:
            # Every row came back non-degraded: the filter was ignored and we
            # got the plain listing. Report it instead of silently returning [].
            raise RuntimeError(
                "labels=groupType:degraded was ignored by the controller "
                f"({len(groups)} non-degraded groups returned) — the filter "
                "may have been removed or renamed in this Protect version")
        return degraded

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
