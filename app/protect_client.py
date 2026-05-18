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
                 groups_cache_ttl: float = 15.0, pool_size: int = 50):
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
