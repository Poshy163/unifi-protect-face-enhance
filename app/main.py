"""
UniFi Protect - Auto Face Enhancement via AI Key

Automatically triggers face enhancement on all face detections using your
UniFi Protect AI Key. Polls the AI Key's task queue and adaptively throttles
to avoid overloading the 200-event queue.

Runs as a long-lived service: each cycle logs in, sweeps every face group
for unenhanced detections, submits enhance jobs, then sleeps for
POLL_INTERVAL seconds before doing it again. Configuration is read from
environment variables (see .env.example).

API endpoints used (undocumented/private):
    GET  /proxy/protect/api/recognition/face/groups
    GET  /proxy/protect/api/recognition/face/groups/{id}/detections?page=N&pageSize=N
    POST /proxy/protect/api/recognition/face/detections/{id}/image/enhance
    GET  /proxy/protect/api/bootstrap (for AI Key queue stats)

Tested on UniFi OS 5.1.11 / Protect 7.1.55 / UP-AI-KEY.
"""

import base64
import json
import os
import signal
import sys
import time
from datetime import datetime

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_PROTECT = "/proxy/protect/api"

_shutdown = False


def _handle_signal(signum, _frame):
    global _shutdown
    _shutdown = True
    print(f"\n[*] Received signal {signum}, shutting down after current cycle...")


def env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[!] Invalid int for {name}={raw!r}, using default {default}")
        return default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[!] Invalid float for {name}={raw!r}, using default {default}")
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def login(session: requests.Session, host: str, username: str, password: str) -> str:
    """Authenticate to UniFi OS and return CSRF token."""
    url = f"https://{host}/api/auth/login"
    payload = {"username": username, "password": password, "rememberMe": True}
    resp = session.post(url, json=payload, verify=False)
    resp.raise_for_status()

    csrf = None
    token_cookie = session.cookies.get("TOKEN")
    if token_cookie:
        parts = token_cookie.split(".")
        if len(parts) >= 2:
            payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
            payload_data = json.loads(base64.b64decode(payload_b64))
            csrf = payload_data.get("csrfToken")

    if csrf:
        session.headers.update({"x-csrf-token": csrf})
    else:
        print("[!] Warning: Could not extract CSRF token. POST requests may fail.")

    return csrf or ""


def get_face_groups(session: requests.Session, host: str) -> list:
    """Fetch all face groups (people)."""
    url = f"https://{host}{BASE_PROTECT}/recognition/face/groups"
    resp = session.get(url, verify=False)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "groups" in data:
        return data["groups"]
    return data


def get_group_detections(session: requests.Session, host: str, group_id: str) -> list:
    """Fetch all detections for a face group, paginating via page/pageSize."""
    all_detections = []
    page = 1
    page_size = 100

    while True:
        url = f"https://{host}{BASE_PROTECT}/recognition/face/groups/{group_id}/detections"
        params = {"page": page, "pageSize": page_size}
        resp = session.get(url, params=params, verify=False)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict) and "detections" in data:
            batch = data["detections"]
            has_next = data.get("links", {}).get("next") is not None
        elif isinstance(data, list):
            batch = data
            has_next = len(batch) == page_size
        else:
            break

        if not batch:
            break

        all_detections.extend(batch)

        if not has_next:
            break

        page += 1
        time.sleep(0.2)

    return all_detections


def enhance_face(session: requests.Session, host: str, detection_id: str) -> dict:
    """Trigger face enhancement for a specific detection."""
    url = f"https://{host}{BASE_PROTECT}/recognition/face/detections/{detection_id}/image/enhance"
    resp = session.post(url, verify=False, timeout=30)
    if resp.status_code >= 400:
        err = requests.exceptions.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        err.response = resp
        raise err
    return {"status": "ok", "http_code": resp.status_code}


def get_queue_stats(session: requests.Session, host: str) -> dict:
    """Get AI Key task queue statistics from the bootstrap endpoint."""
    try:
        url = f"https://{host}{BASE_PROTECT}/bootstrap"
        resp = session.get(url, verify=False, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for ai in data.get("aiprocessors", []):
            stats = ai.get("taskStatistics", {})
            if stats:
                return stats
    except Exception:
        pass
    return {}


def adaptive_delay(queue_size: int, base_delay: float) -> float:
    """Calculate delay based on current AI Key queue depth.

    The AI Key has a 200-event queue and processes ~1,000 events/hour.
    This function slows down submissions as the queue fills up to avoid
    dropping detections.

        Queue 0-10:    base_delay     (full speed)
        Queue 10-50:   base_delay * 2
        Queue 50-100:  base_delay * 5
        Queue 100-150: base_delay * 15
        Queue 150+:    30s pause
    """
    if queue_size >= 150:
        return 30.0
    elif queue_size >= 100:
        return base_delay * 15
    elif queue_size >= 50:
        return base_delay * 5
    elif queue_size >= 10:
        return base_delay * 2
    return base_delay


def format_timestamp(ts) -> str:
    """Convert millisecond epoch timestamp to readable string."""
    if ts and isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M")
    return str(ts) if ts else ""


def run_cycle(host, username, password, base_delay, only_unenhanced, group_filter, limit, dry_run):
    """One full pass: log in, list groups, enhance every pending detection."""
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    print(f"[*] Logging in to {host}...")
    try:
        csrf = login(session, host, username, password)
        print(f"[+] Logged in (CSRF: {csrf[:16]}...)")
    except requests.exceptions.HTTPError as e:
        print(f"[!] Login failed: {e}")
        return

    print("\n[*] Fetching face groups...")
    try:
        groups = get_face_groups(session, host)
    except requests.exceptions.HTTPError as e:
        print(f"[!] Failed to fetch groups: {e}")
        return

    print(f"[+] Found {len(groups)} face group(s):\n")
    print(f"    {'Name':<25} {'Detections':>10}  {'Enhanced':>10}")
    print(f"    {'-' * 25} {'-' * 10}  {'-' * 10}")
    for g in groups:
        name = g.get("name") or g.get("matchedName") or "(unnamed)"
        count = g.get("detectionsCount", "?")
        enhanced = "Yes" if g.get("enhancedPath") else "No"
        print(f"    {name:<25} {str(count):>10}  {enhanced:>10}")

    if group_filter:
        groups = [g for g in groups if
                  (g.get("name") or "").lower() == group_filter.lower() or
                  (g.get("matchedName") or "").lower() == group_filter.lower()]
        if not groups:
            print(f"\n[!] No group found matching '{group_filter}'")
            return
        print(f"\n[*] Filtered to group: {groups[0].get('name')}")

    print(f"\n[*] Fetching detections for {len(groups)} group(s)...")
    all_detections = []
    for g in groups:
        group_name = g.get("name") or g.get("matchedName") or "(unnamed)"
        group_id = g["id"]
        try:
            detections = get_group_detections(session, host, group_id)
            for d in detections:
                d["_group_name"] = group_name
            all_detections.extend(detections)
            print(f"    {group_name:<25} -> {len(detections)} detection(s)")
        except requests.exceptions.HTTPError as e:
            print(f"    {group_name:<25} -> ERROR: {e}")
        time.sleep(0.3)

    print(f"\n[+] Total detections: {len(all_detections)}")

    if only_unenhanced:
        before = len(all_detections)
        all_detections = [d for d in all_detections if not d.get("enhancedImageId")]
        print(f"[*] Unenhanced: {len(all_detections)} (skipped {before - len(all_detections)} already done)")

    if limit > 0:
        all_detections = all_detections[:limit]
        print(f"[*] Limited to {len(all_detections)} detection(s)")

    if not all_detections:
        print("\n[*] Nothing to enhance this cycle.")
        return

    if dry_run:
        enhanced_count = sum(1 for d in all_detections if d.get("enhancedImageId"))
        unenhanced_count = len(all_detections) - enhanced_count
        print(f"\n[DRY RUN] Would enhance {len(all_detections)} detection(s):")
        print(f"    Already enhanced: {enhanced_count}")
        print(f"    Needs enhancement: {unenhanced_count}\n")
        for i, d in enumerate(all_detections[:30]):
            det_id = d["id"]
            group_name = d.get("_group_name", "?")
            ts_str = format_timestamp(d.get("detectedAt"))
            status = "enhanced" if d.get("enhancedImageId") else "pending"
            confidence = d.get("matchedGroupConfidence", "?")
            print(f"    [{i + 1:>3}] {group_name:<20} {ts_str}  conf={confidence}  [{status}]  {det_id[:16]}...")
        if len(all_detections) > 30:
            print(f"    ... and {len(all_detections) - 30} more")
        return

    print(f"\n[*] Starting enhancement of {len(all_detections)} detection(s)...")
    print(f"    Base delay: {base_delay}s (adaptive throttling based on AI Key queue)")

    stats = get_queue_stats(session, host)
    if stats:
        q = stats.get("tasksInQueue", "?")
        fe = stats.get("recentProcessedFaceEnhanceTasks", "?")
        print(f"    Current queue: {q}/200 tasks, {fe} face enhances processed so far\n")

    success = 0
    failed = 0
    skipped = 0
    consecutive_errors = 0
    auth_failed = False
    current_queue_size = stats.get("tasksInQueue", 0) if stats else 0

    for i, detection in enumerate(all_detections):
        if auth_failed or _shutdown:
            break
        det_id = detection["id"]
        group_name = detection.get("_group_name", "?")
        ts_str = format_timestamp(detection.get("detectedAt"))

        max_retries = 3
        for attempt in range(max_retries):
            try:
                enhance_face(session, host, det_id)
                success += 1
                consecutive_errors = 0
                print(f"  [{i + 1:>4}/{len(all_detections)}] + {group_name:<20} {ts_str}  {det_id[:16]}...")
                break
            except Exception as e:
                etype = type(e).__name__
                has_resp = hasattr(e, 'response') and e.response is not None
                status_code = e.response.status_code if has_resp else None
                body = e.response.text[:200] if has_resp else ""

                if status_code in (401, 403):
                    failed += 1
                    auth_failed = True
                    print(f"\n[!] Auth failed (HTTP {status_code}). Token may have expired.")
                    break
                elif status_code == 429:
                    wait = 30 * (attempt + 1)
                    print(f"  [{i + 1:>4}] ~ Rate limited -- pausing {wait}s...")
                    time.sleep(wait)
                    continue
                elif status_code in (409, 500):
                    skipped += 1
                    consecutive_errors = 0
                    print(f"  [{i + 1:>4}/{len(all_detections)}] - Skipped (already enhanced or in queue)  {group_name}")
                    break
                elif isinstance(e, requests.exceptions.ConnectionError):
                    if attempt < max_retries - 1:
                        wait = 3 * (attempt + 1)
                        print(f"  [{i + 1:>4}] ~ Connection error, retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                    failed += 1
                    consecutive_errors += 1
                    print(f"  [{i + 1:>4}/{len(all_detections)}] x {etype}  {group_name}  {e}")
                    break
                else:
                    if attempt < max_retries - 1:
                        print(f"  [{i + 1:>4}] ~ {etype}: HTTP {status_code} -- retry {attempt + 1}...")
                        time.sleep(2 * (attempt + 1))
                        continue
                    failed += 1
                    consecutive_errors += 1
                    print(f"  [{i + 1:>4}/{len(all_detections)}] x {etype}: HTTP {status_code}  {group_name}  {body}")
                    break
        else:
            failed += 1
            consecutive_errors += 1
            print(f"  [{i + 1:>4}/{len(all_detections)}] x All retries failed  {group_name}")

        if consecutive_errors >= 10:
            print(f"\n[!] {consecutive_errors} consecutive errors -- pausing 30s...")
            time.sleep(30)
            consecutive_errors = 0

        if i < len(all_detections) - 1:
            if success > 0 and success % 20 == 0:
                stats = get_queue_stats(session, host)
                if stats:
                    current_queue_size = stats.get("tasksInQueue", 0)
                    fe = stats.get("recentProcessedFaceEnhanceTasks", "?")
                    delay = adaptive_delay(current_queue_size, base_delay)
                    print(f"         [queue: {current_queue_size}/200, processed: {fe}, delay: {delay:.2f}s]")
                else:
                    delay = base_delay
            else:
                delay = adaptive_delay(current_queue_size, base_delay)

            time.sleep(delay)

    stats = get_queue_stats(session, host)
    if stats:
        q = stats.get("tasksInQueue", "?")
        fe = stats.get("recentProcessedFaceEnhanceTasks", "?")
        print(f"\n  AI Key: {q}/200 in queue, {fe} face enhances processed total")

    print(f"\n{'=' * 50}")
    print(f"  Cycle done!")
    print(f"  Enhanced:  {success}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Total:     {len(all_detections)}")
    print(f"{'=' * 50}")


def main():
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    host = env_str("UNIFI_HOST")
    username = env_str("UNIFI_USERNAME")
    password = env_str("UNIFI_PASSWORD")

    if not host or not username or not password:
        print("[!] UNIFI_HOST, UNIFI_USERNAME, and UNIFI_PASSWORD are required.")
        print("    Set them via environment variables (see .env.example).")
        sys.exit(1)

    base_delay = env_float("BASE_DELAY", 0.35)
    only_unenhanced = env_bool("ONLY_UNENHANCED", True)
    group_filter = env_str("GROUP_FILTER") or None
    limit = env_int("LIMIT", 0)
    dry_run = env_bool("DRY_RUN", False)
    run_once = env_bool("RUN_ONCE", False)
    poll_interval = env_int("POLL_INTERVAL", 300)

    print(f"[*] UniFi Protect Face Enhance starting")
    print(f"    host={host} user={username}")
    print(f"    poll_interval={poll_interval}s  run_once={run_once}  dry_run={dry_run}")
    if group_filter:
        print(f"    group_filter={group_filter!r}")

    cycle = 0
    while not _shutdown:
        cycle += 1
        print(f"\n{'#' * 60}\n# Cycle {cycle}  {datetime.now().isoformat(timespec='seconds')}\n{'#' * 60}")
        try:
            run_cycle(host, username, password, base_delay,
                      only_unenhanced, group_filter, limit, dry_run)
        except Exception as e:
            print(f"[!] Cycle crashed: {type(e).__name__}: {e}")

        if run_once or _shutdown:
            break

        print(f"\n[*] Sleeping {poll_interval}s before next cycle...")
        slept = 0
        while slept < poll_interval and not _shutdown:
            time.sleep(min(5, poll_interval - slept))
            slept += 5

    print("[*] Exiting.")


if __name__ == "__main__":
    main()
