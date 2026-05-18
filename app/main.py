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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
import urllib3

from .enhancer_status import STATUS
from .version import __build_date__, __version__

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
    """Fetch all face groups (people), paginating through every page.

    The default page size is 50 and the API silently caps responses. Identity
    groups (Face Identities in Protect 7.1+) and degraded auto-clusters often
    fall outside the first page, so we paginate explicitly.
    """
    all_groups = []
    page = 1
    page_size = 200

    while True:
        url = f"https://{host}{BASE_PROTECT}/recognition/face/groups"
        params = {"page": page, "pageSize": page_size}
        resp = session.get(url, params=params, verify=False)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict) and "groups" in data:
            batch = data["groups"]
            has_next = data.get("links", {}).get("next") is not None
        elif isinstance(data, list):
            batch = data
            has_next = len(batch) == page_size
        else:
            break

        if not batch:
            break

        all_groups.extend(batch)

        if not has_next:
            break

        page += 1
        time.sleep(0.2)

    return all_groups


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


def run_cycle(host, username, password, base_delay, only_unenhanced, group_filter, limit, dry_run, fetch_workers):
    """One full pass: log in, list groups, enhance every pending detection."""
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    STATUS.state = "fetching"
    STATUS.last_started_at = time.time()
    STATUS.last_error = None
    STATUS.progress_done = 0
    STATUS.progress_total = 0
    STATUS.current_group = None

    print(f"[*] Logging in to {host}...")
    try:
        csrf = login(session, host, username, password)
        print(f"[+] Logged in (CSRF: {csrf[:16]}...)")
    except requests.exceptions.HTTPError as e:
        print(f"[!] Login failed: {e}")
        STATUS.state = "error"
        STATUS.last_error = f"login failed: {e}"
        return

    print("\n[*] Fetching face groups...")
    try:
        groups = get_face_groups(session, host)
    except requests.exceptions.HTTPError as e:
        print(f"[!] Failed to fetch groups: {e}")
        STATUS.state = "error"
        STATUS.last_error = f"fetch groups failed: {e}"
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

    workers = max(1, fetch_workers)
    print(f"\n[*] Fetching detections for {len(groups)} group(s) (workers={workers})...")
    all_detections = []

    def fetch_one(g):
        group_name = g.get("name") or g.get("matchedName") or "(unnamed)"
        try:
            detections = get_group_detections(session, host, g["id"])
            for d in detections:
                d["_group_name"] = group_name
            return group_name, detections, None
        except Exception as e:
            return group_name, [], e

    if workers == 1:
        for g in groups:
            group_name, detections, err = fetch_one(g)
            if err:
                print(f"    {group_name:<25} -> ERROR: {err}")
            else:
                all_detections.extend(detections)
                print(f"    {group_name:<25} -> {len(detections)} detection(s)")
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(fetch_one, g) for g in groups]
            done_count = 0
            for fut in as_completed(futures):
                group_name, detections, err = fut.result()
                done_count += 1
                if err:
                    print(f"    [{done_count:>4}/{len(groups)}] {group_name:<25} -> ERROR: {err}")
                else:
                    all_detections.extend(detections)
                    print(f"    [{done_count:>4}/{len(groups)}] {group_name:<25} -> {len(detections)} detection(s)")

    print(f"\n[+] Total detections: {len(all_detections)}")

    if only_unenhanced:
        before = len(all_detections)
        # Split by faceEnhanceState so we can act + report intelligently.
        # done: already enhanced, skip silently.
        # queued: AI Key has the task but hasn't completed it. If recent, give it time.
        #         If old (>1h) it's almost certainly stuck — upstream RAM step failed.
        #         Re-POSTing returns 500 either way, so don't bother.
        # failed/missing/anything else: try POSTing — the AI Key may accept it.
        now_ms = int(time.time() * 1000)
        stuck_threshold_ms = 60 * 60 * 1000  # 1 hour
        keep, stuck, fresh = [], [], []
        for d in all_detections:
            state = (d.get("metadata") or {}).get("faceEnhanceState")
            if d.get("enhancedImageId") or state == "done":
                continue
            if state == "queued":
                age = now_ms - (d.get("detectedAt") or now_ms)
                if age > stuck_threshold_ms:
                    stuck.append(d)
                    continue
            keep.append(d)
            if state in (None, "", "failed"):
                fresh.append(d)
        all_detections = keep
        done_count = before - len(keep) - len(stuck)
        print(f"[*] Already enhanced (done):  {done_count}")
        print(f"[*] Stuck (queued >1h, upstream failed):  {len(stuck)}")
        print(f"[*] To enhance this cycle:    {len(all_detections)}  ({len(fresh)} never tried)")

    if limit > 0:
        all_detections = all_detections[:limit]
        print(f"[*] Limited to {len(all_detections)} detection(s)")

    if not all_detections:
        print("\n[*] Nothing to enhance this cycle.")
        STATUS.last_summary = {"enhanced": 0, "skipped": 0, "stuck": 0, "failed": 0, "total": 0}
        STATUS.last_finished_at = time.time()
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
    queued_stuck = 0
    consecutive_errors = 0
    auth_failed = False
    current_queue_size = stats.get("tasksInQueue", 0) if stats else 0

    STATUS.state = "enhancing"
    STATUS.progress_total = len(all_detections)
    STATUS.progress_done = 0

    for i, detection in enumerate(all_detections):
        if auth_failed or _shutdown:
            break
        det_id = detection["id"]
        group_name = detection.get("_group_name", "?")
        ts_str = format_timestamp(detection.get("detectedAt"))
        STATUS.current_group = group_name
        STATUS.progress_done = i

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
                    # 409 = conflict (already enhanced).
                    # 500 = AI Key won't accept the task. Usually means it's already
                    # queued AND upstream RAM failed, so it'll never complete.
                    state = (detection.get("metadata") or {}).get("faceEnhanceState")
                    if status_code == 500 and state == "queued":
                        queued_stuck += 1
                        print(f"  [{i + 1:>4}/{len(all_detections)}] ! Stuck (queued, upstream RAM failed)  {group_name}  {det_id[:16]}...")
                    else:
                        skipped += 1
                        print(f"  [{i + 1:>4}/{len(all_detections)}] - Skipped (HTTP {status_code}, state={state})  {group_name}")
                    consecutive_errors = 0
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
    print(f"  Enhanced:    {success}")
    print(f"  Skipped:     {skipped}")
    print(f"  Stuck:       {queued_stuck}  (queued in AI Key, upstream RAM failed)")
    print(f"  Failed:      {failed}")
    print(f"  Total:       {len(all_detections)}")
    print(f"{'=' * 50}")

    STATUS.last_summary = {
        "enhanced": success,
        "skipped": skipped,
        "stuck": queued_stuck,
        "failed": failed,
        "total": len(all_detections),
    }
    STATUS.last_finished_at = time.time()
    STATUS.progress_done = STATUS.progress_total
    STATUS.current_group = None


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
    fetch_workers = env_int("FETCH_WORKERS", 8)
    enhancer_enabled = env_bool("ENHANCER_ENABLED", True)
    webapp_enabled = env_bool("WEBAPP_ENABLED", True)
    webapp_port = env_int("WEBAPP_PORT", 8080)
    webapp_host = env_str("WEBAPP_HOST", "0.0.0.0") or "0.0.0.0"

    print(f"[*] UniFi Protect Face Enhance v{__version__} ({__build_date__})")
    print(f"    host={host} user={username}")
    print(f"    enhancer={enhancer_enabled} (poll_interval={poll_interval}s, run_once={run_once}, dry_run={dry_run})")
    print(f"    webapp={webapp_enabled} (port={webapp_port})")
    print(f"    fetch_workers={fetch_workers}")
    if group_filter:
        print(f"    group_filter={group_filter!r}")

    def enhancer_loop():
        cycle = 0
        while not _shutdown:
            cycle += 1
            STATUS.cycle_number = cycle
            print(f"\n{'#' * 60}\n# Cycle {cycle}  {datetime.now().isoformat(timespec='seconds')}\n{'#' * 60}")
            try:
                run_cycle(host, username, password, base_delay,
                          only_unenhanced, group_filter, limit, dry_run, fetch_workers)
            except Exception as e:
                print(f"[!] Cycle crashed: {type(e).__name__}: {e}")
                STATUS.state = "error"
                STATUS.last_error = f"{type(e).__name__}: {e}"

            if run_once or _shutdown:
                break

            print(f"\n[*] Sleeping {poll_interval}s before next cycle...")
            STATUS.state = "sleeping"
            STATUS.sleep_until = time.time() + poll_interval
            slept = 0
            while slept < poll_interval and not _shutdown:
                if STATUS.consume_run_now():
                    print("[*] Manual 'Run now' received — waking up.")
                    break
                time.sleep(min(5, poll_interval - slept))
                slept += 5
            STATUS.sleep_until = None
        STATUS.state = "idle"

    if not webapp_enabled and not enhancer_enabled:
        print("[!] Both WEBAPP_ENABLED and ENHANCER_ENABLED are false — nothing to do.")
        sys.exit(1)

    enhancer_thread = None
    if enhancer_enabled:
        enhancer_thread = threading.Thread(target=enhancer_loop, name="enhancer", daemon=True)
        enhancer_thread.start()

    if webapp_enabled:
        from .protect_client import ProtectClient
        from .webapp import run_webapp
        client = ProtectClient(host, username, password)
        print(f"[*] Webapp serving on http://{webapp_host}:{webapp_port}")
        try:
            run_webapp(client, webapp_host, webapp_port)
        except KeyboardInterrupt:
            pass
    elif enhancer_thread is not None:
        # Webapp disabled — block until the enhancer thread is done.
        enhancer_thread.join()

    print("[*] Exiting.")


if __name__ == "__main__":
    main()
