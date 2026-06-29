# UniFi Protect — Auto Face Enhance + Triage

Two tools in one Docker container:

1. **Enhancer** — a long-lived background loop that auto-triggers face
   enhancement on every face detection in your UniFi Protect system, with
   adaptive throttling against the AI Key's 200-event queue so nothing gets
   dropped.
2. **Face Triage webapp** — a browser UI on port `8080` that makes bulk
   identity management fast: keyboard-driven merging, drag-and-drop,
   per-identity detail view, AI-suggested matches via Google Gemini, and
   one-click replacement of an identity's reference photo with the clearest
   available face crop.

Tested on **UniFi OS 5.1.11** with **Protect 7.1.55** and the **UP-AI-KEY**.

The Protect endpoints used here are undocumented/private and may change
between firmware versions.

## Quick start (Docker Compose)

```bash
git clone https://github.com/Poshy163/unifi-protect-face-enhance.git
cd unifi-protect-face-enhance
cp .env.example .env
# edit .env with your UNIFI_HOST / UNIFI_USERNAME / UNIFI_PASSWORD
docker compose up -d
```

The image is published to GHCR as
`ghcr.io/poshy163/unifi-protect-face-enhance:latest` and pulled automatically
by the bundled `docker-compose.yml`. To build locally instead, swap the
`image:` line for `build: .`.

Once up, open **http://<host>:8080** for the Face Triage UI. Tail enhancer
logs with `docker compose logs -f face-enhance`.

### Create a local-only admin

Do **not** use your Ubiquiti SSO account. In the UniFi console:

> Settings → Admins & Users → Add Admin → Restrict to local access only

Use those credentials for `UNIFI_USERNAME` / `UNIFI_PASSWORD`.

## Face Triage webapp

A browser UI for managing the face-recognition graph that Protect builds.
Hits the same private API the Protect web UI uses, plus a few of our own
quality-of-life touches.

### Triage view (default)

The grid shows every **unnamed** face cluster (`face_NNNN` plus any anonymous
clusters), sorted most-recent first. The sidebar shows every **named identity**.
Avatars mirror Protect's own **reference/cover photo** for each face, so a
photo you set in the Protect UI (or via 🎯 Best avatar here) shows up 1:1.

Click the **"Face Triage" title** in the top-left at any time to jump back to
this view (it clears the current selection and scrolls to the top).

- **Click** card → select. **Shift-click** → range. **A** → select all. **Esc** → clear.
- **Click an identity** in the sidebar → merges the selection into it.
- **Drag a card** onto an identity → also merges (the whole card drags, not the image).
- **Press 1–9** → keyboard merge into the Nth identity.
- **N** → "New identity" dialog: renames the first selected card to a new
  name, then merges the rest into it.
- **Del** or **Backspace** → bulk-deletes selected face groups (cluster *and*
  its detections — permanent).
- **⚡ Auto-match** → the fast path. Classifies every loaded unnamed face
  against your named identities in one shot (in parallel) and shows a single
  **bulk review list**: each unnamed face next to its suggested identity,
  confidence, and reason. Matches at ≥85% confidence are pre-checked, so the
  common case is one click to merge dozens of faces at once. Untick any you
  disagree with first. Requires an AI backend (local or Gemini — see below).
  To match faces beyond the first page, scroll the grid to load more, then run it.
  <br>**Backend note:** with the **local** backend (`AI_PROVIDER=local`) this
  is free and on-device, so the `AI_BATCH_MAX` cap and cost concerns don't
  apply. With **Gemini**, every face is one cloud call that also sends all your
  reference photos, so it asks you to confirm and caps each run at
  `AI_BATCH_MAX` (default 50). Either way results are cached per session so
  re-runs don't recompute.
- **🤖 AI suggest** → the step-through alternative: walks unnamed cards one at
  a time, each shown with **Y** (merge) or **N** (skip). Pre-fetches the next
  card while you review the current. Requires an AI backend (local or Gemini).

### Per-identity detail view

Click an identity in the sidebar (with no cards selected). Shows every
detection in the cluster as a grid.

- **Click** a detection card → select. Standard shift/cmd modifiers.
- **🎯 Best avatar** → opens a picker showing the 8 highest-quality enhanced
  detections (ranked by `matchedGroupConfidence`). Click any one to replace
  the cluster's reference photo via Protect's multipart upload endpoint.
  Useful when Protect's auto-picked avatar is grainy or backlit.
- **Move…** → reassign selected detections to a different identity.
- **Unassign** (or **Del**) → remove selected detections from this identity.
  Protect will re-cluster them on its own.
- **Rename** → rename the identity.
- **Delete identity** → wipes the cluster and all its detections (permanent).

### Enhancer status pill

The pill in the top-right shows live state of the background enhancer:
`enhancing 23/47`, `idle · next in 11h`, etc. Click it for a detail modal
with the last-cycle summary, AI Key queue depth, retroactive-processing
status, and a **Run now** button that signals the enhancer to wake early.

### Retroactive processing

The **Retroactive…** button in the header calls
`POST /aiprocessors/retroactive-processing/start` on the AI Key. Use this
to unstick detections in `faceEnhanceState: queued` whose parent event's
upstream RAM step failed.

## Configuration

All config is read from environment variables — see
[.env.example](.env.example) for the full list.

### Required

| Variable | Purpose |
| --- | --- |
| `UNIFI_HOST` | Console IP/hostname (UDM, UNVR, Cloud Key, etc.) |
| `UNIFI_USERNAME` | Local admin username |
| `UNIFI_PASSWORD` | Local admin password |

### Enhancer loop

| Variable | Default | Purpose |
| --- | --- | --- |
| `ENHANCER_ENABLED` | `true` | Run the periodic enhance sweep |
| `POLL_INTERVAL` | `300` | Seconds to sleep between full sweeps |
| `BASE_DELAY` | `0.35` | Base seconds between enhance requests (adaptive) |
| `ONLY_UNENHANCED` | `true` | Skip detections that already have an enhanced image |
| `GROUP_FILTER` | *(empty)* | Restrict to one person by name |
| `LIMIT` | `0` | Cap per cycle (0 = unlimited) |
| `DRY_RUN` | `false` | List detections without enhancing |
| `RUN_ONCE` | `false` | One cycle then exit (good for cron) |
| `FETCH_WORKERS` | `8` | Parallel workers for fetching detections |
| `ENHANCE_PHANTOMS` | `true` | Also enhance "unknown"/"degraded" faces that Protect keeps out of the groups listing (see below). |

> **Unknown/degraded faces.** Protect's `/recognition/face/groups` listing only
> contains *committed* groups. "Unknown" and "degraded" faces never appear there
> — they exist only as references inside `smartDetectZone` events — so before
> `ENHANCE_PHANTOMS`, the enhancer never enhanced them (on a real instance that
> was ~75% of all faces). With it on (default), each cycle harvests those
> unlisted groups via the same sliding window as the triage view
> (`PHANTOM_WINDOWS` / `PHANTOM_EVENT_LIMIT`) and enhances them too. The first
> run can add a large backlog; the adaptive throttle paces it.

### Webapp

| Variable | Default | Purpose |
| --- | --- | --- |
| `WEBAPP_ENABLED` | `true` | Serve the Face Triage webapp |
| `WEBAPP_PORT` | `8080` | Port to bind (use the same value in compose port mapping) |
| `WEBAPP_HOST` | `0.0.0.0` | Bind interface |
| `PHANTOM_WINDOWS` | `4` | Sliding-window passes over `/events` to harvest faces missing from the groups listing. Each ≈ 1 day deeper. Bump for a deep one-off sweep of old faces. |
| `PHANTOM_EVENT_LIMIT` | `1000` | Events fetched per window (Protect caps around 1000). |
| `PHANTOM_CACHE_TTL` | `45` | Seconds to cache the harvested set. |

### AI suggest (optional)

The ⚡ Auto-match and 🤖 AI suggest matchers run on one of two backends, chosen
with `AI_PROVIDER`:

- **`local`** — on-device face recognition via OpenVINO + ArcFace embeddings
  (the same embedding-and-cosine-similarity approach Frigate, CompreFace, and
  Scrypted use). **Free, fast, private — no API key, no per-call cost.** Runs on
  your CPU by default, or the Intel iGPU with `OPENVINO_DEVICE=GPU`. The model
  auto-downloads on first use. Recommended.
- **`gemini`** — Google Gemini cloud VLM. Costs per call and the bill scales
  with how many identities you have (every call ships them all).
- **`auto`** (default) — uses Gemini when `GEMINI_API_KEY` is set, otherwise
  falls back to local, so existing cloud setups keep working unchanged.

| Variable | Default | Purpose |
| --- | --- | --- |
| `AI_PROVIDER` | `auto` | `local`, `gemini`, or `auto`. |
| `AI_BATCH_WORKERS` | `6` | Parallel workers for the ⚡ Auto-match batch matcher. |
| `AI_BATCH_MAX` | `50` | Cap on faces per ⚡ Auto-match run. Mainly protects the Gemini bill; harmless on local. `0` disables it. |

**Local backend** (`AI_PROVIDER=local`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENVINO_DEVICE` | `CPU` | `CPU`, `GPU` (Intel iGPU), or `AUTO`. 13th-gen "H" chips have no NPU, so CPU/iGPU are the targets. `GET /api/ai/status` lists what OpenVINO detects (`availableDevices`). |
| `LOCAL_FACE_PACK` | `buffalo_l` | Embedding model. `buffalo_l` (ResNet50, ~166 MB, most accurate) or `buffalo_s` (MobileFaceNet, ~13 MB, lighter). |
| `LOCAL_FACE_MODEL` | *(empty)* | Path to your own ArcFace `.onnx` / OpenVINO `.xml`, bypassing the auto-download. |
| `LOCAL_MODEL_DIR` | `~/.cache/unifi-protect-face` | Where downloaded models are cached. Mount a volume here to persist them (the bundled compose file does). |
| `LOCAL_SIM_UNKNOWN` | `0.30` | Cosine floor — below this a face is reported as no match. Raise to cut false matches. |
| `LOCAL_SIM_STRONG` | `0.55` | Cosine at/above which confidence maps to ~1.0 (so the UI pre-checks it). |

**Using the Intel iGPU (`OPENVINO_DEVICE=GPU`).** OpenVINO supports the Iris Xe
iGPU, but the container has to be able to reach the GPU. This works on a **Linux
Docker host** (the same path Frigate/Scrypted use) — *not* Docker Desktop on
Windows/macOS, whose WSL2 GPU isn't exposed as `/dev/dri`. On a Linux host:

1. The published image already bundles the Intel NEO OpenCL runtime
   (`intel-opencl-icd`). Building locally? It's on by default; `--build-arg
   INTEL_GPU=0` skips it.
2. Uncomment the `devices:` + `group_add:` block in
   [docker-compose.yml](docker-compose.yml) to pass `/dev/dri` in.
3. Set `RENDER_GID` in `.env` to your host's render group:
   `getent group render | cut -d: -f3`.
4. Set `OPENVINO_DEVICE=GPU`, recreate, and confirm `"GPU"` shows up in
   `GET /api/ai/status` → `availableDevices`.

This ArcFace model is small and requests are one face at a time, so on a strong
CPU the iGPU isn't always faster per call — its real benefit is offloading the
CPU and sustained batch throughput.

**Gemini backend** (`AI_PROVIDER=gemini`, or `auto` with a key set):

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | *(empty)* | Get one at https://aistudio.google.com/app/apikey |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Cheapest tier by default. `gemini-2.5-flash` is more accurate but ~3-5x pricier; `gemini-2.5-pro` is best/most expensive. |

## How throttling works

The AI Key has a 200-event task queue and processes roughly 1,000 events
per hour. The enhancer polls `/api/bootstrap` every 20 successful enhances
to read the current queue depth and scales the per-request delay:

| Queue depth | Delay multiplier |
| --- | --- |
| 0–10 | `BASE_DELAY × 1` |
| 10–50 | `BASE_DELAY × 2` |
| 50–100 | `BASE_DELAY × 5` |
| 100–150 | `BASE_DELAY × 15` |
| 150+ | 30s pause |

## Private API endpoints used

Discovered from the Protect SPA bundle (`/app-assets/protect/swai.js`).
All require session auth with CSRF and may change between Protect versions.

```
# Face clusters (the central model)
GET    /proxy/protect/api/recognition/face/groups?page=N&pageSize=N
GET    /proxy/protect/api/recognition/face/groups/{id}
GET    /proxy/protect/api/recognition/face/groups/{id}/image
GET    /proxy/protect/api/recognition/face/groups/{id}/detections?page=N&pageSize=N
PATCH  /proxy/protect/api/recognition/face/groups/{id}              # rename
DELETE /proxy/protect/api/recognition/face/groups/{id}              # delete cluster + detections
POST   /proxy/protect/api/recognition/face/groups/{id}/image        # multipart — upload reference photo

# Detections
POST   /proxy/protect/api/recognition/face/detections/{id}/image/enhance
PATCH  /proxy/protect/api/recognition/face/detections/{id}/enhanced-image    # hide/show
POST   /proxy/protect/api/recognition/face/assign-group             # {objectIds, groupId|null}
POST   /proxy/protect/api/recognition/v2/merge-group                # {fromGroupIds, toGroupId}

# Events (face harvest — see note below)
GET    /proxy/protect/api/events?type=smartDetectZone&orderDirection=DESC&start=&end=&limit=

# Thumbnails
GET    /proxy/protect/api/thumbnails/{thumbnailId}                  # raw face crop
GET    /proxy/protect/api/thumbnails/enhanced/{enhancedImageId}     # enhanced crop

# AI Key
GET    /proxy/protect/api/aiprocessors
GET    /proxy/protect/api/bootstrap                                 # queue stats
POST   /proxy/protect/api/aiprocessors/retroactive-processing/start
POST   /proxy/protect/api/aiprocessors/retroactive-processing/{cancel,pause,resume}
```

## Our HTTP API (webapp)

The webapp exposes a thin JSON layer at `/api/*` for the frontend. Quick
reference — see [app/webapp.py](app/webapp.py) for full schemas.

```
# Browse
GET  /api/identities                        # named clusters, sorted by recency
GET  /api/unenrolled?offset=N&limit=N       # unnamed clusters
GET  /api/groups/{id}/avatar                # cover image (Protect's reference photo; ?enhanced=true for a sharpened crop)
GET  /api/groups/{id}/detections            # paginated
GET  /api/identities/{id}/detections        # alias for the above, scoped to a named cluster
GET  /api/thumbnails/{id}?enhanced=bool     # single face crop

# Mutations
POST   /api/merge                           # {fromGroupIds, toGroupId}
POST   /api/groups/delete                   # {ids:[...]}
PATCH  /api/identities/{id}                 # {name}
POST   /api/detections/reassign             # {detectionIds, groupId|null}
GET    /api/identities/{id}/best-avatar-candidates
POST   /api/identities/{id}/best-avatar     # {enhancedImageId}

# Enhancer status
GET  /api/enhancer/status                   # cycle state + AI Key queue
POST /api/enhancer/run-now

# Retroactive processing
GET  /api/retroactive/status
POST /api/retroactive/start                 # {numberOfEvents, allCameras, cameraIds}
POST /api/retroactive/cancel

# AI (Gemini)
GET  /api/ai/status                         # {available, sdk_installed}
GET  /api/ai/suggest?groupId=...            # one face → {identityId, name, confidence, reason}
POST /api/ai/suggest-batch                  # {groupIds:[...]} → {results:[{groupId, identityId, name, confidence, reason}]}
```

The triage list (`GET /api/unenrolled`) merges two sources so no face is
missed: Protect's `/recognition/face/groups` listing **plus** a sliding-window
harvest of recent `smartDetectZone` events (group ids are read from both the
thumbnail `group` object and its `labels` array). Tune depth with
`PHANTOM_WINDOWS` / `PHANTOM_EVENT_LIMIT` / `PHANTOM_CACHE_TTL`.

## Versioning

The app exposes its version via `GET /api/version` and shows it next to
the title in the webapp header. The source of truth is
[app/version.py](app/version.py); every behavioural change bumps it. See
[CHANGELOG.md](CHANGELOG.md) for the per-version log.

Loose semver:

- **Patch** (`0.1.X`) — fixes, small UI tweaks, internal refactors.
- **Minor** (`0.X.0`) — new features.
- **Major** (`X.0.0`) — breaking config / API changes.

## License

MIT
