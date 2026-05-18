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
Cards display the **enhanced** face crop when available — much clearer than
the raw cluster avatar Protect itself uses.

- **Click** card → select. **Shift-click** → range. **A** → select all. **Esc** → clear.
- **Click an identity** in the sidebar → merges the selection into it.
- **Drag a card** onto an identity → also merges (the whole card drags, not the image).
- **Press 1–9** → keyboard merge into the Nth identity.
- **N** → "New identity" dialog: renames the first selected card to a new
  name, then merges the rest into it.
- **Del** or **Backspace** → bulk-deletes selected face groups (cluster *and*
  its detections — permanent).
- **🤖 AI suggest** → walks every unnamed card and uses Google Gemini to
  suggest a matching identity. Each suggestion is shown with **Y** (merge)
  or **N** (skip). Pre-fetches the next card while you review the current.
  Requires `GEMINI_API_KEY`.

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

### Webapp

| Variable | Default | Purpose |
| --- | --- | --- |
| `WEBAPP_ENABLED` | `true` | Serve the Face Triage webapp |
| `WEBAPP_PORT` | `8080` | Port to bind (use the same value in compose port mapping) |
| `WEBAPP_HOST` | `0.0.0.0` | Bind interface |

### Gemini AI suggest (optional)

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | *(empty)* | Enables 🤖 AI suggest. Get one at https://aistudio.google.com/app/apikey |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Override the model. `gemini-2.5-flash-lite` is cheapest (~$0.0003/query) but less accurate; `gemini-2.5-pro` is best (~$0.005/query). |

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
GET  /api/groups/{id}/avatar                # cover image (enhanced when available)
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
GET  /api/ai/suggest?groupId=...            # {identityId, name, confidence, reason}
```

## License

MIT
