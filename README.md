# UniFi Protect — Auto Face Enhance

Automatically triggers face enhancement on every detection in UniFi Protect
using your AI Key. Runs as a long-lived Docker container that polls each
face group on an interval, submits enhance jobs for any detection that
hasn't been enhanced yet, and adaptively throttles based on the AI Key's
200-event task queue so nothing gets dropped.

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

Logs:

```bash
docker compose logs -f face-enhance
```

## Face Triage webapp

The container also serves a small built-in webapp at **http://<host>:8080** for
fast bulk-merging of auto-clustered face groups into named identities.

Keyboard-driven workflow:

- **Click** unnamed face cards (or **Shift-click** for ranges, **A** for all)
- **Press 1–9** to merge the selection into the Nth identity in the sidebar
- **N** create a brand-new identity from the selection (rename the first card,
  then merge the rest into it)
- **Double-click** an identity in the sidebar to rename it
- **Retroactive…** button kicks off `/aiprocessors/retroactive-processing/start`
  on the AI Key — useful for unsticking face detections in `queued` state
  because their parent event's RAM step failed

Disable with `WEBAPP_ENABLED=false`.

## Configuration

All config is read from environment variables — see [.env.example](.env.example)
for the full list.

| Variable | Default | Purpose |
| --- | --- | --- |
| `UNIFI_HOST` | *(required)* | Console IP/hostname (UDM, UNVR, Cloud Key, etc.) |
| `UNIFI_USERNAME` | *(required)* | Local admin username |
| `UNIFI_PASSWORD` | *(required)* | Local admin password |
| `POLL_INTERVAL` | `300` | Seconds to sleep between full sweeps |
| `BASE_DELAY` | `0.35` | Base seconds between enhance requests (adaptive) |
| `ONLY_UNENHANCED` | `true` | Skip detections that already have an enhanced image |
| `GROUP_FILTER` | *(empty)* | Restrict to one person by name |
| `LIMIT` | `0` | Cap per cycle (0 = unlimited) |
| `DRY_RUN` | `false` | List detections without enhancing |
| `RUN_ONCE` | `false` | One cycle then exit (good for cron) |
| `FETCH_WORKERS` | `8` | Parallel workers for fetching detections |
| `WEBAPP_ENABLED` | `true` | Serve the Face Triage webapp on port 8080 |
| `WEBAPP_PORT` | `8080` | Webapp port |
| `ENHANCER_ENABLED` | `true` | Run the periodic enhance sweep |

### Create a local-only admin

Do **not** use your Ubiquiti SSO account. In the UniFi console go to:

> Settings → Admins & Users → Add Admin → Restrict to local access only

Use those credentials for `UNIFI_USERNAME` / `UNIFI_PASSWORD`.

## How throttling works

The AI Key has a 200-event task queue and processes roughly 1,000 events
per hour. The script polls `/api/bootstrap` every 20 successful enhances
to read the current queue depth and scales the per-request delay:

| Queue depth | Delay multiplier |
| --- | --- |
| 0–10 | `BASE_DELAY × 1` |
| 10–50 | `BASE_DELAY × 2` |
| 50–100 | `BASE_DELAY × 5` |
| 100–150 | `BASE_DELAY × 15` |
| 150+ | 30s pause |

## API endpoints used

```
GET  /proxy/protect/api/recognition/face/groups
GET  /proxy/protect/api/recognition/face/groups/{id}/detections?page=N&pageSize=N
POST /proxy/protect/api/recognition/face/detections/{id}/image/enhance
GET  /proxy/protect/api/bootstrap
```

## License

MIT
