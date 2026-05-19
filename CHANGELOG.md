# Changelog

Versioning starts here. Older changes are summarized in the initial entry.

## 0.1.2 — 2026-05-19

- Fix: "🤖 AI pick" badge on the best-avatar tile was getting clipped by
  the candidate grid's scroll container. Moved inside the tile as a
  corner-pill overlay with backdrop blur — no longer cut off.

## 0.1.1 — 2026-05-19

- Add explicit version tracking. `app/version.py` is the source of truth.
- Expose `GET /api/version` returning `{version, buildDate}`.
- Show version next to "Face Triage" in the webapp header.
- Log version on enhancer startup so it's visible in `docker logs`.

## 0.1.0 — 2026-05-19

Initial versioned release. Snapshot of the project at the time versioning
was introduced. Highlights:

- **Enhancer loop**: long-lived background sweep that auto-enhances every
  face detection via the AI Key's `/recognition/face/detections/{id}/image/enhance`
  endpoint, with adaptive throttling based on the AI Key task queue.
- **Face Triage webapp** on port 8080:
  - Lists named identities + unnamed clusters (including phantom groups
    harvested from recent events).
  - Click / drag-drop / 1–9 hotkey to merge unnamed clusters into named identities.
  - Per-identity detail view with detection grid, move/unassign, delete identity.
  - **🎯 Best avatar** picker with optional **🤖 Let AI pick** that previews
    Gemini's choice on a candidate tile before you commit.
  - **🤖 AI suggest** mode that walks unnamed faces and proposes identity
    matches via Gemini, with Y/N confirm and next-card pre-fetching.
  - Enhancer status pill in the header with click-through detail modal and
    a Run-now button.
  - Retroactive processing trigger for the AI Key.
- Modern dark UI (Linear/Vercel-inspired): indigo accent, refined cards,
  modal backdrop blur, smooth motion.
- Show low-quality / show-degraded toggle (defaults on).
- Avatar cache-busting on every mutation so photos refresh immediately.
- Performance: parallel image fetching, server-side group/reference caches,
  client-side infinite scroll on grids.
