# CLAUDE.md — Protect Transcribe

Authoritative guide for working in this codebase. Read this before changing anything.

DO NOT GUESS, IT IS COMPLETELY FORBIDDEN TO GUESS, LIE, AND/OR CHEAT.

IF YOU DO NOT KNOW SOMETHING, DONT PRETEND TO.

---

## What this project is

A FastAPI service that listens for UniFi Protect smart-detection events (speech), pulls the video clip, extracts audio with ffmpeg, and sends it to a **speaches** server (faster-whisper) for transcription. Results are stored in SQLite and served through an Alpine.js + Tailwind CSS single-page UI ("Talky"): Day timeline, Summaries, Search & Ask, Insights, System.

---

## Architecture

```
UniFi Protect NVR
      │  webhook / polling
      ▼
app/routes/webhook.py   ←── POST /webhook  (smart-detect events)
app/routes/sync.py      ←── POST /api/sync (manual backfill)
      │
      │  queue_transcription()
      ▼
SQLite  transcriptions  (status: pending → processing → completed/error)
      │
      │  transcription_worker() polls every 5 s (new events first, backlog newest-first)
      ▼
app/worker.py
      │  _event_window()     →  Protect get_event(): real start/end; defers while ongoing
      │  fetch_audio_clip()  →  uiprotect API  →  MP4 bytes  (start-before .. end+after, ≤15 min)
      │  _extract_audio()    →  ffmpeg          →  WAV bytes
      │  transcribe_audio()  →  speaches        →  JSON result
      ▼
SQLite  (status = completed, transcription, segments, audio_file …)
      │
      ▼
templates/ (Jinja partials)  +  static/js/ (ES modules, Alpine.js SPA)
```

Key invariant: **audio is cached on disk** (WAV, compressed to OGG after N days — the worker decodes OGG back to WAV). `retranscribe-all` resets rows to `pending`, keeps `audio_file` and the old text, and sets `refetch=1`: the worker re-downloads the exact event window from the NVR and falls back to the cached file when the footage is gone.

Row lifecycle columns: `attempts` / `next_attempt_at` (UTC `YYYY-MM-DDTHH:MM:SS`, compared as text against `strftime('now')`) implement deferral (event still ongoing, footage not written yet) and exponential-backoff retries of transient failures (NVR/speaches unreachable) — only after `_MAX_ATTEMPTS` does a row become `error`. `READY_SQL` in worker.py is the single definition of "can be processed now".

Hallucinations: bad segments (compression ratio / no-speech) are dropped, a looping result is retried once at temperature 0.4, then loops are collapsed and the remaining text kept. Only pure junk becomes `filtered` ("Hallucination discarded: <raw>") — not `error`.

---

## Package layout

```
app/
  __init__.py        FastAPI factory, lifespan (DB init, worker task, Protect connect)
  config.py          All env-var constants; no env calls anywhere else
  database.py        get_connection(), init_database(), get_settings(), save_setting()
  protect.py         Singleton uiprotect client with asyncio.Lock
  transcription.py   fetch_audio_clip(), _extract_audio(), transcribe_audio(), save_audio_file()
  worker.py          queue_transcription(), process_pending_transcription(), transcription_worker()
  summaries.py       generate_summary() via Ollama (hierarchical daily/weekly/monthly/yearly)
  routes/
    health.py        GET /health
    webhook.py       POST /webhook
    sync.py          POST /api/sync
    transcriptions.py  All transcription CRUD + retranscribe-all
    settings.py      GET/PUT /api/settings, test-whisper, test-protect, speaches model endpoints
    summaries.py     GET /api/summaries, POST /api/summaries/generate (period: daily|weekly|monthly|yearly)

static/
  js/main.js         Entry module: registers Alpine store, page components, $icon magic, keyboard shortcuts
  js/store.js        $store.ui — hash routing (#/day/2026-11-14?clip=12), stats/health, WebSocket, toasts, confirm, theme
  js/api.js          fetch wrappers + waitForJob() for background jobs
  js/format.js       escaping, Markdown, local-time helpers (timestamps are read as wall-clock strings)
  js/icons.js        Lucide icon subset (generated from lucide-static; add names there)
  js/pages/*.js      One Alpine component per page + clip.js (detail panel with wavesurfer player)
  vendor/            Self-hosted Alpine.js and wavesurfer.js (no CDNs)
  manifest.webmanifest, icon*.{svg,png}   Installable web app
  app.css            Compiled Tailwind CSS (generated during Docker build — do NOT edit by hand, gitignored)

templates/
  index.html         Shell: sidebar / bottom tabs, status strip, page includes
  pages/*.html       day, summaries, search (incl. Ask), insights, system
  partials/*.html    clip panel, dialogs (toasts, confirm, shortcuts), settings save bar

tests/
  conftest.py        tmp_db fixture (patches get_connection), client fixture (no-op lifespan)
  test_*.py          pytest-asyncio tests
```

---

## Key design decisions

### Single SQLite connection factory
`get_connection()` in `app/database.py` is the **only** place a connection is opened for the main DB. All route modules import it. Tests monkeypatch it to redirect to a temp file. Never open `sqlite3.connect(DATABASE_PATH)` in routes — only `worker.py` does this directly because it runs outside the request lifecycle.

### Settings are live
`get_settings()` reads from the DB on every call. `transcribe_audio()` calls it at transcription time, so model/language changes take effect on the next queued item without a restart.

### Speaches model management
Models are discovered from the speaches registry (`GET /v1/registry?task=automatic-speech-recognition`) and downloaded on demand (`POST /v1/models/{model_id:path}`). The `/api/settings/speaches-models` endpoint merges registry + installed model lists. The UI shows which models are installed vs available to download.

### Tailwind CSS
CSS is compiled in a **multi-stage Docker build**: a `node:22-slim` stage installs `tailwindcss` and `@tailwindcss/cli` via npm and runs `npx @tailwindcss/cli` to produce `static/app.css`, which is then copied into the final Python image. The standalone CLI binary is **not** used (it strips the theme). Configuration lives entirely in `app.css.src` — no `tailwind.config.js` (that's v3). Dark mode uses `@custom-variant dark (&:where(.dark, .dark *))` so `dark:` utilities activate when `.dark` is on any ancestor, persisted to `localStorage`.

### Summaries are hierarchical
daily ← transcripts (map-reduce chunked if over `_CHUNK_CHARS`), weekly/monthly ← daily summaries, yearly ← monthly summaries. Rollups generate missing child summaries first. Calls use Ollama's native `/api/chat` with `options.num_ctx` (the OpenAI-compat endpoint can't set it and silently truncates). Staleness is content-based: `summaries.source_hash` fingerprints the inputs (transcript texts for daily, child summary texts for rollups). `auto_summary_worker` waits until the transcription queue has been idle for 30 min (so a mass re-transcription settles first), then `run_rollups()` regenerates missing/stale summaries level by level — changes propagate day → week/month → year in one pass. Periods containing "now" are only included once a night after `auto_summary_time`.

Timestamps are stored as local ISO strings with offset. Group by `substr(timestamp, ...)`, **never** `strftime(..., timestamp)` — SQLite converts to UTC and shifts late-night events to the previous day.

Camera names are microphone *locations*, not speakers; the summary prompt says so explicitly.

### User edits survive re-transcription
`PATCH /api/transcriptions/{id}` stores hand-corrected text and keeps the machine text in `original_transcription`. While `edited_at` is set, the worker's `_finish()` writes new machine output to `original_transcription` and leaves the user's text. Manual "Re-transcribe" sets `priority=1` so it jumps the backlog.

### Keyword alerts
`app/alerts.py` POSTs (plain text + `Title` header; ntfy/Home Assistant compatible) when a first-time transcription of a recent event contains an `alert_keywords` word. Never for re-runs (`refetch`) or old events.

### asyncio.Lock for Protect client
`get_protect_client()` is called from the worker and from routes concurrently. The lock in `protect.py` ensures only one coroutine initialises the singleton at a time.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PROTECT_HOST` | `argos.local` | UniFi Protect NVR hostname/IP |
| `PROTECT_PORT` | `443` | NVR HTTPS port |
| `PROTECT_USERNAME` | — | Local admin username |
| `PROTECT_PASSWORD` | — | Local admin password |
| `WHISPER_URL` | `http://whisper-server:8000` | speaches base URL |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama base URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model for summaries |
| `DATABASE_PATH` | `/data/transcriptions.db` | SQLite file path |
| `AUDIO_PATH` | `/data/audio` | WAV cache directory |
| `AUDIO_BUFFER_BEFORE` | `5` | Seconds before event (default, overridden by DB setting) |
| `AUDIO_BUFFER_AFTER` | `10` | Seconds after event (default, overridden by DB setting) |
| `TZ` | `Europe/Copenhagen` | Timezone for timestamps |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## Speaches API used

| Purpose | Call |
|---|---|
| List installed models | `GET /v1/models` |
| Discover all registry models (ASR) | `GET /v1/registry?task=automatic-speech-recognition` |
| Download a model | `POST /v1/models/{model_id}` (model ID is a path param, may contain `/`) |
| Transcribe audio | `POST /v1/audio/transcriptions` (multipart). Accepted fields: `model, language, prompt, response_format, temperature, timestamp_granularities, stream, hotwords, vad_filter` — anything else (e.g. `initial_prompt`, `condition_on_previous_text`, `beam_size`, `diarize`) is silently ignored. Check `GET /openapi.json` before adding a field. |

Models live in the speaches PVC as a Hugging Face cache (`models--<org>--<name>/refs/main` + `snapshots/<rev>/`). The server has no internet egress, so non-hub models (e.g. `local/roest-v3-whisper-1.5b-ct2`, a CTranslate2 conversion of `CoRal-project/roest-v3-whisper-1.5b`) are converted elsewhere (`ct2-transformers-converter` with the **same ctranslate2 version as the server**) and copied in; the snapshot needs a README with `library_name: ctranslate2` and the `automatic-speech-recognition` tag or speaches rejects it. Do not send `prompt`: tested on real clips, the Danish fine-tunes echo it or invent text from it.

When transcription returns 404 "not installed", the worker auto-triggers a download then retries. Download can take several minutes; the transcription request times out at 300 s, the download at 600 s.

---

## Dev commands

```bash
# Install dev deps
pip install -r requirements-dev.txt

# Lint
ruff check .

# Format check / auto-fix
ruff format --check .
ruff format .

# Type check
mypy app/

# Tests with coverage
pytest --cov=app --cov-report=term-missing

# Compile Tailwind CSS (requires tailwindcss v4 standalone CLI on PATH)
# Download from: https://github.com/tailwindlabs/tailwindcss/releases/latest
tailwindcss -i app.css.src -o static/app.css --minify

# Build Docker image
docker build -t protect-transcribe .

# Run locally (needs a real speaches + Protect reachable)
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

---

## CI pipeline (`.github/workflows/docker-image.yml`)

```
lint → typecheck → test → scan-source → build → release → scan-image
```

- **lint**: `ruff check` + `ruff format --check`
- **typecheck**: `mypy app/`
- **test**: `pytest --cov=app` (coverage report uploaded as artifact)
- **scan-source**: Trivy filesystem scan, exits 1 on HIGH/CRITICAL
- **build**: multi-arch Docker build (`linux/amd64`, `linux/arm64`), pushed to GHCR
- **release**: GitHub Release created on version tags
- **scan-image**: Trivy image scan, SARIF uploaded (non-blocking, exit-code 0)

---

## Testing conventions

- `conftest.py` patches `app.database.get_connection` and the per-module copies of it so all DB calls hit a temp SQLite file
- The `client` fixture stubs out lifespan (no worker, no Protect connect)
- Use `monkeypatch.setattr` not `unittest.mock.patch` for consistency
- `asyncio_mode = auto` in pyproject.toml — mark async tests with `async def`, no decorator needed
- Do not mock the speaches HTTP calls in worker/transcription tests — use `respx` or `httpx.MockTransport` if needed

---

## Rules

- **Never** call `os.getenv()` outside `app/config.py`
- **Never** open `sqlite3.connect()` in route handlers — use `get_connection()`
- **Never** edit `static/app.css` by hand — it is generated
- **Do not** add `AVAILABLE_MODELS` to `config.py` — the model list comes from speaches registry at runtime
- Tailwind scans `templates/**/*.html` and `static/js/**/*.js`; shared component classes (`.btn`, `.card`, `.field`, …) live in `app.css.src` under `@layer components`
- Anything rendered with `x-html` must go through `escapeHtml` (format.js) first; transcript text is untrusted
- Slow LLM work (summary rebuilds, Ask) runs via `app/jobs.py` and is polled with `GET /api/jobs/{id}` — Cloudflare cuts requests at ~100 s
- ruff replaces flake8/black/isort — do not add those as dependencies
- mypy runs with `ignore_missing_imports = true`; add `# type: ignore[attr-defined]` only when stubs are genuinely missing
- Bare `try/except/pass` → `contextlib.suppress(...)`
