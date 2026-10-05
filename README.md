# iFind Scraper API

FastAPI service that scrapes internship listings from 7 sources, runs them through the
moderation pipeline (validation → dedup → scam detection) and stores them in MongoDB
staging collection **`internships.mod-unvectorised`**.

Deployed on **Render** as a Docker web service (`render.yaml`). The Dockerfile installs Chrome for the
Selenium scrapers and listens on Render's `$PORT`.

## Pipeline

```
POST /scrape → for each scraper, one after another:
    scrape → validate required fields → SHA-256 fingerprint dedup (vs staging AND live `internships`)
          → scam_detector → insert into  internships.mod-unvectorised
          → POST {VECTORIZER_URL}/vectorize-hnsw   (best effort; publishes auto-approved items)
```

Results are saved **per scraper, as soon as that scraper finishes** (not at the end of the whole run), so a
crash, restart or out-of-memory kill loses at most the scraper that was running. Listings are scored in
one batch per scraper; duplicate and peer-group checks in the detector work within that batch.

Scam-detector decision → `moderation.status` on the staged document:

| decision | score band | `moderation.status` | what happens next |
|----------|-------------|---------------------|-------------------|
| `clear`  | < 30        | `auto_approved`     | vectorizer publishes it to `internships` |
| `review` | 30 – 70     | `pending_review`    | waits in staging for a moderator |
| `block`  | ≥ 70        | `auto_rejected`     | stays in staging |

If the detector itself fails, **every** listing is routed to `pending_review` (never auto-approved).

Each staged document contains the listing fields plus `moderation` (`status`, `score`,
`flags`, `source`, `reviewedBy`, `reviewedAt`, `rejectionReason`, `scamDetails`). It carries
no vectors; those are added by the vectorizer when the listing is published.

## Endpoints

Base URL: `https://<service>.onrender.com` · Interactive docs: `/docs` (Swagger), `/redoc`, `/openapi.json`

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness check → `{"status": "ok"}` |
| `POST` | `/scrape` | Start a background scrape job → **202** with `job_id` |
| `GET`  | `/scrape/{job_id}` | Poll a job's status and stats |
| `GET`  | `/scrape` | List all jobs, newest first |

There is no auth on these endpoints. Anyone with the URL can start a scrape; keep the service
private or put a gateway in front of it.

### `POST /scrape`

Scraping takes minutes (HF proxies time out at ~60 s), so the job runs in a background thread
and the call returns immediately.

Request body (all fields optional):

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `scrapers` | `list[string \| {id: max}]` | all 7 | Plain id uses the default cap of **300** items; `{id: n}` overrides the cap for that scraper (`n` must be a positive integer). |
| `skip_scrape` | `bool` | `false` | `true` skips scraping and re-pushes `checkpoint_internships.json` through the pipeline. The checkpoint lives in the container and is lost on restart. |

Valid scraper ids: `github`, `internshala`, `indeed`, `naukri`, `unstop`, `freshersworld`, `letsintern`.

```bash
# everything, default caps
curl -X POST https://<service>.onrender.com/scrape -H "Content-Type: application/json" -d '{}'

# per-scraper caps, mixed with plain ids
curl -X POST https://<service>.onrender.com/scrape -H "Content-Type: application/json" \
  -d '{"scrapers": [{"github": 50}, {"internshala": 200}, "naukri"]}'
```

Response `202`:

```json
{
  "job_id": "7c1f…",
  "status": "running",
  "scrapers": ["github", "internshala", "naukri"],
  "max_values": {"github": 50, "internshala": 200, "naukri": 300},
  "message": "Scrape job started for: github, internshala, naukri. Poll GET /scrape/7c1f… for status.",
  "started_at": "2026-10-05T14:00:00+00:00"
}
```

Errors: `400` for an unknown scraper id, a non-positive max, or an entry that is neither a string nor `{id: max}`.

### `GET /scrape/{job_id}`

```json
{
  "job_id": "7c1f…",
  "status": "running | done | failed",
  "started_at": "…",
  "finished_at": "… | null",
  "stats": {"saved": 120, "duplicate": 30, "rejected": 12, "errors": 0},
  "error": null
}
```

`stats` is updated **live after each scraper finishes** (so you can watch `saved` grow while the job is still
`running`) and is final when `done`. It also includes `scraped` (items returned by scrapers). Fields: `saved` = inserted into staging, `duplicate` = fingerprint already in
staging or live, `rejected` = missing name/company/link/summary or non-http link, `errors` = unexpected
per-item failures. `404` if the job id is unknown.

### `GET /scrape`

```json
{"jobs": [{"job_id": "…", "status": "done", "started_at": "…", "finished_at": "…", "stats": {…}, "error": null}], "total": 1}
```

### Notes

- Jobs are stored **in memory** (single instance, `--workers 1`). They disappear on restart or when
  the service sleeps; staged documents in MongoDB are unaffected.
- Concurrent `POST /scrape` calls are not blocked. Avoid starting a second job while one is running.

## Environment secrets

Set in the Render dashboard (**Environment**); `sync: false` entries in `render.yaml` are prompted for on first deploy:

| Secret | Required | Description |
|--------|----------|-------------|
| `MONGODB_URI` | yes | MongoDB Atlas connection string (database `ifind`) |
| `COHERE_API_KEY` | for GitHub / Indeed / Internshala | AI enrichment |
| `MAIN_SERVER` | optional | Frontend origin(s) allowed by CORS, comma-separated (e.g. `https://ifind.example.com`). `http://localhost:3000`–`3004` (and `127.0.0.1`) are always allowed. Empty is fine. |
| `VECTORIZER_URL` | recommended | Base URL of the `vectorisationResume` service. After a push, `POST {VECTORIZER_URL}/vectorize-hnsw` publishes auto-approved listings. Unset → they wait until the vectorizer is triggered some other way. |

## Deploying to Render

1. Push the repo to GitHub. In Render: **New + → Blueprint**, select the repo; it reads `render.yaml`
   (or create a **Web Service** manually: Runtime *Docker*, health check path `/health`).
2. Fill in the env vars above. Render sets `PORT` itself.
3. After the build, check `GET /health`, then smoke test `POST /scrape` with `{"scrapers": [{"github": 5}]}`,
   poll the job, and confirm documents appear in `internships.mod-unvectorised`.

Things to know about Render:

- **Memory:** Chrome plus the scam-detector stack (pandas, scikit-learn) needs roughly 2 GB. `render.yaml`
  uses the `standard` plan; the free and starter plans (512 MB) are likely to be OOM-killed during a scrape.
- **Free plan spin-down:** free services sleep after ~15 minutes without traffic, and a sleeping service
  also kills a running scrape thread. Use a paid plan if scrapes must complete unattended.
- **Disk is ephemeral:** the checkpoint file, scam-detector reputation/baseline files and the in-memory job list
  reset on every deploy or restart. Staged documents in MongoDB are unaffected.
- **Region:** pick the one closest to your MongoDB Atlas cluster, and allow Render's outbound IPs in
  Atlas Network Access (or `0.0.0.0/0`).
