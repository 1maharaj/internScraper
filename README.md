---
title: iFind Scraper API
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# iFind Scraper API

FastAPI service that runs internship scrapers and pushes results through the
moderation pipeline into MongoDB.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/scrape` | Start a scrape job (returns `job_id` immediately) |
| `GET` | `/scrape/{job_id}` | Poll job status |
| `GET` | `/scrape` | List all jobs |
| `GET` | `/docs` | Swagger UI |

## Environment secrets

Set these in the Space **Settings → Secrets** tab:

| Secret | Description |
|--------|-------------|
| `MONGODB_URI` | MongoDB Atlas connection string |
| `COHERE_API_KEY` | Cohere API key (used by GitHub / Indeed / Internshala scrapers for AI enrichment) |

## Usage

```bash
# Start all scrapers
curl -X POST https://<your-space>.hf.space/scrape \
  -H "Content-Type: application/json" \
  -d '{"scrapers": ["github", "internshala", "indeed", "naukri", "unstop", "freshersworld", "letsintern"]}'

# Poll for result
curl https://<your-space>.hf.space/scrape/<job_id>
```
