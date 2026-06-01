# Vercel + Supabase Deployment Guide

Deploy the Django grape leaf assessment app to **Vercel** with **Supabase Storage** for upload previews and **Hugging Face Space** for model inference (no PyTorch on Vercel).

---

## Architecture

```
User → Vercel (Django UI)
         ├─► Supabase Storage (upload preview URL)
         └─► HF Space Gradio API (prediction)
```

Complete the [Hugging Face guide](../huggingface/GUIDE.md) first so `HF_INFERENCE_URL` points to a running Space.

---

## Prerequisites

- GitHub repo with this project pushed
- [Vercel](https://vercel.com) account
- [Supabase](https://supabase.com) account
- HF Space running (from HF deploy kit)

---

## Step 1 — Supabase project & storage bucket

1. Create a project at [supabase.com/dashboard](https://supabase.com/dashboard)
2. Open **Database → Extensions** → enable **pg_cron**
3. Open **SQL Editor** → **New query**
4. Paste and run [`supabase_setup.sql`](supabase_setup.sql)

This creates:

| Item | Detail |
|------|--------|
| **Bucket** | `leaf-uploads` — **public** read for preview URLs |
| **File types** | JPEG, PNG, WebP, GIF (max 10 MB) |
| **Cron job** | `purge-leaf-uploads-every-48h` — deletes **all** images in the bucket every **48 hours** (00:00 UTC on odd calendar days: 1st, 3rd, 5th, …) |

5. Confirm bucket `leaf-uploads` exists under **Storage**
6. Confirm cron job under **Database → Cron** (or run):

```sql
select jobid, jobname, schedule, command
from cron.job
where jobname = 'purge-leaf-uploads-every-48h';
```

To add the cron job later without re-running the full setup, use [`supabase_cron_cleanup.sql`](supabase_cron_cleanup.sql).

Copy from **Project Settings → API**:

| Setting | Env var |
|---------|---------|
| Project URL | `SUPABASE_URL` |
| `service_role` key (secret) | `SUPABASE_SERVICE_ROLE_KEY` |

Use the **service role** key server-side only (never expose in frontend code).

---

## Step 2 — Hugging Face inference URL

From your running HF Space, set:

```
HF_INFERENCE_URL=https://YOUR_USERNAME-grape-leaf-inference.hf.space
```

If the Space or model repo is private, also set `HF_TOKEN`.

---

## Step 3 — Import project to Vercel

1. Go to [vercel.com/new](https://vercel.com/new) → import your GitHub repo
2. **Root Directory:** `website`
3. **Framework Preset:** Other
4. Copy settings from [`vercel.json`](vercel.json):

| Setting | Value |
|---------|-------|
| Install Command | `pip install -r deploy/vercel/requirements-vercel.txt` |
| Build Command | `bash deploy/vercel/build.sh` |

Alternatively, copy `deploy/vercel/vercel.json` to `website/vercel.json` (root of Vercel project).

---

## Step 4 — Environment variables

In Vercel → **Project → Settings → Environment Variables**, add:

| Variable | Example | Required |
|----------|---------|----------|
| `DJANGO_SECRET_KEY` | long random string | Yes |
| `DEBUG` | `False` | Yes |
| `ALLOWED_HOSTS` | `.vercel.app,your-domain.com` | Yes |
| `SUPABASE_URL` | `https://xxx.supabase.co` | Yes |
| `SUPABASE_SERVICE_ROLE_KEY` | `eyJ...` | Yes |
| `SUPABASE_STORAGE_BUCKET` | `leaf-uploads` | Yes |
| `HF_INFERENCE_URL` | `https://user-grape-leaf-inference.hf.space` | Yes |
| `HF_TOKEN` | `hf_...` | If private Space/repo |

Use [`.env.example`](.env.example) as a template.

---

## Step 5 — Deploy

1. Click **Deploy**
2. Wait for build (`collectstatic` + dependency install)
3. Open the deployment URL (e.g. `https://your-app.vercel.app`)

---

## Step 6 — Verify

1. Open the app URL
2. Upload a leaf image
3. Confirm:
   - Preview image loads from a Supabase `supabase.co/storage/...` URL
   - Prediction percentage appears (may take 30–60s on first request if Space was sleeping)
   - CSS/styles load correctly

---

## Environment variables reference

| Variable | Description |
|----------|-------------|
| `DJANGO_SECRET_KEY` | Django secret (generate a strong value) |
| `DEBUG` | Must be `False` in production |
| `ALLOWED_HOSTS` | Comma-separated hosts (include `.vercel.app`) |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Server-side storage uploads |
| `SUPABASE_STORAGE_BUCKET` | Bucket name (`leaf-uploads`) |
| `HF_INFERENCE_URL` | HF Space base URL |
| `HF_TOKEN` | HF token for private Spaces |

---

## Important notes

### Upload retention (48-hour purge)

Uploaded leaf previews are **temporary**. A Supabase **pg_cron** job deletes **every file** in `leaf-uploads` on a 48-hour cycle. After a purge, old preview URLs in the browser will break — that is expected.

To run a one-off purge manually:

```sql
delete from storage.objects where bucket_id = 'leaf-uploads';
```

To change the schedule, edit the cron expression in `supabase_setup.sql` and re-run [`supabase_cron_cleanup.sql`](supabase_cron_cleanup.sql).

### No persistent disk on Vercel

Do **not** rely on `MEDIA_ROOT` in production. Supabase must be configured or upload previews will fail.

### No PyTorch on Vercel

Inference runs on your HF Space. The slim `requirements-vercel.txt` excludes `torch`.

### Cold starts

HF free-tier Spaces sleep when idle. The first prediction after idle can take 30–60 seconds.

### Custom domain

Add your domain in Vercel → **Domains**, then append it to `ALLOWED_HOSTS`.

### Local development

Use full `requirements.txt` (includes torch) and a local `.pth` file. See root `commands.txt` and HF guide.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `DisallowedHost` | Add your Vercel URL/domain to `ALLOWED_HOSTS` |
| Static CSS missing | Check build logs for `collectstatic`; WhiteNoise must be in middleware |
| Upload fails | Verify Supabase env vars and run `supabase_setup.sql`; enable **pg_cron** extension |
| Preview URL broken after 2 days | Expected — cron purges all leaf uploads every 48 hours |
| No prediction / timeout | Wake HF Space in browser; check `HF_INFERENCE_URL` |
| 401 from HF | Set `HF_TOKEN` in Vercel env |
| Build too large | Ensure `deploy/vercel/requirements-vercel.txt` is used (no torch) |

---

## Verification checklist

- [ ] Deploy succeeds without torch in the bundle
- [ ] Upload shows Supabase public image URL in the UI
- [ ] Prediction returns via HF Space
- [ ] Static CSS loads (WhiteNoise + collectstatic)

---

## Files in this folder

| File | Purpose |
|------|---------|
| `GUIDE.md` | This guide |
| `.env.example` | Environment variable template |
| `supabase_setup.sql` | Public bucket, policies, 48h purge cron |
| `supabase_cron_cleanup.sql` | Cron job only (if setup already ran) |
| `requirements-vercel.txt` | Slim production dependencies |
| `build.sh` | Install + collectstatic |
| `vercel.json` | Vercel config template |
