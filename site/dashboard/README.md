# RSIBox Delivery Dashboard (`site/dashboard/`)

A lightweight, dual-mode engineering delivery dashboard for `hao-ai-lab/rsibox`.

- **Static Mode** (GitHub Pages / Cloudflare Pages / any static file server): renders `site/facts/delivery-matrix.json` and `assets/` charts cleanly with zero backend dependency and zero console errors.
- **Live Mode** (`server.py`): adds 5-second `/api/state` polling for real-time B200 GPU VRAM/utilization (`nvidia-smi`), disk/CPU/RAM telemetry, Git commits/worktrees, working-tree status, and cached Open PRs (`gh pr list`).

## 1. Local Preview

From the repository root:

```bash
python3 site/dashboard/server.py --port 8765
```

Open:
- `http://127.0.0.1:8765/dashboard` (or `http://127.0.0.1:8765/`, which redirects to `/dashboard`)
- `http://127.0.0.1:8765/site/dashboard/`

## 2. Static Hosting

The dashboard is dependency-free and requires no build step. Publish the repository root (or at minimum `site/dashboard/`, `site/facts/`, and `assets/`):

```bash
python3 -m http.server 8765 --bind 127.0.0.1
# Open http://127.0.0.1:8765/site/dashboard/
```

When `/api/state` returns `404` or is unreachable, the UI silently enters **Static Mode**.

## 3. B200 Deployment Behind `cloudflared` (`rsibox.fastrsi.ai/dashboard`)

1. Run the dashboard server bound to loopback on the B200 node:

   ```bash
   python3 site/dashboard/server.py --host 127.0.0.1 --port 8765
   ```

2. Route `rsibox.fastrsi.ai` ingress rules in your `cloudflared` tunnel config to `http://127.0.0.1:8765` (covering `/dashboard`, `/dashboard/*`, `/api/state`, `/site/facts/*`, `/site/dashboard/*`, and `/assets/*`).
3. Keep `--host 127.0.0.1` so the telemetry endpoint is only exposed through the authenticated tunnel.
