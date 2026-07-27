# Deploy

The app serves a **pre-built index**. All heavy AI runs at index time; serving
only needs a small web server plus query-time calls to the inference endpoint (for
embedding + reranking the search query). No GPU. ~250 MB RAM.

## 1. Build the index (once, on a machine with the gateway + ffmpeg + whisper)

    CLIP_VIDEO=data/clip2.mp4 python pipeline/build.py

This produces `data/index.json` and leaves the clip + frames in `data/`.

## 2. Build the container

    docker build --build-arg CLIP=clip2.mp4 -t situation-search .

## 3. Run it (key stays server-side, never exposed to viewers)

Put the endpoint + key in a git-ignored `.env` next to the compose/run command:

    OPENROUTER_URL=https://<endpoint>/v1
    OPENROUTER_API_KEY=sk-...

Then:

    docker run -d --name situation-search -p 8008:8008 \
      --env-file .env situation-search

App is now at http://<host>:8008.

## 4. Expose a public HTTPS URL (no port-forwarding)

Cloudflare Tunnel is the easiest — free, gives an https://…trycloudflare.com URL:

    # on the same host, pointed at the container
    cloudflared tunnel --url http://localhost:8008

(or a named tunnel on your own domain). Tailscale Funnel works too.

## Homelab vs cloud

- **Homelab:** perfect — no GPU, small. Just needs outbound reach to the
  inference endpoint (`OPENROUTER_URL`) and the key. Use a tunnel for the public URL.
- **Small cloud VM** (Fly.io / Render / a $5 droplet): use if you want the demo
  always-up and independent of your home network. Same image, still no GPU.

## Notes

- The `OPENROUTER_API_KEY` is read from the environment (server-side). Every viewer
  search uses it, so it's your quota; rotate-safe deployment would need the
  local-model variant (query embed + rerank via sentence-transformers).
- To ship a different clip: rebuild the index for it, then
  `docker build --build-arg CLIP=<file>.mp4 ...`.
