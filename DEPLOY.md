# Deploy

The app serves a **pre-built index**. All heavy AI runs at index time. To serve
the app, you need only a small web server and query-time calls to the inference
endpoint (for the query embedding and the reranker). There is no GPU. The app uses
about 250 MB of RAM.

## 1. Build the index

Run this one time, on a machine with the endpoint, `ffmpeg`, and Whisper:

    CLIP_VIDEO=data/clip2.mp4 python pipeline/build.py

This makes `data/clips/clip2/index.json` and puts the frames and a copy of the clip
in the same directory. A new clip also needs a line in `CLIP_CONF` in `app/server.py`.

## 2. Build the container

    docker build --build-arg CLIP=clip2.mp4 -t situation-search .

## 3. Run it

The key stays on the server. Viewers never see it. Put the endpoint and the key in
a git-ignored `.env` next to the run command:

    OPENROUTER_URL=https://<endpoint>/v1
    OPENROUTER_API_KEY=sk-...

Then run:

    docker run -d --name situation-search -p 8008:8008 \
      --env-file .env situation-search

The app is now at http://<host>:8008.

## 4. Expose a public HTTPS URL

A Cloudflare Tunnel is one option. It is free. It gives an
https://…trycloudflare.com URL:

    # on the same host, pointed at the container
    cloudflared tunnel --url http://localhost:8008

You can also use a named tunnel on your own domain, or Tailscale Funnel.

## Homelab or cloud

- **Homelab:** a good fit — no GPU, small. It needs outbound access to the
  inference endpoint (`OPENROUTER_URL`) and the key. Use a tunnel for the public
  URL.
- **Small cloud VM** (Fly.io, Render, or a low-cost droplet): use this if you want
  the demo always up and independent of your home network. It uses the same image
  and still needs no GPU.

## Notes

- The app reads `OPENROUTER_API_KEY` from the environment (server-side). Every
  viewer search uses it, so it uses your quota. A rotate-safe deployment needs the
  local-model variant (the query embedding and the reranker through
  sentence-transformers).
- To ship a different clip, build the index for it. Then run
  `docker build --build-arg CLIP=<file>.mp4 ...`.
