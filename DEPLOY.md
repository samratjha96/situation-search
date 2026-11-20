# Serve a local index with Docker

The image contains the application and Python dependencies. Its build context
excludes the footage, indexes, model weights, credentials, and evaluation data.
Build it on a local Docker engine:

```bash
docker build -t situation-search .
```

Mount the prepared clips read-only at runtime. This example also mounts the
local CLIP model used by visual candidate search. Omit that mount for a
structured-search-only instance.

```bash
docker run --rm -p 127.0.0.1:8008:8008 \
  --mount "type=bind,src=$PWD/data/clips,dst=/app/data/clips,readonly" \
  --mount "type=bind,src=$PWD/data/models/clip-search,dst=/app/data/models/clip-search,readonly" \
  situation-search
```

Open http://127.0.0.1:8008. Each clip needs its processed `index.json`, the
video named by that index, and its referenced thumbnails. Visual candidates
also need the clip's `visual.npz`. The media routes serve only the indexed
video and thumbnails. Set `DEFAULT_CLIP` to the desired clip directory name to
choose which match opens first.

All queries run on the serving machine. The image does not require an inference
endpoint or API key. Building the image fetches its public base and Python
dependencies. See [the README](README.md) for local index preparation and
the separate extraction workflow for new footage.
