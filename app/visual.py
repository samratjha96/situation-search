"""Local, opt-in retrieval of playable intervals around visually similar frames."""
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
MODEL = DATA / "models" / "clip-search"
REVISION = "6b7c523703ae26e39be4b03774966f8571c7f523"


def session(model_file):
    import onnxruntime as ort

    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    return ort.InferenceSession(str(model_file), options, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def text_encoder():
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    return tokenizer, session(MODEL / "onnx" / "text_model_quantized.onnx")


@lru_cache(maxsize=8)
def frame_index(path, mtime):
    # Replacement of the on-disk index gives the next request a new cache key.
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        if data["revision"].item() != REVISION:
            raise ValueError("Frame vectors require reindexing")
        vectors = data["vectors"]
        names = data["frames"].tolist()
        if vectors.shape != (len(names), 512) or not np.isfinite(vectors).all():
            raise ValueError("Invalid frame vectors")
        return vectors, names


def visual_candidates(clip, query):
    def response(rows, status, message):
        return {"results": rows, "ranking": "visual candidates", "count": len(rows),
                "status": status, "no_match": False, "message": message}

    if not query.strip():
        return response([], "unverified", "Enter a description to look for visually similar frames.")
    index = DATA / "clips" / clip.name / "visual.npz"
    files = (index, MODEL / "tokenizer.json", MODEL / "onnx" / "text_model_quantized.onnx")
    if not all(p.is_file() for p in files):
        return response([], "unavailable", "Local visual search is unavailable for this match.")
    try:
        import numpy as np

        vectors, names = frame_index(index, index.stat().st_mtime_ns)
        if clip.frames != [clip.prefix + name for name in names]:
            raise ValueError("Indexed frame list differs from match")
        tokenizer, encoder = text_encoder()
        ids = tokenizer.encode(query).ids
        if len(ids) > 77:
            return response([], "unverified", "Shorten the description to fit the visual model’s 77-token input.")
        prediction, = encoder.run(["text_embeds"], {"input_ids": np.array([ids], dtype=np.int64)})
        scores = vectors @ (prediction[0] / max(np.linalg.norm(prediction[0]), 1e-9))
    except ImportError:
        return response([], "unavailable", "The local visual-search dependencies are unavailable.")
    except Exception:
        # ONNX and tokenizer errors include internal paths; keep them out of the response.
        import logging
        logging.getLogger(__name__).exception("Local visual index or model failed for %s", clip.name)
        return response([], "unavailable", "The local visual model or frame index could not be read.")

    rows, spans = [], []
    for i in np.argsort(-scores, kind="stable"):
        frame = int(i) * clip.interval
        start, end = max(0, frame - 3), min(clip.duration, frame + 5)
        if start >= end or any(start < b and a < end for a, b in spans):
            continue
        spans.append((start, end))
        rows.append({"t_seconds": start, "timecode": f"{int(start)//60:02d}:{int(start)%60:02d}",
                     "episode_end": end, "event": "visual candidate", "image": clip.frames[i],
                     "prose": f"The frame at {int(frame)//60:02d}:{int(frame)%60:02d} is visually similar to the description.",
                     "evidence": [{"t": frame, "detail": "Sampled video frame", "source": "frame similarity"}]})
        if len(rows) == 5:
            break
    return response(rows, "unverified", "Ranked by frame similarity. Review the video to check movement, roles, and outcome.")
