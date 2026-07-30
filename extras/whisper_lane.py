"""Whisper-small ASR on one Inferentia2 NeuronCore, mlx-models parity.

Mirror of github.com/alpharomercoma/mlx-models 3_whisper/transcribe.py so the
demo is comparable across silicon (Apple M5 there, Inferentia2 here): the same
public-domain JFK inaugural clip (whisper.cpp samples), the same stdlib-wave
reader, the same "--- transcription ---" output. What the port target forces:

  * optimum-neuron traces ONE static graph ahead of time (batch 1, decoder
    sequence_length 128) where MLX runs dynamic shapes JIT. The compile is a
    cached first-class artifact under /opt/np/models/neuron-compiled/;
    compile_s is reported as 0 on a cache hit.
  * Whisper-with-KV-cache is not yet supported on Neuron (optimum-neuron
    0.4.3 forces use_cache=False), so per-token decode cost grows with the
    sequence -- RTF against the clip's duration is the honest headline.

    python3 extras/whisper_lane.py --out /tmp/whisper.json
    # expect (M5 reference, mlx 3_whisper README): "And so my fellow
    # Americans ask not what your country can do for you ask what you can
    # do for your country."  -- head_match=True, rtf < 1
"""
import argparse
import json
import os
import sys
import time
import traceback
import wave
from urllib.request import urlretrieve

MODEL_ID = "openai/whisper-small"
# Public-domain JFK inaugural clip (16 kHz mono 16-bit WAV), ~11s.
# Byte-identical source to mlx-models 3_whisper/transcribe.py SAMPLE_URL.
SAMPLE_URL = "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav"
# Known opening words of the clip, normalized; used for the fuzzy head match.
REFERENCE_HEAD = "and so my fellow americans"
BATCH_SIZE = 1
DECODER_SEQ_LEN = 128     # static decoder length, per the 0.4.3 whisper docs


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--compiled-dir",
                    default="/opt/np/models/neuron-compiled/whisper-small")
    ap.add_argument("--data-dir", default="/opt/np/models/neuron-compiled/data")
    ap.add_argument("--audio", default=None,
                    help="path to a 16-bit PCM WAV (default: download JFK clip)")
    return ap


def compute_rtf(wall_s, audio_s):
    """Real-time factor: processing seconds per audio second (< 1 means
    faster than realtime). Pure; unit-tested."""
    if audio_s <= 0:
        raise ValueError("audio_s must be positive")
    return wall_s / audio_s


def normalize_text(text):
    """Lowercase, strip punctuation, collapse whitespace. Pure; unit-tested."""
    cleaned = "".join(c.lower() if c.isalnum() or c.isspace() else " "
                      for c in text)
    return " ".join(cleaned.split())


def matches_reference_head(text):
    """Fuzzy startswith against the known JFK opening words: robust to
    punctuation/casing drift, strict about actual mistranscription."""
    return normalize_text(text).startswith(REFERENCE_HEAD)


def get_sample(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "jfk.wav")
    if not os.path.exists(path):
        print(f"downloading sample audio -> {path}", flush=True)
        urlretrieve(SAMPLE_URL, path)
    return path


def read_wav(path):
    """Read a 16 kHz mono 16-bit PCM WAV into float32 in [-1, 1] (the same
    stdlib-wave path as the mlx script: no ffmpeg dependency)."""
    import numpy as np
    with wave.open(str(path), "rb") as w:
        sr, n, ch, width = (w.getframerate(), w.getnframes(),
                            w.getnchannels(), w.getsampwidth())
        frames = w.readframes(n)
    if width != 2:
        raise SystemExit(f"expected 16-bit PCM WAV, got {width * 8}-bit ({path})")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        print(f"warning: sample rate is {sr} Hz; Whisper expects 16 kHz")
    return audio


def _atomic_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, path)


def _record_failure(out, stage):
    """Structured failure receipt NEXT TO --out (launch_vllm load_failure
    pattern): the lane is retried on the next driver pass because --out
    itself stays absent, but the evidence survives."""
    reason = traceback.format_exc().strip().splitlines()[-1]
    path = out[:-5] + ".failure.json" if out.endswith(".json") \
        else out + ".failure.json"
    _atomic_json(path, {
        "lane": "whisper", "status": "lane_failed", "stage": stage,
        "reason": reason,
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    print(f"whisper lane FAILED at {stage}: {reason}", file=sys.stderr)
    print(f"failure recorded: {path}", file=sys.stderr)


def main():
    args = build_parser().parse_args()

    try:
        from transformers import AutoProcessor
        from optimum.neuron import NeuronWhisperForConditionalGeneration
    except Exception:
        _record_failure(args.out, "import")
        return 1

    compiled = os.path.abspath(args.compiled_dir)
    cached = os.path.exists(os.path.join(compiled, "config.json"))
    compile_s = 0.0
    if not cached:
        print(f"--- compiling {args.model} -> {compiled} (one-time, minutes) ---",
              flush=True)
        t0 = time.perf_counter()
        try:
            # Export API per optimum-neuron 0.4.3 whisper docs: static
            # shapes, weights kept outside the NEFF so the artifact stays
            # cache-friendly. Task: automatic-speech-recognition (implied by
            # the model class).
            model = NeuronWhisperForConditionalGeneration.from_pretrained(
                args.model,
                export=True,
                inline_weights_to_neff=False,
                auto_cast="all",
                auto_cast_type="bf16",
                batch_size=BATCH_SIZE,
                sequence_length=DECODER_SEQ_LEN,
            )
            model.save_pretrained(compiled)
            # Ship the processor with the artifact so a re-provisioned box
            # loads everything from one directory.
            AutoProcessor.from_pretrained(args.model).save_pretrained(compiled)
            del model
        except Exception:
            _record_failure(args.out, "compile")
            return 1
        compile_s = time.perf_counter() - t0

    # Always (re)load from the compiled dir so load_s means the same thing
    # on cold and warm runs.
    t0 = time.perf_counter()
    try:
        model = NeuronWhisperForConditionalGeneration.from_pretrained(compiled)
        processor = AutoProcessor.from_pretrained(compiled)
    except Exception:
        _record_failure(args.out, "load")
        return 1
    load_s = time.perf_counter() - t0

    audio_path = args.audio or get_sample(args.data_dir)
    audio = read_wav(audio_path)
    audio_s = len(audio) / 16000.0

    print(f"# whisper  model={args.model} batch={BATCH_SIZE} "
          f"decoder_seq={DECODER_SEQ_LEN} auto_cast=all/bf16 cached={cached} "
          f"compiled_dir={compiled} "
          f"audio={os.path.basename(str(audio_path))} ({audio_s:.1f}s)",
          flush=True)

    try:
        input_features = processor(
            audio, sampling_rate=16000, return_tensors="pt").input_features
        # Warmup: the first call pays one-off runtime init; the timed call
        # below is a plain second transcription of the same clip.
        model.generate(input_features, max_length=DECODER_SEQ_LEN)
        t0 = time.perf_counter()
        # TODO-VERIFY: the 0.4.3 docs call bare generate(); we cap
        # max_length at the compiled decoder length because transformers'
        # whisper default (448) would overrun the 128-token static graph if
        # EOS never fired. If the venv build rejects max_length here, drop it.
        predicted_ids = model.generate(input_features,
                                       max_length=DECODER_SEQ_LEN)
        wall = time.perf_counter() - t0
        text = processor.batch_decode(
            predicted_ids, skip_special_tokens=True)[0].strip()
    except Exception:
        _record_failure(args.out, "transcribe")
        return 1

    print("\n--- transcription ---")
    print(text)

    rtf = compute_rtf(wall, audio_s)
    payload = {
        "model": args.model,
        "compile_s": round(compile_s, 1),
        "load_s": round(load_s, 1),
        "transcribe_wall_s": round(wall, 3),
        "audio_s": round(audio_s, 2),
        "rtf": round(rtf, 3),
        "transcript_text": text,
        "transcript_matches_reference_head": matches_reference_head(text),
        "mlx_reference": "github.com/alpharomercoma/mlx-models 3_whisper (Apple M5)",
        "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(args.out, payload)
    print(f"rtf {rtf:.3f}  |  {wall:.2f}s for {audio_s:.1f}s audio  |  "
          f"compile {payload['compile_s']}s  load {payload['load_s']}s  |  "
          f"head_match={payload['transcript_matches_reference_head']}"
          f"  ->  {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
