"""
Audio fetching, ffmpeg extraction, Whisper transcription, and audio analysis.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import re
import struct
import tempfile
import wave
from datetime import datetime  # noqa: TC003 — runtime annotation
from pathlib import Path

import httpx

from app.config import AUDIO_PATH, WHISPER_URL
from app.database import get_settings
from app.http_clients import get_whisper_client
from app.metrics import inc
from app.protect import get_protect_client
from app.util import find_camera

logger = logging.getLogger(__name__)


def compute_audio_rms(audio_data: bytes) -> float:
    """
    Compute the RMS energy of 16-bit PCM WAV audio, normalised to 0-1.

    Returns 0.0 for empty or unparseable audio.
    """
    try:
        with wave.open(io.BytesIO(audio_data), "rb") as wf:
            n_frames = wf.getnframes()
            if n_frames == 0:
                return 0.0
            raw = wf.readframes(n_frames)
            n_samples = len(raw) // 2
            if n_samples == 0:
                return 0.0
            samples = struct.unpack(f"<{n_samples}h", raw)
            rms = (sum(s * s for s in samples) / n_samples) ** 0.5
            return rms / 32768.0
    except Exception as exc:
        logger.warning("Could not compute audio RMS: %s", exc)
        return 0.0


async def fetch_audio_clip(
    camera_id: str,
    start_time: datetime,
    end_time: datetime,
) -> bytes | None:
    """
    Fetch a video clip from Protect and extract 16 kHz mono WAV audio via ffmpeg.
    camera_id can be a UUID or a MAC address.
    """
    try:
        client = await get_protect_client()
        camera = find_camera(client, camera_id)

        if not camera:
            logger.error("Camera %s not found (tried UUID and MAC)", camera_id)
            logger.info(
                "Available cameras: %s",
                [(c.name, c.mac, c.id) for c in client.bootstrap.cameras.values()],
            )
            return None

        logger.info(
            "Fetching clip from %s (%s to %s)",
            camera.name,
            start_time.isoformat(),
            end_time.isoformat(),
        )

        video_data: bytes | None = None
        try:
            if hasattr(camera, "get_video"):
                video_data = await camera.get_video(start_time, end_time)
            elif hasattr(camera, "export_video"):
                video_data = await camera.export_video(start_time, end_time)
            else:
                video_methods = [
                    m for m in dir(camera) if "video" in m.lower() or "export" in m.lower()
                ]
                logger.error("No video export method found. Available: %s", video_methods)
                return None
        except TypeError as exc:
            logger.warning("Video method TypeError (%s), trying output_file fallback", exc)
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                if hasattr(camera, "get_video"):
                    await camera.get_video(start_time, end_time, output_file=tmp_path)
                    video_data = tmp_path.read_bytes()
                    tmp_path.unlink(missing_ok=True)
            except Exception as exc2:
                logger.error("Fallback also failed: %s", exc2)
                raise

        if not video_data:
            logger.error("No video data received from Protect")
            return None

        logger.info("Received %d bytes of video data", len(video_data))
        return await _extract_audio(video_data)

    except Exception:
        logger.exception("Error fetching audio clip for camera %s", camera_id)
        inc("nvr_fetch_errors_total")
        return None


async def _extract_audio(video_data: bytes) -> bytes | None:
    """Run ffmpeg to extract 16 kHz mono WAV from raw video bytes (non-blocking)."""
    # Create a placeholder file FIRST so cleanup is guaranteed even if write fails.
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)  # noqa: SIM115 — manual close below
    video_path = Path(tmp.name)
    try:
        try:
            tmp.write(video_data)
        finally:
            tmp.close()
    except Exception:
        video_path.unlink(missing_ok=True)
        raise

    audio_path = video_path.with_suffix(".wav")

    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-af",
            "highpass=f=200,loudnorm",
            str(audio_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            logger.error("ffmpeg timeout extracting audio")
            inc("ffmpeg_failures_total")
            return None

        if proc.returncode != 0:
            logger.error(
                "ffmpeg error (rc=%d): %s", proc.returncode, stderr.decode("utf-8", "replace")
            )
            inc("ffmpeg_failures_total")
            return None

        if not audio_path.exists() or audio_path.stat().st_size == 0:
            logger.error("ffmpeg produced empty audio file")
            inc("ffmpeg_failures_total")
            return None

        audio_bytes = audio_path.read_bytes()
        logger.info("Extracted %d bytes of audio", len(audio_bytes))
        return audio_bytes

    finally:
        video_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)


_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def _normalize(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    out = _PUNCT_RE.sub(" ", text.lower()).strip()
    return re.sub(r"\s+", " ", out)


# Earlier versions sent this as a prompt; Whisper models echo it back on
# silence, so treat it as a hallucination if it ever shows up in output.
_PROMPT = "Dette er en optagelse fra et overvågningskamera i et privat hjem."

# Common Whisper hallucinations across Danish / English / German.
# Stored normalized (lower-cased, no punctuation, single-spaced) so the
# matcher can compare directly against normalised transcript text.
_HALLUCINATION_PHRASES = frozenset(
    _normalize(p)
    for p in (
        # Danish — sub-title / sign-off filler
        "tak for at se med",
        "tak fordi du så med",
        "tak fordi i så med",
        "tekstning af nicolai winther",
        "undertekster af nicolai winther",
        # English
        "thanks for watching",
        "subscribe to my channel",
        "subtitles by the amara.org community",
        "thank you for watching",
        # German
        "untertitel der amara.org-community",
        "untertitel von stephanie geiges",
        # Echo of our own prompt
        _PROMPT,
        # Generic
        "you",
        ".",
    )
)


def _is_hallucination(text: str) -> bool:
    """
    Detect Whisper hallucinations.

    Two heuristics:
    1. The same 2-5 word n-gram repeating 4+ times consecutively.
    2. The transcription is one of the well-known boilerplate hallucinations
       (sign-offs, "thanks for watching", subtitle credits) repeating or alone.
    """
    if not text:
        return False

    stripped = _normalize(text)
    if not stripped:
        return False

    # 1. Whole transcription is a known hallucination phrase
    if stripped in _HALLUCINATION_PHRASES:
        return True

    # 1b. Repeated occurrence of a known phrase (e.g. "tak for at se med tak for at se med")
    for phrase in _HALLUCINATION_PHRASES:
        if not phrase or " " not in phrase:
            continue
        # Phrase appears at least twice and dominates the transcript
        occurrences = stripped.count(phrase)
        if occurrences >= 2:
            covered = occurrences * len(phrase)
            if covered / max(len(stripped), 1) > 0.6:
                return True

    if len(text) < 20:
        return False
    words = stripped.split()
    if len(words) < 6:
        return False

    # 2. Consecutive repeating n-grams
    for n in range(2, 6):
        for i in range(len(words) - n * 3):
            phrase_t = tuple(words[i : i + n])
            repeats = 1
            j = i + n
            while j + n <= len(words) and tuple(words[j : j + n]) == phrase_t:
                repeats += 1
                j += n
            if repeats >= 4:
                return True
    return False


# faster-whisper's default thresholds. speaches doesn't let clients set them, so
# they are applied per segment here: loops and noise segments are dropped
# instead of throwing away the whole clip.
_NO_SPEECH_PROB = 0.6
_LOGPROB_FLOOR = -1.0
_MAX_COMPRESSION = 2.4


def _drop_bad_segments(result: dict) -> dict:
    """Remove looping (high compression) or no-speech segments and rebuild the text."""
    segments = result.get("segments") or []
    keep = [
        s
        for s in segments
        if not (
            float(s.get("compression_ratio") or 0) > _MAX_COMPRESSION
            or (
                float(s.get("no_speech_prob") or 0) > _NO_SPEECH_PROB
                and float(s.get("avg_logprob") or 0) < _LOGPROB_FLOOR
            )
        )
    ]
    if len(keep) == len(segments):
        return result
    logger.info("Dropped %d of %d bad segments", len(segments) - len(keep), len(segments))
    text = " ".join(str(s.get("text") or "").strip() for s in keep).strip()
    return {**result, "segments": keep, "text": text}


def _collapse_repeats(text: str, min_repeats: int = 4) -> str:
    """Collapse runs of the same 1-5 word phrase repeated *min_repeats*+ times to one copy."""
    words = text.split()
    norm = [_normalize(w) for w in words]
    out: list[str] = []
    i = 0
    while i < len(words):
        unit_len, end = 0, i + 1
        for n in range(1, 6):
            unit = norm[i : i + n]
            if len(unit) < n:
                break
            j = i + n
            while norm[j : j + n] == unit:
                j += n
            if (j - i) // n >= min_repeats:
                unit_len, end = n, j
                break
        out.extend(words[i : i + unit_len] if unit_len else words[i : i + 1])
        i = end
    return " ".join(out)


async def _post_with_5xx_retry(
    client: httpx.AsyncClient,
    audio_data: bytes,
    data: dict,
) -> httpx.Response:
    """
    POST to speaches with a single retry on 5xx (transient upstream blip).

    Uses a small backoff so we don't hammer a struggling speaches pod.
    """
    last_response: httpx.Response | None = None
    for attempt in (1, 2):
        files = {"file": ("audio.wav", audio_data, "audio/wav")}
        try:
            resp = await client.post(
                f"{WHISPER_URL}/v1/audio/transcriptions",
                files=files,
                data=data,
            )
        except httpx.RequestError as exc:
            if attempt == 2:
                raise
            logger.warning("Whisper request error on attempt %d: %s — retrying", attempt, exc)
            await asyncio.sleep(2)
            continue
        last_response = resp
        if 500 <= resp.status_code < 600 and attempt == 1:
            logger.warning(
                "Whisper returned %d (attempt %d); retrying after 2s",
                resp.status_code,
                attempt,
            )
            inc("transcription_5xx_retries_total")
            await asyncio.sleep(2)
            continue
        return resp
    assert last_response is not None
    return last_response


async def _post_transcription(
    client: httpx.AsyncClient,
    audio_data: bytes,
    model: str,
    data: dict,
) -> dict:
    """POST audio to speaches; auto-download model on 404 then retry once."""
    response = await _post_with_5xx_retry(client, audio_data, data)

    if response.status_code == 404:
        try:
            detail = response.json().get("detail", "")
        except Exception:
            detail = response.text
        if "not installed" in detail:
            logger.info("Model %s not installed — attempting download via speaches…", model)
            dl = await client.post(
                f"{WHISPER_URL}/v1/models/{model}",
                timeout=600.0,
            )
            if dl.status_code not in (200, 201):
                logger.error(
                    "Model %s could not be downloaded (speaches %d). "
                    "Set WHISPER_MODEL=%s on the speaches deployment to pre-install it.",
                    model,
                    dl.status_code,
                    model,
                )
                return {
                    "error": (
                        f"Model '{model}' is not installed on the speaches server and "
                        f"could not be downloaded automatically. "
                        f"Set WHISPER_MODEL={model} in the speaches deployment env vars."
                    )
                }
            logger.info("Model downloaded, retrying transcription…")
            response = await _post_with_5xx_retry(client, audio_data, data)

    if response.status_code != 200:
        logger.error("Whisper API error %d: %s", response.status_code, response.text)
        return {"error": response.text}

    return response.json()


async def transcribe_audio(audio_data: bytes) -> dict:
    """
    Submit audio to the Whisper (speaches) server and return the parsed JSON.

    Returns a dict with an 'error' key on failure, 'empty' when no speech was
    found, or 'hallucination' (+ 'raw_text') when only looping/boilerplate
    output remained after a retry and loop-collapsing.
    """
    settings = get_settings()
    model = settings.get("whisper_model", "Systran/faster-whisper-large-v3")
    language = settings.get("language", "da")
    vad_filter = settings.get("vad_filter", "true").lower() == "true"

    # Only send fields speaches' /v1/audio/transcriptions form accepts
    # (model, language, prompt, response_format, temperature, vad_filter, ...).
    # Unknown fields are silently dropped by the server. No `prompt`: tested on
    # real clips, the Danish fine-tunes echo it or invent text from it.
    base_data: dict = {
        "model": model,
        "response_format": "verbose_json",
        "temperature": "0.0",
        "vad_filter": str(vad_filter).lower(),
    }
    if language and language != "auto":
        base_data["language"] = language

    client = get_whisper_client()
    try:
        logger.info("Transcribing model=%s lang=%s vad=%s", model, language, vad_filter)

        result = await _post_transcription(client, audio_data, model, base_data)
        if "error" in result:
            inc("transcription_errors_total")
            return result

        result = _drop_bad_segments(result)
        text = (result.get("text") or "").strip()
        if not text:
            logger.info("Transcription returned empty text; treating as silence")
            inc("transcriptions_empty_total")
            return {"empty": True, **result, "text": ""}

        if not _is_hallucination(text):
            inc("transcriptions_succeeded_total")
            logger.info("Transcription: %s...", text[:100])
            return result

        logger.warning("Hallucination detected, retrying: %r", text[:120])
        inc("hallucinations_detected_total")
        # speaches passes a single temperature (no fallback ladder), so retry
        # once with some sampling temperature to break the repetition.
        strict_data = {**base_data, "temperature": "0.4"}
        retry = await _post_transcription(client, audio_data, model, strict_data)
        if "error" in retry:
            inc("transcription_errors_total")
            return retry
        retry = _drop_bad_segments(retry)
        retry_text = (retry.get("text") or "").strip()
        if retry_text and not _is_hallucination(retry_text):
            inc("hallucination_retry_succeeded_total")
            logger.info("Retry succeeded: %s...", retry_text[:100])
            return retry

        # Still looping: keep whatever real speech is left once the loops are
        # collapsed, rather than discarding the whole clip.
        for candidate, res in ((retry_text, retry), (text, result)):
            cleaned = _collapse_repeats(candidate)
            if len(cleaned.split()) >= 3 and not _is_hallucination(cleaned):
                inc("hallucination_cleaned_total")
                logger.info("Kept text after collapsing loops: %s...", cleaned[:100])
                return {**res, "text": cleaned, "segments": []}

        inc("hallucinations_persisted_total")
        logger.warning("Hallucination persisted after retry, discarding: %r", retry_text[:120])
        return {"hallucination": True, "raw_text": retry_text or text}

    except Exception as exc:
        inc("transcription_errors_total")
        logger.exception("Error calling Whisper API: %s", exc)
        return {"error": str(exc)}


_SAFE_CAMERA_NAME = re.compile(r"[^A-Za-z0-9._\- ]+")


def save_audio_file(audio_data: bytes, event_time: datetime, camera_name: str) -> str:
    """Persist audio bytes to AUDIO_PATH and return the filename."""
    Path(AUDIO_PATH).mkdir(parents=True, exist_ok=True)
    audio_hash = hashlib.md5(audio_data).hexdigest()[:8]
    safe_camera = _SAFE_CAMERA_NAME.sub("_", camera_name or "camera")
    filename = f"{event_time.strftime('%Y%m%d_%H%M%S')}_{safe_camera}_{audio_hash}.wav"
    (Path(AUDIO_PATH) / filename).write_bytes(audio_data)
    return filename
