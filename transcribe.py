# -*- coding: utf-8 -*-
"""
Call transcriber: turns meeting recordings into Markdown notes
(summary + action items + verbatim transcript with speakers) via an
audio-capable LLM on OpenRouter (Gemini by default).

Usage:
    python transcribe.py              # every new recording in RECORDINGS_DIR
    python transcribe.py <folder>     # one Meetily recording folder
    python transcribe.py <file>       # any audio/video file ffmpeg can read

Long recordings are cut into CHUNK_MINUTES pieces: in practice Gemini
returns an empty answer for ~30-minute inputs, 15 minutes is reliable.
Already processed recordings are skipped (by output file name).

Configuration comes from environment variables or a .env file next to
this script, see .env.example.
"""

import base64
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ENV_FILE = Path(__file__).parent / ".env"


def load_env(path: Path = ENV_FILE) -> None:
    """Minimal .env loader: KEY=VALUE lines, real environment wins."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$", line)
        if m and m.group(1) not in os.environ:
            os.environ[m.group(1)] = m.group(2).strip().strip('"')


load_env()

# --- config ---------------------------------------------------------------

RECORDINGS_DIR = Path(os.getenv("RECORDINGS_DIR", Path.home() / "Music" / "meetily-recordings"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", Path(__file__).parent / "out"))
FFMPEG = os.getenv("FFMPEG", "ffmpeg")

MODEL = os.getenv("MODEL", "google/gemini-3.5-flash")
API_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
CHUNK_MINUTES = int(os.getenv("CHUNK_MINUTES", "15"))
REQUEST_TIMEOUT = 600
MAX_ATTEMPTS = 3

PROMPT = """Это запись созвона (микрофон + системный звук смешаны в одну дорожку).
Сделай по ней документ на русском языке строго в таком формате:

## Саммари
2-5 предложений: о чём говорили, к чему пришли.

## Action items
Маркированный список договорённостей и задач с ответственными, если понятны.
Если их нет — напиши «Нет».

## Транскрипт
Полная дословная расшифровка. Каждая реплика с новой строки в формате:
**[MM:SS] Спикер N:** текст реплики

Разделяй спикеров по голосам. Речь передавай дословно, но без слов-паразитов.
Если часть неразборчива — пометь [неразборчиво]. Ничего не выдумывай.
"""

# --- helpers --------------------------------------------------------------


def api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        sys.exit(f"No API key: set OPENROUTER_API_KEY in the environment or in {ENV_FILE}")
    return key


def parse_duration(ffmpeg_stderr: str) -> float:
    """Duration from `ffmpeg -i` output (works where ffprobe is not shipped)."""
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", ffmpeg_stderr)
    if not m:
        raise ValueError("ffmpeg output has no Duration line")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def audio_duration_seconds(path: Path) -> float:
    proc = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True, errors="replace")
    try:
        return parse_duration(proc.stderr)
    except ValueError:
        sys.exit(f"ffmpeg could not read {path}")


def chunk_count(duration: float, chunk_seconds: int) -> int:
    """How many pieces a recording is cut into (ceil, at least one)."""
    return max(1, -(-int(duration) // chunk_seconds))


def encode_chunk(src: Path, start: float, length: float, dst: Path) -> None:
    """Audio piece -> mono mp3 at 32 kbps: enough for speech, small base64 payload."""
    subprocess.run(
        [FFMPEG, "-y", "-ss", str(start), "-t", str(length), "-i", str(src),
         "-ac", "1", "-b:a", "32k", "-f", "mp3", str(dst)],
        check=True, capture_output=True,
    )


def transcribe_chunk(key: str, mp3: Path, part: str) -> str:
    audio_b64 = base64.b64encode(mp3.read_bytes()).decode()
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT + (f"\n\n{part}" if part else "")},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
            ],
        }],
        # without this, reasoning tokens eat the budget and content comes back None
        "reasoning": {"effort": "minimal"},
        "max_tokens": 32000,
    }
    # Gemini sometimes returns empty content or cuts the text with finish_reason=error:
    # accept only a finished answer, retry otherwise
    for attempt in range(1, MAX_ATTEMPTS + 1):
        resp = requests.post(API_URL, headers={"Authorization": f"Bearer {key}"},
                             json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"OpenRouter: {data['error']}")
        choice = data["choices"][0]
        content = (choice["message"].get("content") or "").strip()
        if content and choice.get("finish_reason") == "stop":
            return content
        print(f"    attempt {attempt}: finish_reason={choice.get('finish_reason')}, "
              f"{len(content)} chars, retrying", flush=True)
        time.sleep(10)
    raise RuntimeError(f"Model did not return a finished text in {MAX_ATTEMPTS} attempts")


def sanitize(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "-", name).strip()


def output_path(title: str, when: datetime) -> Path:
    return OUTPUT_DIR / f"{when:%Y-%m-%d %H-%M} — {sanitize(title)}.md"


def join_parts(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return "\n\n---\n\n".join(f"# Часть {i + 1}\n\n{p}" for i, p in enumerate(parts))


def transcribe_file(key: str, audio: Path, title: str, when: datetime) -> Path:
    duration = audio_duration_seconds(audio)
    chunk_len = CHUNK_MINUTES * 60
    n_chunks = chunk_count(duration, chunk_len)

    parts = []
    tmp = audio.parent / "_chunk.mp3"
    try:
        for i in range(n_chunks):
            part_note = (
                f"Это часть {i + 1} из {n_chunks} длинной записи, "
                f"начинается с {i * CHUNK_MINUTES}-й минуты. Таймкоды считай от начала этой части."
                if n_chunks > 1 else ""
            )
            print(f"  transcribing{f' part {i + 1}/{n_chunks}' if n_chunks > 1 else ''}...")
            encode_chunk(audio, i * chunk_len, chunk_len, tmp)
            parts.append(transcribe_chunk(key, tmp, part_note))
    finally:
        tmp.unlink(missing_ok=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path(title, when)
    header = (
        f"---\n"
        f"date: {when:%Y-%m-%d %H:%M}\n"
        f"source: {audio}\n"
        f"duration_min: {duration / 60:.0f}\n"
        f"model: {MODEL}\n"
        f"---\n\n"
    )
    out.write_text(header + join_parts(parts) + "\n", encoding="utf-8")
    return out


# --- Meetily recordings ---------------------------------------------------


def meetily_meta(folder: Path) -> tuple[str, datetime]:
    """Title and start time from Meetily's metadata.json (created_at is UTC)."""
    title, when = folder.name, datetime.fromtimestamp(folder.stat().st_mtime)
    meta_file = folder / "metadata.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        title = meta.get("meeting_name") or title
        try:
            when = datetime.fromisoformat(meta["created_at"]).astimezone()
        except (KeyError, ValueError):
            pass
    return title, when


def process_meetily_folder(key: str, folder: Path) -> None:
    audio = folder / "audio.mp4"
    if not audio.exists():
        print(f"- {folder.name}: no audio.mp4, skipping")
        return
    title, when = meetily_meta(folder)
    if output_path(title, when).exists():
        print(f"- {title}: already done")
        return
    print(f"+ {title} ({when:%d.%m %H:%M})")
    print(f"  -> {transcribe_file(key, audio, title, when)}")


def main() -> None:
    key = api_key()
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if target.is_dir():
            process_meetily_folder(key, target)
        else:
            print(f"  -> {transcribe_file(key, target, target.stem, datetime.now())}")
        return

    if not RECORDINGS_DIR.exists():
        sys.exit(f"No recordings folder {RECORDINGS_DIR}")
    for folder in sorted(RECORDINGS_DIR.iterdir()):
        if folder.is_dir():
            process_meetily_folder(key, folder)


if __name__ == "__main__":
    main()
