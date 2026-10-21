# call-transcriber

Turns meeting recordings into Markdown notes: a short summary, action items and a verbatim transcript split by speaker. One script, one dependency, about $0.20 per hour of audio.

I run 3–6 client calls a week and the notes used to be the bottleneck: either I wrote them during the call and listened worse, or I didn't write them and lost the details. This script picks up every new recording, sends it to an audio-capable model and drops a ready note into my Obsidian vault.

```text
recording (mp4/webm/any)
  └─ ffmpeg: cut into 15-min pieces, mono mp3 32 kbps
      └─ OpenRouter → Gemini (audio input), one request per piece
          └─ retry until finish_reason == "stop"
              └─ Markdown: front matter + summary + action items + transcript
```

## Example output

```markdown
---
date: 2026-10-20 22:27
duration_min: 68
model: google/gemini-3.5-flash
---

# Часть 1

## Саммари
Обсуждали опыт собеседований на позиции Middle+/Senior и подготовку резюме ...

## Action items
- Подготовить резюме и отправить на ревью
- ...

## Транскрипт
**[00:08] Спикер 1:** ...
**[00:13] Спикер 2:** ...
```

The prompt is in Russian because my calls are; change `PROMPT` in `transcribe.py` for any other language.

## Run

Requires Python 3.10+ and `ffmpeg` (in `PATH` or via the `FFMPEG` variable).

```bash
pip install -r requirements.txt
cp .env.example .env          # put your OpenRouter key there
python transcribe.py                      # every new recording in RECORDINGS_DIR
python transcribe.py path/to/folder       # one Meetily recording folder
python transcribe.py path/to/call.webm    # any file ffmpeg can read
```

`RECORDINGS_DIR` defaults to the folder where [Meetily](https://github.com/Zackriya-Solutions/meeting-minutes) stores recordings. A recording is skipped if its note already exists in `OUTPUT_DIR`.

Tests: `pip install -r requirements-dev.txt && pytest`.

## Things that were not obvious

- **Chunking.** Gemini returned empty content for ~30-minute audio, so recordings are cut into 15-minute pieces and each piece is told where it starts. Timecodes restart per part, which keeps the model's timestamps honest.
- **Reasoning tokens.** With default settings the model spent most of the output budget on reasoning and `content` came back `None`. `reasoning.effort = minimal` fixes it.
- **Unfinished answers.** Sometimes the text stops mid-sentence with `finish_reason: error`. Only a response with `stop` is accepted, otherwise the piece is retried.
- **No ffprobe.** The ffmpeg bundled with Meetily ships without ffprobe, so duration is parsed from `ffmpeg -i` output.
- **Time zones.** Meetily writes `created_at` in UTC; file names use local time.

## Limitations

- Speakers are labelled `Спикер 1/2`, not by name, and numbering may change between parts of a long call.
- Audio goes to a third-party API. Don't use it for calls you are not allowed to share.
- No parallelism: parts are processed one after another (a 1-hour call takes 3–5 minutes).

## License

MIT
