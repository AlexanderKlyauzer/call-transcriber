import json
from datetime import datetime, timezone

import pytest

import transcribe as t


@pytest.mark.parametrize("duration,expected", [
    (0, 1), (59, 1), (900, 1), (901, 2), (4061.7, 5), (5400, 6),
])
def test_chunk_count(duration, expected):
    assert t.chunk_count(duration, 900) == expected


def test_parse_duration():
    stderr = "Input #0, mov,mp4 ...\n  Duration: 01:07:41.71, start: 0.000000, bitrate: 148 kb/s\n"
    assert t.parse_duration(stderr) == pytest.approx(4061.71)


def test_parse_duration_missing():
    with pytest.raises(ValueError):
        t.parse_duration("No such file or directory")


def test_sanitize_strips_forbidden_chars():
    assert t.sanitize(' Call: client/agency? "v2" ') == "Call- client-agency- -v2-"


def test_join_parts_single_and_many():
    assert t.join_parts(["only"]) == "only"
    joined = t.join_parts(["a", "b"])
    assert joined.startswith("# Часть 1\n\na") and "# Часть 2\n\nb" in joined


def test_meetily_meta_reads_utc_created_at(tmp_path):
    folder = tmp_path / "Meeting 2026-10-20"
    folder.mkdir()
    (folder / "metadata.json").write_text(json.dumps({
        "meeting_name": "Weekly sync", "created_at": "2026-10-20T17:27:34.949786+00:00",
    }), encoding="utf-8")
    title, when = t.meetily_meta(folder)
    assert title == "Weekly sync"
    assert when.astimezone(timezone.utc) == datetime(2026, 10, 20, 17, 27, 34, 949786, tzinfo=timezone.utc)


def test_load_env_does_not_override_real_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('CHUNK_MINUTES=7\nMODEL="some/model"\n', encoding="utf-8")
    monkeypatch.setenv("CHUNK_MINUTES", "20")
    monkeypatch.delenv("MODEL", raising=False)
    t.load_env(env)
    import os
    assert os.environ["CHUNK_MINUTES"] == "20"
    assert os.environ["MODEL"] == "some/model"
