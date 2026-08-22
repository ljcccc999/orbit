import threading

from orbit import teacher


def test_teacher_generation_tracks_progress_and_usage(monkeypatch):
    calls = []

    def fake_request(endpoint, api_key, body, attempts=3, stop_event=None):
        calls.append((endpoint, api_key, body))
        return {
            "choices": [{"message": {"content": "<|user|>问题 Q\n<|assistant|>回答 A"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

    monkeypatch.setattr(teacher, "_request", fake_request)
    progress = []
    text, usage = teacher.generate_dataset(
        teacher.TeacherConfig(instruction="Teach carefully", examples=6, language="bilingual", model_profile={"preset": "1b", "parameters": 1_063_000_000}),
        "secret", threading.Event(), lambda current, total: progress.append((current, total)),
    )
    assert "<|assistant|>回答 A" in text
    assert progress == [(5, 6), (6, 6)]
    assert usage["total_tokens"] == 10
    assert all(call[1] == "secret" for call in calls)
    assert all(call[2]["model"] == "deepseek-v4-flash" for call in calls)
    assert "1063000000" in calls[0][2]["messages"][1]["content"]
    assert "简体中文" in calls[0][2]["messages"][1]["content"]
    assert "English" in calls[0][2]["messages"][1]["content"]
    assert "Orbit" in calls[0][2]["messages"][1]["content"]
    assert "YUNSH" in calls[0][2]["messages"][1]["content"]
    assert "你是谁" in text
    assert "由 YUNSH 开发" in text
    assert "identity_training_rule" not in text


def test_teacher_rejects_insecure_remote_http():
    config = teacher.TeacherConfig(base_url="http://example.com", instruction="x")
    try:
        config.validate()
    except ValueError as exc:
        assert "HTTPS" in str(exc)
    else:
        raise AssertionError("remote HTTP should be rejected")


def test_teacher_retries_empty_content(monkeypatch):
    calls = []

    def fake_request(endpoint, api_key, body, attempts=3, stop_event=None):
        calls.append(1)
        content = "" if len(calls) == 1 else "<|user|>Q\n<|assistant|>A"
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(teacher, "_request", fake_request)
    text, _ = teacher.generate_dataset(
        teacher.TeacherConfig(instruction="x", examples=1), "secret", threading.Event()
    )
    assert len(calls) == 2
    assert "<|assistant|>A" in text


def test_teacher_reports_each_completed_chunk_for_durable_saving(monkeypatch):
    def fake_request(endpoint, api_key, body, attempts=3, stop_event=None):
        return {
            "choices": [{"message": {"content": "中文 sample in English"}}],
            "usage": {"total_tokens": 1},
        }

    monkeypatch.setattr(teacher, "_request", fake_request)
    saved = []
    teacher.generate_dataset(
        teacher.TeacherConfig(instruction="chat", examples=6, language="bilingual"),
        "secret",
        threading.Event(),
        chunk_callback=lambda content, current, total: saved.append((content, current, total)),
    )

    assert [row[1:] for row in saved] == [(5, 6), (6, 6)]
    assert all(row[0] == "中文 sample in English" for row in saved)
