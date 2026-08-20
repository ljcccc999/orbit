import threading

from orbit import teacher


def test_teacher_generation_tracks_progress_and_usage(monkeypatch):
    calls = []

    def fake_request(endpoint, api_key, body, attempts=3):
        calls.append((endpoint, api_key, body))
        return {
            "choices": [{"message": {"content": "<|user|>Q\n<|assistant|>A"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

    monkeypatch.setattr(teacher, "_request", fake_request)
    progress = []
    text, usage = teacher.generate_dataset(
        teacher.TeacherConfig(instruction="Teach carefully", examples=6, model_profile={"preset": "1b", "parameters": 1_063_000_000}),
        "secret", threading.Event(), lambda current, total: progress.append((current, total)),
    )
    assert "<|assistant|>A" in text
    assert progress == [(5, 6), (6, 6)]
    assert usage["total_tokens"] == 10
    assert all(call[1] == "secret" for call in calls)
    assert all(call[2]["model"] == "deepseek-v4-flash" for call in calls)
    assert "1063000000" in calls[0][2]["messages"][1]["content"]


def test_teacher_rejects_insecure_remote_http():
    config = teacher.TeacherConfig(base_url="http://example.com", instruction="x")
    try:
        config.validate()
    except ValueError as exc:
        assert "HTTPS" in str(exc)
    else:
        raise AssertionError("remote HTTP should be rejected")
