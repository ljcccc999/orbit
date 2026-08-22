import threading

from orbit import teacher


def _samples(count: int, prefix: str = "sample") -> str:
    return teacher.serialize_samples([
        f"标题 {prefix}-{index}\n这是第 {index} 条经过边界标记的中文训练文献，包含足够长的事实说明、定义、例子、反例、适用边界和总结。"
        f"{chr(0x4e00 + (sum(map(ord, prefix)) * 31 + index * 97) % 2000) * 240}"
        f" English explanation for unique {prefix} item {index} includes definitions, examples, boundaries, and a concise summary."
        for index in range(count)
    ])


def test_teacher_generation_tracks_progress_and_usage(monkeypatch):
    calls = []

    def fake_request(endpoint, api_key, body, attempts=3, stop_event=None):
        calls.append((endpoint, api_key, body))
        count = 5 if len(calls) == 1 else 1
        return {
            "choices": [{"message": {"content": _samples(count, str(len(calls)))}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

    monkeypatch.setattr(teacher, "_request", fake_request)
    progress = []
    text, usage = teacher.generate_dataset(
        teacher.TeacherConfig(instruction="Teach carefully", examples=6, language="bilingual", model_profile={"preset": "1b", "parameters": 1_063_000_000}),
        "secret", threading.Event(), lambda current, total: progress.append((current, total)),
    )
    assert teacher.SAMPLE_START in text
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
        content = "" if len(calls) == 1 else _samples(1, "retry")
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(teacher, "_request", fake_request)
    text, _ = teacher.generate_dataset(
        teacher.TeacherConfig(instruction="x", examples=1), "secret", threading.Event()
    )
    assert len(calls) == 2
    assert "retry" in text


def test_teacher_reports_each_completed_chunk_for_durable_saving(monkeypatch):
    def fake_request(endpoint, api_key, body, attempts=3, stop_event=None):
        count = 5 if not saved else 1
        return {
            "choices": [{"message": {"content": _samples(count, f"batch-{len(saved)}")}}],
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
    assert all(teacher.SAMPLE_START in row[0] for row in saved)


def test_teacher_rejects_unbounded_or_duplicate_samples(monkeypatch):
    calls = []

    def fake_request(endpoint, api_key, body, attempts=3, stop_event=None):
        calls.append(1)
        if len(calls) == 1:
            return {"choices": [{"message": {"content": "没有严格样本边界的普通段落"}}]}
        return {"choices": [{"message": {"content": _samples(1, "valid")}}]}

    monkeypatch.setattr(teacher, "_request", fake_request)
    text, _ = teacher.generate_dataset(
        teacher.TeacherConfig(instruction="x", examples=1, language="bilingual"),
        "secret", threading.Event(),
    )
    assert len(calls) == 2
    assert len(teacher.extract_samples(text)) == 1


def test_generated_corpus_has_deterministic_validation_split():
    corpus = teacher.ORBIT_TRAINING_ANCHOR + "\n\n" + _samples(20, "split")
    train, validation, stats = teacher.split_generated_corpus(corpus)
    assert stats["samples"] == 20
    assert stats["training_samples"] + stats["validation_samples"] == 20
    assert stats["validation_samples"] >= 1
    assert teacher.ORBIT_TRAINING_ANCHOR in train
    assert teacher.COMMUNICATION_TAIL in train
    assert teacher.ORBIT_TRAINING_ANCHOR not in validation
