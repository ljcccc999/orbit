from orbit.memory import LongTermMemory


def test_local_memory_supports_manual_and_assistant_entries(tmp_path):
    memory = LongTermMemory(tmp_path)
    manual = memory.add("Tim prefers concise Chinese answers.")
    automatic = memory.consider("请记住：复杂的新技术问题执行前先搜索权威资料")

    assert manual["source"] == "manual"
    assert automatic is not None
    assert automatic["source"] == "assistant"
    assert "权威资料" in memory.system_context()


def test_local_memory_skips_short_lived_requests_and_redacts_secrets(tmp_path):
    memory = LongTermMemory(tmp_path)

    assert memory.consider("帮我把按钮改成蓝色") is None
    row = memory.consider("请记住 api key: sk-abcdefghijklmnop")

    assert row is not None
    assert "sk-abcdefghijklmnop" not in row["content"]
    assert "已隐藏" in row["content"]
