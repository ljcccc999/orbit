from orbit.settings import OrbitSettings


def test_operational_settings_persist_with_safe_defaults(tmp_path):
    settings = OrbitSettings(tmp_path)
    assert settings.get() == {
        "auto_update": False,
        "prevent_sleep": False,
        "background_service": True,
        "computer_control": False,
    }
    settings.update({"prevent_sleep": True, "background_service": False, "computer_control": True})
    assert OrbitSettings(tmp_path).get() == {
        "auto_update": False,
        "prevent_sleep": True,
        "background_service": False,
        "computer_control": True,
    }
