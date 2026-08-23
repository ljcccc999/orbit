import subprocess

from orbit import web


def test_sleep_inhibitor_uses_caffeinate_and_stops_it(monkeypatch):
    started = []

    class Process:
        def terminate(self):
            started.append("terminate")

        def wait(self, timeout):
            started.append(("wait", timeout))

    monkeypatch.setattr(web.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(web.Path, "is_file", lambda path: True)

    def fake_popen(command, **kwargs):
        started.append(command)
        return Process()

    monkeypatch.setattr(web.subprocess, "Popen", fake_popen)
    inhibitor = web.SleepInhibitor()
    inhibitor.set_enabled(True)
    inhibitor.set_enabled(False)

    assert started[0][:2] == ["/usr/bin/caffeinate", "-dimsu"]
    assert started[-2:] == ["terminate", ("wait", 3)]
