import json
import zipfile

from orbit.jobs import create_job_bundle


def test_remote_job_bundle_is_self_contained(tmp_path):
    bundle = create_job_bundle(tmp_path, "1b", 10, 2, 128, 3e-4, "Orbit data", data_language="bilingual")
    assert bundle.exists()
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        assert any(name.endswith("/job.json") for name in names)
        assert any(name.endswith("/dataset.txt") for name in names)
        assert any(name.endswith("/orbit_project/orbit/train.py") for name in names)
        job_name = next(name for name in names if name.endswith("/job.json"))
        job = json.loads(archive.read(job_name))
        assert job["preset"] == "1b"
        assert job["data_language"] == "bilingual"


def test_ai_assisted_remote_bundle_keeps_api_key_out(tmp_path):
    bundle = create_job_bundle(
        tmp_path, "300m", 2, 1, 32, 3e-4, "fallback text",
        assistant={
            "provider": "deepseek", "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat", "instruction": "Teach Python safely",
            "examples": 4, "language": "bilingual", "api_key": "must-not-be-bundled",
        },
    )
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        teacher_name = next(name for name in names if name.endswith("/teacher.json"))
        teacher = json.loads(archive.read(teacher_name))
        run_sh = archive.read(next(name for name in names if name.endswith("/run.sh"))).decode()
        assert "api_key" not in teacher
        assert "must-not-be-bundled" not in archive.read(teacher_name).decode()
        assert "orbit.remote_assist" in run_sh
