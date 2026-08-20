import json
import zipfile

from orbit.jobs import create_job_bundle


def test_remote_job_bundle_is_self_contained(tmp_path):
    bundle = create_job_bundle(tmp_path, "1b", 10, 2, 128, 3e-4, "Orbit data")
    assert bundle.exists()
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        assert any(name.endswith("/job.json") for name in names)
        assert any(name.endswith("/dataset.txt") for name in names)
        assert any(name.endswith("/orbit_project/orbit/train.py") for name in names)
        job_name = next(name for name in names if name.endswith("/job.json"))
        assert json.loads(archive.read(job_name))["preset"] == "1b"
