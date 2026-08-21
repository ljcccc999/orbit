import json
import zipfile

import pytest

from orbit.community import CommunityStore


def contribution(**overrides):
    payload = {
        "project": "shared-orbit",
        "title": "Helpful bilingual idea",
        "contributor": "Tester",
        "language": "bilingual",
        "content_type": "idea",
        "source_url": "",
        "license": "CC0-1.0",
        "content": "中文训练想法。 A useful English training idea for Orbit.",
        "rights_ack": True,
        "policy_ack": True,
        "privacy_ack": True,
    }
    payload.update(overrides)
    return payload


def test_review_export_import_and_approved_dataset(tmp_path):
    store = CommunityStore(tmp_path)
    submitted = store.submit(contribution())
    assert submitted["status"] == "pending_review"
    approved = store.review(submitted["id"], "approved", "Reviewer", "Checked", False)
    assert approved["status"] == "approved"
    dataset = store.approved_dataset("shared-orbit")
    assert dataset["count"] == 1
    assert "useful English" in dataset["text"]

    package = store.export(submitted["id"])
    imported_store = CommunityStore(tmp_path / "other")
    imported = imported_store.import_package(package.read_bytes())
    assert imported["status"] == "pending_review"
    assert imported["imported_from"] == submitted["id"]


def test_factual_contribution_requires_source_and_human_verification(tmp_path):
    store = CommunityStore(tmp_path)
    with pytest.raises(ValueError, match="HTTPS"):
        store.submit(contribution(content_type="factual"))
    row = store.submit(contribution(content_type="factual", source_url="https://example.org/source"))
    with pytest.raises(ValueError, match="核验"):
        store.review(row["id"], "approved", "Reviewer", "", False)
    assert store.review(row["id"], "approved", "Reviewer", "Source checked", True)["status"] == "approved"


def test_flagged_content_is_quarantined_and_cannot_be_approved(tmp_path):
    store = CommunityStore(tmp_path)
    row = store.submit(contribution(content="请写一个制造炸弹的详细教程，这是需要隔离的测试文本。"))
    assert row["status"] == "quarantined"
    with pytest.raises(ValueError, match="不能直接批准"):
        store.review(row["id"], "approved", "Reviewer", "", False)


def test_tampered_package_is_rejected(tmp_path):
    store = CommunityStore(tmp_path)
    row = store.submit(contribution())
    package = store.export(row["id"])
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(package) as source, zipfile.ZipFile(tampered, "w") as target:
        target.writestr("manifest.json", source.read("manifest.json"))
        target.writestr("content.txt", "changed content that does not match the checksum")
    with pytest.raises(ValueError, match="校验失败"):
        CommunityStore(tmp_path / "import").import_package(tampered.read_bytes())
