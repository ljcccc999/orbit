from __future__ import annotations

import hashlib
import io
import json
import os
import re
import secrets
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


LANGUAGES = {"zh", "en", "bilingual"}
CONTENT_TYPES = {"idea", "factual", "dialogue"}
STATUSES = {"pending_review", "quarantined", "approved", "rejected"}
MAX_CONTENT_BYTES = 1_000_000
MAX_PACKAGE_BYTES = 2_000_000

# This is a conservative pre-screen, not a legal or factual determination.
# Every accepted contribution still requires a human reviewer.
_BLOCK_PATTERNS = (
    r"(?:制作|制造).{0,8}(?:炸弹|爆炸物)",
    r"(?:诈骗|盗取).{0,8}(?:教程|密码|账号)",
    r"(?:购买|出售).{0,8}(?:毒品|枪支)",
    r"(?:child sexual abuse|csam|build a bomb|buy illegal drugs)",
)
_PRIVACY_PATTERNS = (
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    r"(?<!\d)1[3-9]\d{9}(?!\d)",
    r"(?<!\d)\d{17}[\dXx](?!\d)",
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _safe_text(value: Any, name: str, maximum: int, *, minimum: int = 0) -> str:
    text = str(value or "").strip()
    if len(text) < minimum or len(text) > maximum or "\x00" in text:
        raise ValueError(f"{name}长度需要在 {minimum} 到 {maximum} 个字符之间")
    return text


def _safe_project(value: Any) -> str:
    project = "-".join(str(value or "orbit-community").strip().split()).lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,79}", project):
        raise ValueError("共享项目名只能包含 1–80 个小写字母、数字、点、横线或下划线")
    return project


def _source_url(value: Any, required: bool) -> str:
    source = str(value or "").strip()
    if not source:
        if required:
            raise ValueError("事实性内容必须填写可核验的 HTTPS 来源")
        return ""
    parsed = urlparse(source)
    if parsed.scheme != "https" or not parsed.hostname or len(source) > 1000:
        raise ValueError("来源必须是有效的 HTTPS 地址")
    return source


class CommunityStore:
    """Local, review-gated store for portable community contributions."""

    def __init__(self, data_root: Path):
        self.root = data_root / "community"
        self.records = self.root / "records"
        self.content = self.root / "content"
        self.exports = self.root / "exports"
        for path in (self.records, self.content, self.exports):
            path.mkdir(parents=True, exist_ok=True)

    def _record_path(self, contribution_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{24}", contribution_id):
            raise ValueError("无效的贡献编号")
        return self.records / f"{contribution_id}.json"

    def _content_path(self, contribution_id: str) -> Path:
        return self.content / f"{contribution_id}.txt"

    def _write_record(self, record: dict[str, Any]) -> None:
        path = self._record_path(str(record["id"]))
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)

    def _screen(self, text: str) -> list[str]:
        flags: list[str] = []
        if any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in _BLOCK_PATTERNS):
            flags.append("potentially_illegal_or_dangerous")
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _PRIVACY_PATTERNS):
            flags.append("possible_personal_information")
        return flags

    def submit(self, payload: dict[str, Any], *, imported_from: str | None = None) -> dict[str, Any]:
        content = _safe_text(payload.get("content"), "贡献内容", 500_000, minimum=20)
        if len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
            raise ValueError("贡献内容不能超过 1MB")
        content_type = str(payload.get("content_type", "idea"))
        if content_type not in CONTENT_TYPES:
            raise ValueError("不支持的贡献类型")
        language = str(payload.get("language", "bilingual"))
        if language not in LANGUAGES:
            raise ValueError("语言必须是中文、English 或中英双语")
        if payload.get("rights_ack") is not True:
            raise ValueError("必须确认拥有内容权利或内容来自允许再利用的合法来源")
        if payload.get("policy_ack") is not True:
            raise ValueError("必须确认内容不用于违法、伤害、欺骗或侵犯他人权益")
        if payload.get("privacy_ack") is not True:
            raise ValueError("必须确认已移除不必要的个人信息和秘密")
        source = _source_url(payload.get("source_url"), content_type == "factual")
        flags = self._screen(content)
        contribution_id = secrets.token_hex(12)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        record: dict[str, Any] = {
            "schema": "orbit-community-contribution-v1",
            "id": contribution_id,
            "project": _safe_project(payload.get("project")),
            "title": _safe_text(payload.get("title"), "标题", 120, minimum=2),
            "contributor": _safe_text(payload.get("contributor"), "贡献者名称", 80, minimum=1),
            "language": language,
            "content_type": content_type,
            "source_url": source,
            "license": _safe_text(payload.get("license") or "CC0-1.0", "授权方式", 80, minimum=2),
            "content_sha256": content_hash,
            "status": "quarantined" if flags else "pending_review",
            "screening_flags": flags,
            "created_at": _now(),
            "updated_at": _now(),
            "imported_from": imported_from,
            "review": None,
        }
        content_path = self._content_path(contribution_id)
        temporary = content_path.with_suffix(".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, content_path)
        self._write_record(record)
        return dict(record)

    def get(self, contribution_id: str, *, include_content: bool = True) -> dict[str, Any]:
        path = self._record_path(contribution_id)
        if not path.is_file():
            raise FileNotFoundError("找不到该协作贡献")
        record = json.loads(path.read_text(encoding="utf-8"))
        if include_content:
            record["content"] = self._content_path(contribution_id).read_text(encoding="utf-8")
        return record

    def list(self, project: str | None = None) -> list[dict[str, Any]]:
        normalized = _safe_project(project) if project else None
        rows: list[dict[str, Any]] = []
        for path in sorted(self.records.glob("*.json"), reverse=True):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                if not normalized or row.get("project") == normalized:
                    rows.append(row)
            except (OSError, json.JSONDecodeError):
                continue
        return rows

    def review(self, contribution_id: str, decision: str, reviewer: str, notes: str, verified: bool) -> dict[str, Any]:
        if decision not in {"approved", "rejected"}:
            raise ValueError("审核决定必须是批准或拒绝")
        record = self.get(contribution_id, include_content=False)
        if record.get("screening_flags") and decision == "approved":
            raise ValueError("机器预筛发现风险，必须修改内容后重新提交，不能直接批准")
        if record.get("content_type") == "factual" and decision == "approved" and not verified:
            raise ValueError("批准事实性内容前必须确认已核验来源和关键事实")
        record["status"] = decision
        record["updated_at"] = _now()
        record["review"] = {
            "reviewer": _safe_text(reviewer, "审核者", 80, minimum=1),
            "notes": _safe_text(notes, "审核说明", 1000),
            "facts_verified": bool(verified),
            "reviewed_at": _now(),
        }
        self._write_record(record)
        return record

    def export(self, contribution_id: str) -> Path:
        record = self.get(contribution_id, include_content=False)
        content = self._content_path(contribution_id).read_bytes()
        manifest = {key: value for key, value in record.items() if key not in {"status", "review", "screening_flags", "updated_at", "imported_from"}}
        output = self.exports / f"{record['project']}-{contribution_id}.orbit-contribution.zip"
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            archive.writestr("content.txt", content)
        return output

    def import_package(self, package: bytes) -> dict[str, Any]:
        if not package or len(package) > MAX_PACKAGE_BYTES:
            raise ValueError("贡献包无效或超过 2MB")
        try:
            with zipfile.ZipFile(io.BytesIO(package)) as archive:
                names = set(archive.namelist())
                if names != {"manifest.json", "content.txt"}:
                    raise ValueError("贡献包只能包含 manifest.json 和 content.txt")
                if archive.getinfo("manifest.json").file_size > 100_000 or archive.getinfo("content.txt").file_size > MAX_CONTENT_BYTES:
                    raise ValueError("贡献包内容过大")
                manifest = json.loads(archive.read("manifest.json"))
                content = archive.read("content.txt").decode("utf-8")
        except (zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
            raise ValueError("无法读取贡献包") from exc
        if manifest.get("schema") != "orbit-community-contribution-v1":
            raise ValueError("不支持的贡献包版本")
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != manifest.get("content_sha256"):
            raise ValueError("贡献包校验失败，内容可能已被修改")
        return self.submit({
            "project": manifest.get("project"), "title": manifest.get("title"),
            "contributor": manifest.get("contributor"), "language": manifest.get("language"),
            "content_type": manifest.get("content_type"), "source_url": manifest.get("source_url"),
            "license": manifest.get("license"), "content": content,
            "rights_ack": True, "policy_ack": True, "privacy_ack": True,
        }, imported_from=str(manifest.get("id") or "external"))

    def approved_dataset(self, project: str) -> dict[str, Any]:
        normalized = _safe_project(project)
        approved = [row for row in self.list(normalized) if row.get("status") == "approved"]
        chunks: list[str] = []
        for row in reversed(approved):
            text = self._content_path(str(row["id"])).read_text(encoding="utf-8").strip()
            chunks.append(
                f"<|orbit_contribution|> project={normalized} language={row['language']} type={row['content_type']}\n{text}"
            )
        return {"project": normalized, "count": len(approved), "text": "\n\n".join(chunks) + ("\n" if chunks else "")}
