from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from threading import Event, Thread
from typing import Callable
from urllib.parse import urlparse

from .identity import ORBIT_SYSTEM_PROMPT, ORBIT_TRAINING_ANCHOR


@dataclass
class TeacherConfig:
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    instruction: str = ""
    examples: int = 20
    language: str = "中文"
    corpus_mode: str = "document"
    model_profile: dict | None = None

    def endpoint(self) -> str:
        value = self.base_url.strip().rstrip("/")
        parsed = urlparse(value)
        local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
            raise ValueError("AI API 必须使用 HTTPS；只有本机 localhost 端点可以使用 HTTP")
        if not parsed.hostname:
            raise ValueError("AI API 地址无效")
        if value.endswith("/chat/completions"):
            return value
        return value + "/chat/completions"

    def validate(self) -> None:
        self.endpoint()
        if not self.model.strip() or len(self.model) > 200:
            raise ValueError("请填写有效的教师模型名称")
        if not self.instruction.strip() or len(self.instruction) > 20_000:
            raise ValueError("请填写不超过 20,000 字的训练目标")
        if not 1 <= self.examples <= 100:
            raise ValueError("自动生成样本数必须在 1 到 100 之间")
        if self.corpus_mode not in {"document", "mixed"}:
            raise ValueError("训练语料模式必须是 document 或 mixed")


def _request(endpoint: str, api_key: str, body: dict, attempts: int = 3, stop_event: Event | None = None) -> dict:
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=encoded,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("用户停止了 AI 数据生成")
        result: dict | None = None
        request_error: Exception | None = None

        def request_once() -> None:
            nonlocal result, request_error
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    result = json.loads(response.read())
            except Exception as exc:  # the supervisor below converts it to a normal request error
                request_error = exc

        request_thread = Thread(target=request_once, name="orbit-teacher-request", daemon=True)
        request_thread.start()
        while request_thread.is_alive():
            if stop_event is not None and stop_event.wait(0.2):
                raise InterruptedError("用户停止了 AI 数据生成")
            request_thread.join(0.2)
        if request_error is None and isinstance(result, dict):
            return result
        if isinstance(request_error, urllib.error.HTTPError):
            detail = request_error.read(1000).decode("utf-8", errors="replace")
            if request_error.code not in {408, 429, 500, 502, 503, 504}:
                raise RuntimeError(f"教师 API 返回 HTTP {request_error.code}：{detail}") from request_error
            last_error = RuntimeError(f"教师 API 暂时不可用（HTTP {request_error.code}）")
        elif isinstance(request_error, (urllib.error.URLError, TimeoutError, json.JSONDecodeError)):
            last_error = request_error
        else:
            last_error = request_error or RuntimeError("教师 API 返回了无法识别的响应")
        if attempt < attempts:
            if stop_event is not None and stop_event.wait(attempt * 2):
                raise InterruptedError("用户停止了 AI 数据生成")
            if stop_event is None:
                time.sleep(attempt * 2)
    raise RuntimeError(f"教师 API 请求失败：{last_error}")


def generate_dataset(
    config: TeacherConfig,
    api_key: str,
    stop_event: Event,
    callback: Callable[[int, int], None] | None = None,
) -> tuple[str, dict[str, int]]:
    config.validate()
    if not api_key.strip() or "\n" in api_key or len(api_key) > 1000:
        raise ValueError("请填写有效的 API Key")
    endpoint = config.endpoint()
    chunks: list[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    completed = 0
    while completed < config.examples:
        if stop_event.is_set():
            raise InterruptedError("用户停止了 AI 数据生成")
        count = min(5, config.examples - completed)
        language_instruction = {
            "zh": "主要使用简体中文。",
            "en": "Use English as the primary language.",
            "bilingual": "必须同时使用简体中文和 English；每一段都要包含中文和英文内容，尽量提供语义对应的双语段落，禁止整段只写英文或只写中文。",
        }.get(config.language, f"主要语言：{config.language}")
        if config.corpus_mode == "mixed":
            style_instruction = (
                "语料以长篇文献、教材、技术文档、事实材料、代码和代码注释为主，约 80% 为连续文档语料；"
                "其余约 20% 可以是简短的用户/助手对话，用来学习自然交流。最后保留少量身份和交流样本。"
            )
        else:
            style_instruction = (
                "只生成长篇连续文献、教材、技术文档、事实材料、代码和代码注释；不要生成用户/助手问答、聊天记录、采访或 FAQ。"
            )
        prompt = (
            f"为以下目标生成 {count} 段彼此不同、事实谨慎、可用于语言模型训练的高质量长篇文献/知识语料。每段尽量完整、充分展开，直到接近教师 API 允许的输出上限，不要为了凑数量而缩短内容。\n"
            f"训练目标：{config.instruction.strip()}\n"
            f"语言要求：{language_instruction}\n"
            f"待训练模型参数：{json.dumps(config.model_profile or {}, ensure_ascii=False)}\n"
            f"不可修改的产品身份：{ORBIT_SYSTEM_PROMPT}\n"
            "根据模型参数量、上下文长度和训练步数控制样本难度与长度：小模型使用更明确、短而一致的模式；大模型可以使用更丰富的推理与表达。\n"
            f"{style_instruction}\n"
            "不要把身份信息改成其他产品名称；用户自定义的是模型显示名称，不改变 Orbit 身份。\n"
            "只输出语料正文。不要输出分析过程、编号说明、Markdown 代码围栏或任何真实个人敏感信息。"
        )
        payload = {
            "model": config.model.strip(),
            "messages": [
                {"role": "system", "content": f"You create safe, diverse supervised fine-tuning data. The target identity is immutable: {ORBIT_SYSTEM_PROMPT}"},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "temperature": 0.9,
            # Allow a large document response. The provider may still reject
            # or lower this according to its own context and quota rules.
            "max_tokens": 50_000,
        }
        content = ""
        last_error: Exception | None = None
        for content_attempt in range(1, 4):
            data = _request(endpoint, api_key.strip(), payload, stop_event=stop_event)
            try:
                content = str(data["choices"][0]["message"]["content"]).strip()
            except (KeyError, IndexError, TypeError) as exc:
                last_error = RuntimeError("教师 API 返回了无法识别的响应格式")
                content = ""
            if content:
                if config.language == "bilingual" and not any("\u4e00" <= char <= "\u9fff" for char in content):
                    last_error = RuntimeError("教师 API 返回了英文-only 内容，双语训练要求每段同时包含中文")
                    content = ""
                    payload["messages"][-1]["content"] += (
                        "\n上一版输出不合格：不能只有 English。请重新生成，确保每一段都同时包含简体中文和 English，"
                        "并保持文献/技术语料为主。"
                    )
                else:
                    break
            last_error = RuntimeError("教师 API 返回了空内容")
            if content_attempt < 3:
                if stop_event.wait(content_attempt * 2):
                    raise InterruptedError("用户停止了 AI 数据生成")
        if not content:
            raise last_error or RuntimeError("教师 API 返回了空内容")
        chunks.append(content)
        response_usage = data.get("usage") or {}
        for key in usage:
            usage[key] += int(response_usage.get(key, 0) or 0)
        completed += count
        if callback:
            callback(completed, config.examples)
    # This is a real training corpus prefix, not a runtime answer shortcut. It
    # is included in every AI-assisted dataset even if the teacher omits it.
    corpus = ORBIT_TRAINING_ANCHOR + "\n\n" + "\n\n".join(chunks) + "\n"
    # The communication tail is training data, not a runtime shortcut. It is
    # intentionally small for both first training and secondary training, so
    # the corpus remains document-first while the model learns basic dialogue
    # behavior and its identity.
    corpus += (
        "\nCOMMUNICATION EXAMPLES\n"
        "<|user|>你好\n"
        "<|assistant|>你好，我会根据本地训练语料和当前上下文帮助你。\n"
        "<|user|>你是谁？\n"
        "<|assistant|>我是 Orbit，由 YUNSH 开发的本地 AI。\n"
        "<|user|>请解释你掌握的内容。\n"
        "<|assistant|>我会先依据训练得到的文献、技术资料和代码模式组织回答；不确定时会明确说明。\n"
    )
    return corpus, usage
