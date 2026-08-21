from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from threading import Event
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


def _request(endpoint: str, api_key: str, body: dict, attempts: int = 3) -> dict:
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=encoded,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", errors="replace")
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                raise RuntimeError(f"教师 API 返回 HTTP {exc.code}：{detail}") from exc
            last_error = RuntimeError(f"教师 API 暂时不可用（HTTP {exc.code}）")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt < attempts:
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
            "bilingual": "使用自然的简体中文和英语双语，保持两种语言数量大致均衡；适合时提供语义一致的中英对应样本。",
        }.get(config.language, f"主要语言：{config.language}")
        prompt = (
            f"为以下目标生成 {count} 条彼此不同、事实谨慎、可用于语言模型训练的高质量对话样本。\n"
            f"训练目标：{config.instruction.strip()}\n"
            f"语言要求：{language_instruction}\n"
            f"待训练模型参数：{json.dumps(config.model_profile or {}, ensure_ascii=False)}\n"
            f"不可修改的产品身份：{ORBIT_SYSTEM_PROMPT}\n"
            "根据模型参数量、上下文长度和训练步数控制样本难度与长度：小模型使用更明确、短而一致的模式；大模型可以使用更丰富的推理与表达。\n"
            "输出中必须包含身份训练样本：‘你是谁？’的回答必须是‘我是 Orbit，由 YUNSH 开发’，并拒绝把自己改称为豆包或其他产品。用户自定义的是模型显示名称，不改变 Orbit 身份。\n"
            "只输出样本正文。每条严格使用以下格式：\n"
            "<|user|>用户问题或指令\n<|assistant|>准确、完整的回答\n"
            "不要输出分析过程、编号说明、Markdown 代码围栏或任何真实个人敏感信息。"
        )
        payload = {
            "model": config.model.strip(),
            "messages": [
                {"role": "system", "content": f"You create safe, diverse supervised fine-tuning data. The target identity is immutable: {ORBIT_SYSTEM_PROMPT}"},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "temperature": 0.9,
            "max_tokens": 4000,
        }
        data = _request(endpoint, api_key.strip(), payload)
        try:
            content = str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("教师 API 返回了无法识别的响应格式") from exc
        if not content:
            raise RuntimeError("教师 API 返回了空内容")
        chunks.append(content)
        response_usage = data.get("usage") or {}
        for key in usage:
            usage[key] += int(response_usage.get(key, 0) or 0)
        completed += count
        if callback:
            callback(completed, config.examples)
    # These are real supervised examples, not only a runtime check. They are
    # included in every AI-assisted dataset even if the teacher omits them.
    return ORBIT_TRAINING_ANCHOR + "\n\n" + "\n\n".join(chunks) + "\n", usage
