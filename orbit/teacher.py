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
    training_round: int = 1
    corpus_plan: str = ""
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
        if not 1 <= self.examples <= 5_000:
            raise ValueError("自动生成样本数必须在 1 到 5,000 之间")
        if self.corpus_mode not in {"document", "mixed", "pretraining", "fine_tuning"}:
            raise ValueError("训练语料模式必须是 document、pretraining 或 fine_tuning")
        if not 1 <= int(self.training_round) <= 1000:
            raise ValueError("训练次数必须在 1 到 1000 之间")
        if len(self.corpus_plan) > 20_000:
            raise ValueError("语料规划不能超过 20,000 字")


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
        if config.corpus_mode == "fine_tuning":
            document_count = round(count * 0.25)
            task_count = round(count * 0.25)
            dialogue_count = count - document_count - task_count
            composition = (
                f"这批 {count} 段样本尽量包含约 {document_count} 段专业文献/技术资料、约 {task_count} 段单轮结构化任务、约 {dialogue_count} 段对话（总体目标约 25%/25%/50%）。"
                if count >= 4
                else f"本次只有 {count} 段样本，不能在单批次强行凑出三类比例；请按 25%/25%/50% 的总体目标优先保证类型多样，后续批次补足比例。"
            )
            style_instruction = (
                composition
                + "单轮结构化任务不是聊天气泡，不要添加 user/assistant 对话格式；必须覆盖分类/情感分析、实体抽取（NER）、代码或 SQL 生成、摘要总结、文本扩写/润色等类型，使用清晰的输入→输出格式。"
                + "对话部分才使用 user/assistant 格式，并保留少量 Orbit 身份和交流样本。"
            )
        elif config.corpus_mode in {"pretraining", "mixed"}:
            document_count = round(count * 0.8)
            task_count = round(count * 0.1)
            dialogue_count = count - document_count - task_count
            composition = (
                f"这是预训练第 1～N 次的文献优先语料：本批 {count} 段尽量包含约 {document_count} 段长篇文献/教材/技术资料/代码、约 {task_count} 段单轮任务、约 {dialogue_count} 段对话（总体目标约 80%/10%/10%）；"
                if count >= 5
                else f"这是预训练第 1～N 次的文献优先语料；本次只有 {count} 段样本，不能在单批次强行凑出比例，请优先生成长篇文献/技术资料，后续批次补足任务和对话尾部。"
            )
            style_instruction = (
                composition
                + "约 10% 是分类/情感分析、实体抽取（NER）、代码或 SQL 生成、摘要总结、文本扩写/润色等单轮输入→输出任务；"
                + "约 10% 才是简短对话和身份样本。继续预训练必须保持文献主体，不要把整个数据集做成聊天气泡。"
            )
        else:
            style_instruction = (
                "只生成长篇连续文献、教材、技术文档、事实材料、代码和代码注释；不要生成用户/助手问答、聊天记录、采访或 FAQ。"
            )
        prompt = (
            f"为以下目标生成 {count} 段彼此不同、事实谨慎、可用于语言模型训练的高质量长篇文献/知识语料。每段尽量完整、充分展开，直到接近教师 API 允许的输出上限，不要为了凑数量而缩短内容。\n"
            f"训练目标：{config.instruction.strip()}\n"
            f"语料规划（必须执行）：{config.corpus_plan.strip() or '按下方 Orbit 默认语料配比生成'}\n"
            f"语言要求：{language_instruction}\n"
            f"待训练模型参数和训练轮次：{json.dumps(config.model_profile or {}, ensure_ascii=False)}；这是 Orbit 训练第 {config.training_round} 次。\n"
            f"不可修改的产品身份：{ORBIT_SYSTEM_PROMPT}\n"
            "根据模型参数量、上下文长度和训练步数控制样本难度与长度：小模型使用更明确、短而一致的模式；大模型可以使用更丰富的推理与表达。\n"
            f"{style_instruction}\n"
            "不要把身份信息改成其他产品名称；用户自定义的是模型显示名称，不改变 Orbit 身份。\n"
            "结构化任务示例应类似：输入：这手机电池太差了；输出：负面。输入：我住在北京朝阳区；输出：北京（地名）、朝阳区（地名）。输入：查询年龄大于18岁的用户；输出：SELECT * FROM users WHERE age > 18。摘要和扩写任务要给出完整目标输出，不要只写一句解释。"
            "生成要求（全部执行）：1）内容必须围绕训练目标，不要泛泛重复；2）长篇文献要有标题、概念定义、事实依据、例子、边界和小结，不能用空话凑长度；3）样本之间更换主题、措辞、难度和输入形式，禁止复制、近重复和模板循环；4）不确定事实要标注不确定性，不得捏造论文、作者、数据、法规、网址或实验结果；5）不得生成真实个人隐私、账号密钥、恶意代码、违法操作或危险操作教程；6）代码尽量语法正确并说明语言，SQL 给出假定表/字段或保持查询条件清晰；7）分类任务使用稳定标签，情感标签前后一致；8）NER 统一实体类型并覆盖不同句式；9）摘要不得添加原文没有的事实，扩写/润色不得改变原意；10）结构化任务必须是一轮输入→输出，禁止写成多轮聊天；11）对话样本只用于表达、拒答和 Orbit 身份，不能占据文献主体；12）中文、English 或双语必须服从所选语言；13）先生成知识与代码语料，再在末尾追加少量结构化任务、对话和身份样本；14）不要把‘你是谁’写进程序逻辑，只作为训练样本；15）输出适合 UTF-8 文本训练，避免不可见控制字符；16）只输出语料正文，不输出分析过程、编号说明、Markdown 代码围栏或真实个人敏感信息。"
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
