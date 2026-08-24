from __future__ import annotations

import json
import hashlib
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from threading import Event, Thread
from typing import Callable
from urllib.parse import urlparse

from .identity import ORBIT_SYSTEM_PROMPT, ORBIT_TRAINING_ANCHOR


SAMPLE_START = "<|orbit_sample|>"
SAMPLE_END = "<|end_orbit_sample|>"
COMMUNICATION_TAIL = (
    "COMMUNICATION EXAMPLES\n"
    "<|user|>你好\n"
    "<|assistant|>你好，我会根据本地训练语料和当前上下文帮助你。\n"
    "<|user|>你是谁？\n"
    "<|assistant|>我是 Orbit，由 YUNSH 开发的本地 AI。\n"
    "<|user|>请解释你掌握的内容。\n"
    "<|assistant|>我会先依据训练得到的文献、技术资料和代码模式组织回答；不确定时会明确说明。\n"
)
_SAMPLE_PATTERN = re.compile(
    re.escape(SAMPLE_START) + r"\s*(.*?)\s*" + re.escape(SAMPLE_END),
    re.DOTALL,
)


def extract_samples(text: str) -> list[str]:
    """Return explicitly delimited teacher samples, never guessed paragraphs."""
    return [match.strip() for match in _SAMPLE_PATTERN.findall(text) if match.strip()]


def serialize_samples(samples: list[str]) -> str:
    return "\n\n".join(f"{SAMPLE_START}\n{sample.strip()}\n{SAMPLE_END}" for sample in samples)


def _normalized_sample(sample: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", sample.casefold())


def _sample_simhash(sample: str) -> int:
    normalized = _normalized_sample(sample)
    if len(normalized) < 5:
        return int.from_bytes(hashlib.sha256(normalized.encode()).digest()[:8], "big")
    weights = [0] * 64
    # Cap work for very long documents while retaining coverage across them.
    stride = max(1, (len(normalized) - 4) // 5000)
    for index in range(0, len(normalized) - 4, stride):
        value = int.from_bytes(hashlib.blake2b(normalized[index:index + 5].encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    return sum((1 << bit) for bit, weight in enumerate(weights) if weight >= 0)


def _valid_sample(sample: str, language: str) -> bool:
    if len(sample) < 200 or "\ufffd" in sample:
        return False
    if any(ord(char) < 32 and char not in "\n\r\t" for char in sample):
        return False
    lines = [re.sub(r"\s+", " ", line).strip() for line in sample.splitlines() if line.strip()]
    if len(lines) >= 8 and len(set(lines)) / len(lines) < 0.55:
        return False
    has_zh = any("\u4e00" <= char <= "\u9fff" for char in sample)
    has_en = bool(re.search(r"\b[A-Za-z]{3,}\b", sample))
    if language == "bilingual" and not (has_zh and has_en):
        return False
    if language == "zh" and not has_zh:
        return False
    if language == "en" and not has_en:
        return False
    return True


def corpus_statistics(text: str) -> dict[str, int | str]:
    samples = extract_samples(text)
    return {
        "samples": len(samples),
        "characters": len(text),
        "bytes": len(text.encode("utf-8")),
        # Orbit's current tokenizer maps one UTF-8 byte to one token.
        "tokens": len(text.encode("utf-8")),
        "tokenizer": "orbit-byte-v1",
        "structural_validation": "passed" if samples else "legacy_or_unstructured",
        "fact_check_status": "not_independently_verified",
    }


def split_generated_corpus(corpus: str, validation_ratio: float = 0.05) -> tuple[str, str, dict[str, int | str]]:
    """Create a deterministic held-out split from strict sample boundaries."""
    samples = extract_samples(corpus)
    if not samples:
        raise ValueError("教师语料没有可验证的样本边界")
    validation: list[str] = []
    training: list[str] = []
    for sample in samples:
        bucket = int.from_bytes(hashlib.sha256(_normalized_sample(sample).encode()).digest()[:8], "big") % 10_000
        (validation if bucket < round(validation_ratio * 10_000) else training).append(sample)
    if len(samples) > 1 and not validation:
        validation.append(training.pop())
    if not training:
        training.append(validation.pop())
    train_text = ORBIT_TRAINING_ANCHOR + "\n\n" + serialize_samples(training) + "\n\n" + COMMUNICATION_TAIL
    validation_text = serialize_samples(validation) + ("\n" if validation else "")
    stats = corpus_statistics(serialize_samples(samples))
    stats.update({"training_samples": len(training), "validation_samples": len(validation)})
    return train_text, validation_text, stats


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
        if not 1 <= self.examples <= 10_000_000:
            raise ValueError("自动生成样本数必须在 1 到 10,000,000 之间")
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
    chunk_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[str, dict[str, int]]:
    config.validate()
    if not api_key.strip() or "\n" in api_key or len(api_key) > 1000:
        raise ValueError("请填写有效的 API Key")
    endpoint = config.endpoint()
    chunks: list[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    seen_hashes: set[str] = set()
    seen_simhashes: list[int] = []
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
            composition = (
                f"这是第 {config.training_round} 次微调数据生成，本批 {count} 段不要机械套用固定百分比。请在基础知识、专业文献、结构化任务、代码/数学、对话表达和 Orbit 身份之间保持多样性；优先补足当前训练记录中数量不足或验证集效果较差的类别。"
            )
            style_instruction = (
                composition
                + "单轮结构化任务不是聊天气泡，不要添加 user/assistant 对话格式；覆盖分类/情感分析、实体抽取（NER）、代码或 SQL 生成、摘要总结、文本扩写/润色等类型。对话部分才使用 user/assistant 格式，并保留少量 Orbit 身份和交流样本。不要因为追求样本数量而降低质量。"
            )
        elif config.corpus_mode in {"pretraining", "mixed"}:
            composition = (
                f"这是预训练第 1～N 次的基础能力语料：本批 {count} 段以基础认知、通用知识、教材/科学/数学/逻辑为主体，同时加入适量代码技术资料和结构化任务，最后只保留少量对话与身份样本；不要把预训练做成聊天数据集。"
            )
            style_instruction = (
                composition
                + "结构化任务覆盖分类/情感分析、实体抽取（NER）、代码或 SQL 生成、摘要总结、文本扩写/润色等单轮输入→输出任务；对话和身份样本只占小部分。"
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
            "生成要求（全部执行）：1）内容必须围绕训练目标，不要泛泛重复；2）长篇文献要有标题、概念定义、事实依据、例子、边界和小结，不能用空话凑长度；3）样本之间更换主题、措辞、难度和输入形式，禁止复制、近重复和模板循环；4）不确定事实要标注不确定性，不得捏造论文、作者、数据、法规、网址或实验结果；5）不得生成真实个人隐私、账号密钥、恶意代码、违法操作或危险操作教程；6）代码尽量语法正确并说明语言，SQL 给出假定表/字段或保持查询条件清晰；7）分类任务使用稳定标签，情感标签前后一致；8）NER 统一实体类型并覆盖不同句式；9）摘要不得添加原文没有的事实，扩写/润色不得改变原意；10）结构化任务必须是一轮输入→输出，禁止写成多轮聊天；11）对话样本只用于表达、拒答和 Orbit 身份，不能占据文献主体；12）中文、English 或双语必须服从所选语言；13）先生成知识与代码语料，再在末尾追加少量结构化任务、对话和身份样本；14）不要把‘你是谁’写进程序逻辑，只作为训练样本；15）输出适合 UTF-8 文本训练，避免不可见控制字符；16）只输出语料正文，不输出分析过程、编号说明、Markdown 代码围栏或真实个人敏感信息；17）优先生成可验证、可复用、教育性强的教材式内容，而不是营销文案、短口号或无上下文事实列表；18）对同一知识点提供定义、例子、反例、边界、步骤和小练习；19）生成前后检查语言、代码语法、SQL 逻辑、标签一致性和事实谨慎性；20）不要重复用户提供的人工专属语料，AI 语料负责基础认知和通用能力，用户语料负责专属知识；21）保留一部分未参与训练的验证题型建议，避免训练集和测试集泄漏；22）不同主题、来源和难度要均衡混合，不能让单一模板占据全部样本。"
            f"\n输出边界（必须严格执行）：恰好输出 {count} 条独立样本；每条以 {SAMPLE_START} 单独一行开始，以 {SAMPLE_END} 单独一行结束。禁止在边界外输出任何文字，禁止省略或合并边界。"
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
        batch_samples: list[str] = []
        last_error: Exception | None = None
        for content_attempt in range(1, 4):
            data = _request(endpoint, api_key.strip(), payload, stop_event=stop_event)
            response_usage = data.get("usage") or {}
            for key in usage:
                usage[key] += int(response_usage.get(key, 0) or 0)
            try:
                content = str(data["choices"][0]["message"]["content"]).strip()
            except (KeyError, IndexError, TypeError) as exc:
                last_error = RuntimeError("教师 API 返回了无法识别的响应格式")
                content = ""
            if content:
                candidates = extract_samples(content)
                accepted: list[str] = []
                local_hashes: set[str] = set()
                local_simhashes: list[int] = []
                for sample in candidates:
                    normalized = _normalized_sample(sample)
                    digest = hashlib.sha256(normalized.encode()).hexdigest()
                    simhash = _sample_simhash(sample)
                    near_duplicate = any(bin(simhash ^ previous).count("1") <= 3 for previous in seen_simhashes + local_simhashes)
                    if _valid_sample(sample, config.language) and digest not in seen_hashes and digest not in local_hashes and not near_duplicate:
                        accepted.append(sample)
                        local_hashes.add(digest)
                        local_simhashes.append(simhash)
                if len(accepted) >= count:
                    batch_samples = accepted[:count]
                    break
                last_error = RuntimeError(f"教师 API 返回 {len(candidates)} 条带边界内容，但只有 {len(accepted)} 条通过长度、语言和去重检查；需要 {count} 条")
            else:
                last_error = RuntimeError("教师 API 返回了空内容")
            if content_attempt < 3:
                if stop_event.wait(content_attempt * 2):
                    raise InterruptedError("用户停止了 AI 数据生成")
                payload["messages"][-1]["content"] += (
                    f"\n上一版不合格：{last_error}。请重新生成，严格使用样本边界并保证每条独立、完整、非重复。"
                )
        if len(batch_samples) != count:
            raise last_error or RuntimeError("教师 API 没有返回足够的有效样本")
        chunk = serialize_samples(batch_samples)
        chunks.append(chunk)
        for sample in batch_samples:
            normalized = _normalized_sample(sample)
            seen_hashes.add(hashlib.sha256(normalized.encode()).hexdigest())
            seen_simhashes.append(_sample_simhash(sample))
        completed += len(batch_samples)
        if chunk_callback:
            chunk_callback(chunk, completed, config.examples)
        if callback:
            callback(completed, config.examples)
    # This is a real training corpus prefix, not a runtime answer shortcut. It
    # is included in every AI-assisted dataset even if the teacher omits it.
    corpus = ORBIT_TRAINING_ANCHOR + "\n\n" + "\n\n".join(chunks) + "\n"
    # The communication tail is training data, not a runtime shortcut. It is
    # intentionally small for both first training and secondary training, so
    # the corpus remains document-first while the model learns basic dialogue
    # behavior and its identity.
    corpus += "\n" + COMMUNICATION_TAIL
    return corpus, usage
