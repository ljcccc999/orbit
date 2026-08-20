<p align="center">
  <img src="orbit/static/orbit-logo.png" width="180" alt="Orbit Logo">
</p>

<p align="center">
  <a href="README.md">English</a> | <strong>中文</strong>
</p>

# Orbit

Orbit 是一个本地 AI 工作室，把模型训练、checkpoint 管理、对话和 OpenAI 兼容 API 集中在同一个浏览器界面中。默认情况下，训练数据、模型和对话都保留在用户自己的电脑上。

## Orbit 可以做什么

- 在浏览器中打开图形化训练和对话界面。
- 提供约 300M 到 38B 参数的架构预设。
- 分配模型前检查本机内存。
- 将本机训练的 checkpoint 保存在 `~/.orbit/models`。
- 为 CUDA GPU 服务器生成自包含训练任务包。
- 通过 `GET /v1/models` 和 `POST /v1/chat/completions` 提供本地模型 API。
- 让本机智能体通过 OpenAI 兼容配置接入。

## 运行要求

- 系统内存必须大于 10GB；更大的模型需要更多内存。
- Python 3.9 或更高版本。
- macOS 或 Linux；正式训练建议使用 CUDA。
- 首次下载安装器和依赖时需要联网。完成后，训练、推理和本机 API 可以离线运行。

## 一行安装并启动

```bash
curl -fsSL https://raw.githubusercontent.com/ljcccc999/orbit/main/install.sh | sh
```

安装器会检查内存，在 `~/.orbit/runtime` 创建隔离运行环境，安装 Orbit，启动本机服务并打开浏览器。以后执行 `orbit` 即可再次打开。

Orbit 默认只监听 `127.0.0.1:8765`，不会暴露给局域网或互联网。

## 模型规模

| 预设 | 近似参数量 | 保守训练内存估算 | 适用环境 |
| --- | ---: | ---: | --- |
| 300M | 0.308B | 9.9GB | 性能较好的个人电脑 |
| 1B | 1.063B | 34.0GB | 大内存工作站或 GPU |
| 3B | 2.969B | 95.0GB | 工作站或多 GPU 服务器 |
| 7B | 7.179B | 229.7GB | 多 GPU 服务器 |
| 14B | 14.075B | 450.4GB | 分布式训练系统 |
| 38B | 38.260B | 1,224.3GB | 大型分布式 GPU 集群 |

这些是包含优化器状态和工作内存的保守完整训练估算。实际需求会受到精度、优化器、激活检查点、分片、序列长度和硬件的影响。

## 训练

打开“训练”页面，选择预设，粘贴用户自己的文本，然后选择两种路径之一：

1. **在本机开始训练**：只有通过内存检查后才会启动本地任务，checkpoint 保存在 `~/.orbit/models`。
2. **生成远程 GPU 任务包**：下载一个 ZIP，其中包含数据集、配置、Orbit 源码和供 CUDA 机器执行的 `run.sh` 入口。

300M–38B 表示架构规模，不是已经预训练好的模型下载。要从零训练出真正可用的基础模型，需要大量经过认真处理的数据和可观的算力。

## 本地对话

训练完成后，打开“对话”页面，加载 checkpoint 并发送消息。模型从本机存储加载，推理不需要互联网连接。

Orbit 当前使用实验性的字节级 tokenizer 和模型架构。很小或很短的训练只能验证流程，无法直接产生通用能力很强的助手。

## 接入本机智能体

打开“API”页面即可复制地址和示例。默认配置如下：

```json
{
  "provider": "openai-compatible",
  "base_url": "http://127.0.0.1:8765/v1",
  "api_key": "orbit-local",
  "model": "你的本地模型名称"
}
```

Python 示例：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="orbit-local",
)

response = client.chat.completions.create(
    model="你的本地模型名称",
    messages=[{"role": "user", "content": "你好"}],
)
print(response.choices[0].message.content)
```

当前兼容层支持模型列表和非流式聊天补全。Orbit 只绑定 localhost 时，API Key 的值仅用于兼容客户端，不进行校验。

## 本地存储与隐私

| 内容 | 默认位置 |
| --- | --- |
| 模型与 checkpoint | `~/.orbit/models` |
| 远程训练任务包 | `~/.orbit/jobs` |
| 隔离 Python 运行环境 | `~/.orbit/runtime` |

Orbit 不会上传本地训练文本、checkpoint 或聊天消息。只有用户选择在另一台机器训练时，才会主动移动远程训练任务包。

## 开发

```bash
git clone https://github.com/ljcccc999/orbit.git
cd orbit
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q
.venv/bin/orbit
```

## 许可证

Orbit 使用 [MIT License](LICENSE)。
