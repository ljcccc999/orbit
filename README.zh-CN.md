<p align="center">
  <img src="orbit/static/orbit-logo.png" width="180" alt="Orbit Logo">
</p>

<p align="center">
  <a href="README.md">English</a> | <strong>中文</strong>
</p>

# Orbit

Orbit 是一个本地 AI 工作室，把模型训练、checkpoint 管理、对话和 OpenAI 兼容 API 集中在同一个界面中。默认情况下，训练数据、模型和对话都保留在用户自己的电脑上。

## Orbit 可以做什么

- 提供可拖动的桌面工作台，并把对话、模型、训练、训练历史和 API 分成独立区域。
- 默认跟随操作系统语言，也可手动切换 English / 中文。
- 提供约 300M 到 38B 参数的架构预设。
- 分配模型前检查本机内存。
- 在 macOS、Linux 和 Windows 上运行可崩溃恢复的后台 API。
- 关闭窗口后仍在 macOS 菜单栏或 Windows/Linux 系统托盘常驻；只有选择“退出 Orbit”才停止本地服务。
- 按需加载模型；空闲五分钟后自动卸载权重并释放加速器缓存。
- 可使用 DeepSeek 或其他 OpenAI 兼容 API，根据所选模型参数生成监督数据并自动进入本机训练；每个提供商分别保存模型、地址和 API Key，更换一个 Key 不影响其他提供商。
- 根据模型规模、训练设备、当前可用内存以及本机或教师样本的数据量自动调整高级训练参数；本机无法安全完成的配置会在分配内存前阻止启动，并建议生成远程 GPU 任务包。
- 将本机训练的 checkpoint 保存在 `~/.orbit/models`。
- 支持自定义模型名、从已有 checkpoint 二次训练、父模型血缘，以及查看每次训练的原文、参数和结果。
- Orbit 身份独立于用户自定义模型名；即使还没训练模型也知道自己是 Orbit，服务器导出包也会保留这个身份。
- 为 CUDA GPU 服务器生成自包含训练任务包。
- 可把训练模型导出为带 OpenAI 兼容 API 的便携 Orbit 服务器；存在兼容的同名 GGUF 时也可生成 Ollama 包。
- 通过 `GET /v1/models`、`POST /v1/chat/completions` 和 `POST /v1/responses` 提供本地模型 API。
- 可生成多个随机、可撤销的 API Key，并把每个 Key 限定到某一个本地模型。
- 让本机智能体通过 OpenAI 兼容配置接入。

## 运行要求

- 系统内存必须大于 10GB；更大的模型需要更多内存。
- macOS、Linux 或 Windows；正式训练建议使用 CUDA。
- 用户不必提前安装 Python。安装器会使用合适的现有 Python，或自动准备隔离的 Python 3.11 运行时。
- 首次下载安装器和依赖时需要联网。完成后，训练、推理和本机 API 可以离线运行。

## 一行安装并启动

### 桌面 App

下载最新的 [macOS 通用 App](https://github.com/ljcccc999/orbit/releases/latest/download/Orbit-macOS-universal.zip)、[Windows x64 App](https://github.com/ljcccc999/orbit/releases/latest/download/Orbit-Windows-x64.exe)、[Windows ARM64 App](https://github.com/ljcccc999/orbit/releases/latest/download/Orbit-Windows-arm64.exe)、[Linux x64 AppImage](https://github.com/ljcccc999/orbit/releases/latest/download/Orbit-Linux-x64.AppImage) 或 [Linux ARM64 AppImage](https://github.com/ljcccc999/orbit/releases/latest/download/Orbit-Linux-arm64.AppImage)。双击 App 后会在图形界面中自动准备本机运行时，不需要打开命令行。

首次启动后，Orbit 会随用户登录自动运行。关闭或最小化窗口时，Orbit 会继续留在 macOS 右上角菜单栏、Windows 通知区域或 Linux 系统托盘中，本地 API 保持可用。只有从该菜单选择“退出 Orbit”，才会停止 API 并取消自动启动；以后手动打开 Orbit 会重新启用。

当前公开构建还没有 Developer ID/Authenticode 正式签名和公证，因此 macOS Gatekeeper 或 Windows SmartScreen 在首次打开时可能要求用户明确确认。彻底去掉系统提示需要对应平台的代码签名证书，程序代码不能安全绕过。

### macOS 与 Linux 命令行

```bash
curl --retry 5 --retry-delay 2 --connect-timeout 20 -fsSL https://raw.githubusercontent.com/ljcccc999/orbit/main/install.sh | sh
```

安装器会检查内存，在 `~/.orbit/runtime` 创建隔离运行环境，安装 Orbit，启动本机服务并打开浏览器。以后执行 `orbit` 即可再次打开。网页只是管理客户端，关掉网页不会停止 OpenAI 兼容 API。

下载中断后可以安全地重新执行同一条命令。安装器会继续使用已有的隔离环境，并自动重试网络下载。

Orbit 默认只监听 `127.0.0.1:8765`，不会暴露给局域网或互联网。

```bash
orbit start
orbit status
orbit chat "你好"
orbit unload
orbit stop
```

macOS LaunchAgent 和 Linux 用户级 systemd 服务会在 Orbit 异常退出后重启。`orbit unload` 可立即释放当前模型；不手动执行时，默认空闲计时器也会自动卸载。

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

打开“训练”页面，选择预设，可选填写模型名或选择父 checkpoint，粘贴用户自己的文本，然后选择三种路径之一：

1. **在本机开始训练**：只有通过内存检查后才会启动本地任务，checkpoint 保存在 `~/.orbit/models`。
2. **生成远程 GPU 任务包**：下载一个 ZIP，其中包含数据集、配置、Orbit 源码和供 CUDA 机器执行的 `run.sh` 入口。
3. **生成样本并自动训练**：调用 DeepSeek 或其他 OpenAI 兼容教师 API。Orbit 会把所选模型的参数量、上下文长度、步数、父模型和用户目标告诉教师模型，让样本适应该模型规模。样本数会影响数据覆盖面、生成时间和 API 费用。生成的数据集保存在本机。每个提供商的模型、地址和 Key 分别保存在 `~/.orbit/teacher-api.json`，权限仅限当前用户；切换回来会恢复原配置，输入新 Key 只替换当前提供商的旧 Key。Key 不会进入训练历史或导出包。训练目标会发送给提供商且可能产生费用，因此必须由用户明确确认。

每次实际训练都会在 `~/.orbit/training-runs` 保存可查看的记录，包括原始训练内容、参数、状态、loss、结果、模型名和父模型。点击“二次训练”会创建新 checkpoint 和新优化器，不覆盖父模型。

300M–38B 表示架构规模，不是已经预训练好的模型下载。要从零训练出真正可用的基础模型，需要大量经过认真处理的数据和可观的算力。

## 本地对话

训练完成后，打开“对话”页面直接发送消息，不需要手动加载。Orbit 会像 Ollama 一样自动选择并加载本机 checkpoint；推理不需要互联网连接。“新对话”只清除当前可见消息，不会删除模型或 Key。网页和桌面 App 关闭后，后台 API 仍可回答；如果模型因空闲被卸载，下一条消息会自动重新加载，因此第一次响应会更慢。

Orbit 当前使用实验性的字节级 tokenizer 和模型架构。很小或很短的训练只能验证流程，无法直接产生通用能力很强的助手。

## 导出与部署

模型页面提供两条明确的导出路径：

- **服务器导出**：生成包含 checkpoint、Orbit 身份元数据、源码、macOS/Linux 与 Windows 启动脚本和 OpenAI 兼容服务的 ZIP，可移动到用户自己的服务器。
- **Ollama 导出**：只有存在兼容的 `~/.orbit/models/<模型名>.gguf` 时才生成基于 GGUF 的 `Modelfile` 包。Orbit 实验性的原生 `.pt` 混合架构不会被伪装成 GGUF；这种格式应使用服务器导出。

## 接入本机智能体

打开“API”页面即可复制地址和示例。默认配置如下：

```json
{
  "provider": "openai-compatible",
  "base_url": "http://127.0.0.1:8765/v1",
  "api_key": "Orbit 页面显示的随机 Key",
  "model": "你的本地模型名称"
}
```

Python 示例：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="Orbit 页面显示的随机 Key",
)

response = client.chat.completions.create(
    model="你的本地模型名称",
    messages=[{"role": "user", "content": "你好"}],
)
print(response.choices[0].message.content)
```

Orbit 会校验每一个 `/v1` 请求。首次启动会创建第一个随机 Key；API 页面可以继续生成多个 Key，并限定为全部模型或某一个 checkpoint。撤销其中一个 Key 不会影响其他智能体。

API 按 OpenAI 的模型列表、非流式 Chat Completions 和非流式 Responses 请求/响应格式工作。把 OpenAI SDK 的 `base_url` 指向 Orbit 本地地址即可，调用这些接口不需要联网。

## 本地存储与隐私

| 内容 | 默认位置 |
| --- | --- |
| 模型与 checkpoint | `~/.orbit/models` |
| 远程训练任务包 | `~/.orbit/jobs` |
| AI 生成的数据集 | `~/.orbit/datasets` |
| 训练历史与每次训练原文 | `~/.orbit/training-runs` |
| 便携模型导出包 | `~/.orbit/exports` |
| 随机 API Keys | `~/.orbit/api-keys.json` |
| 保存的教师 API 设置 | `~/.orbit/teacher-api.json` |
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
