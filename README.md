<p align="center">
  <img src="orbit/static/orbit-logo.png" width="180" alt="Orbit logo">
</p>

<p align="center">
  <strong>English</strong> | <a href="README.zh-CN.md">中文</a>
</p>

# Orbit

Orbit is a local AI studio that puts model training, checkpoint management, chat, and an OpenAI-compatible API in one interface. Training data, models, and conversations stay on the user's computer by default.

## What Orbit does

- Opens a graphical training and chat interface in the browser.
- Supports architecture presets from approximately 300M to 38B parameters.
- Checks local memory before allocating a model.
- Runs a crash-recovering background API on macOS, Linux, and Windows.
- Keeps a native menu-bar or system-tray controller running when its window is closed; only **Quit Orbit** stops the local service.
- Loads weights on demand and automatically unloads an idle model after five minutes.
- Can use DeepSeek or another OpenAI-compatible API to generate a supervised dataset and then start local training.
- Saves locally trained checkpoints under `~/.orbit/models`.
- Creates self-contained training bundles for CUDA GPU servers.
- Serves local checkpoints through `GET /v1/models`, `POST /v1/chat/completions`, and `POST /v1/responses`.
- Creates multiple random, revocable API keys that can be restricted to one local model.
- Lets local agents connect through an OpenAI-compatible configuration.

## Requirements

- More than 10 GB of system memory. More memory is required for larger presets.
- macOS, Linux, or Windows. CUDA is recommended for serious training.
- No preinstalled Python is required by the installers. Orbit uses a suitable existing Python or prepares a private Python 3.11 runtime.
- Internet access is needed once to download the installer and dependencies. Training, inference, and the local API can run offline afterward.

## One-line install and launch

### Desktop apps

Download the current [macOS universal app](https://github.com/ljcccc999/orbit/releases/latest/download/Orbit-macOS-universal.zip), [Windows x64 app](https://github.com/ljcccc999/orbit/releases/latest/download/Orbit-Windows-x64.exe), [Windows ARM64 app](https://github.com/ljcccc999/orbit/releases/latest/download/Orbit-Windows-arm64.exe), [Linux x64 AppImage](https://github.com/ljcccc999/orbit/releases/latest/download/Orbit-Linux-x64.AppImage), or [Linux ARM64 AppImage](https://github.com/ljcccc999/orbit/releases/latest/download/Orbit-Linux-arm64.AppImage). Open the app and it prepares the local runtime without opening a terminal.

After the first launch, Orbit starts with the user's login. Closing or minimizing the window leaves Orbit in the macOS menu bar, Windows notification area, or Linux system tray and keeps the local API available. Choose **Quit Orbit** from that menu to stop the API and disable automatic startup; manually opening Orbit again re-enables it.

The current public builds are not Developer ID/Authenticode signed or notarized. macOS Gatekeeper or Windows SmartScreen may therefore require an explicit first-open confirmation. Removing that system warning requires platform signing certificates; it is not something application code can safely bypass.

### macOS and Linux command line

```bash
curl --retry 5 --retry-delay 2 --connect-timeout 20 -fsSL https://raw.githubusercontent.com/ljcccc999/orbit/main/install.sh | sh
```

The installer checks memory, creates an isolated runtime under `~/.orbit/runtime`, installs Orbit, starts the local service, and opens the browser. Run `orbit` later to open it again. The web page is only a management client: closing it does not stop the OpenAI-compatible API.

The command is safe to run again after an interrupted download. It resumes the existing isolated runtime and retries network downloads automatically.

Orbit listens on `127.0.0.1:8765` by default. It is not exposed to the local network or the internet.

```bash
orbit start
orbit status
orbit chat "Hello"
orbit unload
orbit stop
```

The macOS LaunchAgent and Linux user-level systemd service restart Orbit after a crash. `orbit unload` releases the active weights immediately; otherwise the default idle timer does this automatically.

## Model sizes

| Preset | Approximate parameters | Conservative training-memory estimate | Intended environment |
| --- | ---: | ---: | --- |
| 300M | 0.308B | 9.9 GB | capable personal computer |
| 1B | 1.063B | 34.0 GB | high-memory workstation or GPU |
| 3B | 2.969B | 95.0 GB | workstation or multi-GPU server |
| 7B | 7.179B | 229.7 GB | multi-GPU server |
| 14B | 14.075B | 450.4 GB | distributed training system |
| 38B | 38.260B | 1,224.3 GB | large distributed GPU cluster |

These are conservative full-training estimates that include optimizer state and working memory. Actual requirements depend on precision, optimizer, activation checkpointing, sharding, sequence length, and hardware.

## Training

Open the **Training** page, choose a preset, paste the user's own text, and select one of two paths:

1. **Train on this computer** starts a local job only after the memory gate passes. The checkpoint stays in `~/.orbit/models`.
2. **Create remote GPU bundle** downloads a ZIP containing the dataset, configuration, Orbit source, and a `run.sh` entry point for a CUDA machine.
3. **Generate data and automatically train** calls DeepSeek or another OpenAI-compatible teacher API, stores the generated dataset locally, and then enters the same guarded local training path. The API key is kept only in request memory. The training goal leaves the computer and the provider may charge for usage, so this path requires explicit confirmation.

The 300M–38B choices describe architecture sizes; they are not pretrained model downloads. Training a useful foundation model from scratch requires a large, carefully prepared dataset and substantial compute.

## Chat locally

After training finishes, open **Chat**, load a checkpoint, and send a message. The model is loaded from local storage and inference does not require an internet connection. The background API can answer while the web page and desktop app are closed. If weights were unloaded during idle time, the first new request loads them again and may take longer.

Orbit currently uses a byte-level experimental tokenizer and architecture. A very small or short training run validates the workflow but will not produce a generally capable assistant.

## Connect a local agent

Open the **API** page to copy the endpoint and examples. The default configuration is:

```json
{
  "provider": "openai-compatible",
  "base_url": "http://127.0.0.1:8765/v1",
  "api_key": "the-random-key-shown-by-orbit",
  "model": "your-local-model-id"
}
```

Python example:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="the-random-key-shown-by-orbit",
)

response = client.chat.completions.create(
    model="your-local-model-id",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

Orbit authenticates every `/v1` request. The first random key is created on first launch; additional keys can be created in the API page and scoped to all models or one checkpoint. Revoking one key does not interrupt other agents.

The API follows the OpenAI request and response shapes for model listing, non-streaming Chat Completions, and non-streaming Responses. Point an OpenAI SDK at Orbit's local `base_url`; no internet connection is used for those requests.

## Local storage and privacy

| Content | Default location |
| --- | --- |
| Models and checkpoints | `~/.orbit/models` |
| Remote training bundles | `~/.orbit/jobs` |
| AI-generated datasets | `~/.orbit/datasets` |
| Random API keys | `~/.orbit/api-keys.json` |
| Isolated Python runtime | `~/.orbit/runtime` |

Orbit does not upload local training text, checkpoints, or chat messages. Users explicitly move a remote training bundle if they choose to train on another machine.

## Development

```bash
git clone https://github.com/ljcccc999/orbit.git
cd orbit
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q
.venv/bin/orbit
```

## License

Orbit is available under the [MIT License](LICENSE).
