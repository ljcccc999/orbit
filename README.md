<p align="center">
  <img src="orbit/static/orbit-logo.png" width="180" alt="Orbit logo">
</p>

<p align="center">
  <strong>English</strong> | <a href="README.zh-CN.md">中文</a>
</p>

# Orbit

Orbit is a local AI studio that puts model training, checkpoint management, chat, and an OpenAI-compatible API in one browser interface. Training data, models, and conversations stay on the user's computer by default.

## What Orbit does

- Opens a graphical training and chat interface in the browser.
- Supports architecture presets from approximately 300M to 38B parameters.
- Checks local memory before allocating a model.
- Saves locally trained checkpoints under `~/.orbit/models`.
- Creates self-contained training bundles for CUDA GPU servers.
- Serves local checkpoints through `GET /v1/models` and `POST /v1/chat/completions`.
- Lets local agents connect through an OpenAI-compatible configuration.

## Requirements

- More than 10 GB of system memory. More memory is required for larger presets.
- Python 3.9 or newer.
- macOS or Linux. CUDA is recommended for serious training.
- Internet access is needed once to download the installer and dependencies. Training, inference, and the local API can run offline afterward.

## One-line install and launch

```bash
curl -fsSL https://raw.githubusercontent.com/ljcccc999/orbit/main/install.sh | sh
```

The installer checks memory, creates an isolated runtime under `~/.orbit/runtime`, installs Orbit, starts the local service, and opens the browser. Run `orbit` later to open it again.

Orbit listens on `127.0.0.1:8765` by default. It is not exposed to the local network or the internet.

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

The 300M–38B choices describe architecture sizes; they are not pretrained model downloads. Training a useful foundation model from scratch requires a large, carefully prepared dataset and substantial compute.

## Chat locally

After training finishes, open **Chat**, load a checkpoint, and send a message. The model is loaded from local storage and inference does not require an internet connection.

Orbit currently uses a byte-level experimental tokenizer and architecture. A very small or short training run validates the workflow but will not produce a generally capable assistant.

## Connect a local agent

Open the **API** page to copy the endpoint and examples. The default configuration is:

```json
{
  "provider": "openai-compatible",
  "base_url": "http://127.0.0.1:8765/v1",
  "api_key": "orbit-local",
  "model": "your-local-model-id"
}
```

Python example:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="orbit-local",
)

response = client.chat.completions.create(
    model="your-local-model-id",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

The current compatibility layer supports model listing and non-streaming chat completions. The API key value is accepted for client compatibility but is not checked while Orbit is bound to localhost.

## Local storage and privacy

| Content | Default location |
| --- | --- |
| Models and checkpoints | `~/.orbit/models` |
| Remote training bundles | `~/.orbit/jobs` |
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
