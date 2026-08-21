<p align="center">
  <img src="orbit/static/orbit-logo.png" width="180" alt="Orbit logo">
</p>

<p align="center">
  <strong>English</strong> | <a href="README.zh-CN.md">中文</a>
</p>

# Orbit

Orbit is a local AI studio that puts model training, checkpoint management, chat, and an OpenAI-compatible API in one interface. Training data, models, and conversations stay on the user's computer by default.

## What Orbit does

- Opens a draggable desktop workspace with separate Chat, Models, Training, Training History, and API areas.
- Follows the operating-system language by default and can be switched between English and Chinese.
- Supports architecture presets from approximately 300M to 38B parameters.
- Checks local memory before allocating a model.
- Runs a crash-recovering background API on macOS, Linux, and Windows.
- Keeps a native menu-bar or system-tray controller running when its window is closed; only **Quit Orbit** stops the local service.
- Loads weights on demand and automatically unloads an idle model after five minutes.
- Can use DeepSeek or another OpenAI-compatible API to generate a parameter-aware supervised dataset and then start local training. Each provider keeps its own local model, URL, and API key; replacing one key does not affect another provider.
- Supports Chinese, English, or balanced bilingual training data.
- Creates portable community contribution packages for shared models. Contributions are quarantined or held for review and cannot enter a dataset until a local human reviewer approves them.
- Automatically tunes advanced parameters for the selected model size, device, available memory, and amount of local or teacher-generated data. Unsafe local configurations are blocked before allocation and the UI recommends a remote GPU bundle.
- Saves locally trained checkpoints under `~/.orbit/models`.
- Supports custom model names, secondary training from an existing checkpoint, parent-model lineage, and inspectable content/configuration for every training run.
- Keeps the user's custom display name separate from the internal unique checkpoint ID, so timestamp suffixes never replace the name shown in Orbit.
- Preserves the immutable product identity **Orbit, developed by YUNSH**, independently of a user's custom model name or training text, including before the first model is trained and in exported server packages.
- Creates self-contained training bundles for CUDA GPU servers.
- Exports a trained model as a portable Orbit server with the OpenAI-compatible API. An Ollama package can also be generated when a compatible same-name GGUF file is present.
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

The **Check for updates** control inside Orbit checks the official release channel and can update the local Orbit runtime from inside the app. The service restarts automatically after an update; models, conversations, API keys, and training history remain in the user's local data directory.

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

Open the **Training** page, choose a preset, optionally name the model or select a parent checkpoint, paste the user's own text, and select one of three paths:

1. **Train on this computer** starts a local job only after the memory gate passes. The checkpoint stays in `~/.orbit/models`.
2. **Create remote GPU bundle** downloads a ZIP containing the dataset, configuration, Orbit source, and a `run.sh` entry point for a CUDA machine.
3. **Generate data and automatically train** calls DeepSeek or another OpenAI-compatible teacher API. Orbit tells the teacher the selected parameter count, context length, steps, parent model, user goal, and the immutable identity rule that the model is Orbit developed by YUNSH. Every generated dataset also receives real supervised identity examples before training. The sample count controls dataset coverage, generation time, and provider cost. The generated dataset stays local. Each provider's model, URL, and key is saved separately in `~/.orbit/teacher-api.json` with user-only file permissions, so switching back restores that provider and entering a new key replaces only its previous key. Keys are never copied into training history or exports. The goal leaves the computer and the provider may charge for usage, so this path requires explicit confirmation.

Every actual training run writes an inspectable record under `~/.orbit/training-runs`, including the exact dataset, parameters, status, loss, result, model name, and parent model. Selecting **Train again** creates a new checkpoint and a fresh optimizer instead of overwriting the parent.

The Training page offers two stop actions: **Safe stop and keep checkpoint** preserves a resumable partial model, while **Stop and delete unfinished model** removes the incomplete checkpoint and metadata but keeps the training history.

The advanced-parameter panel automatically recommends the safest configuration for the selected model size, device, available memory, sample count, and goal. It shows the estimated total time, time per step, peak memory, and the effect of changing steps, batch size, sequence length, gradient accumulation, and model scale. These are pre-training estimates; once a job starts, the measured step rate and remaining-time ETA replace them. On a typical 24 GB Apple Silicon Mac, the 300M preset with a small bilingual coding/world-knowledge run is the recommended starting point; larger presets may be blocked by the local memory gate or require a remote GPU.

## Collaborative training

The **Community** page lets anyone write an idea, factual source, or dialogue example and export it as an `.orbit-contribution.zip` package. Another user can import the package into a local review queue. Machine pre-screening can quarantine obvious dangerous instructions or personal information; factual material requires an HTTPS source and explicit reviewer verification. Only approved contributions can be assembled into training text.

This workflow does not claim that automatic screening can prove content true or legal. The final reviewer remains responsible for source, rights, privacy, and policy checks, and flagged content cannot be directly approved. Orbit currently exchanges portable packages; it does not silently upload contributions to a central server.

See the bilingual [Community Contribution Policy](COMMUNITY_POLICY.md) for the full review flow and official references.

## Optional Orbit Hub

`server/` contains an optional small-server Hub for accounts, administrator review, and finished-model uploads. The Hub does not train, load, execute, or inspect uploaded model files. Users train locally, then choose whether to upload a finished checkpoint for administrator review or contribute a portable content package. See [server/README.md](server/README.md) for deployment and security boundaries.

The 300M–38B choices describe architecture sizes; they are not pretrained model downloads. Training a useful foundation model from scratch requires a large, carefully prepared dataset and substantial compute.

## Chat locally

After training finishes, open **Chat** and send a message—manual loading is not required. Orbit automatically chooses and loads the local checkpoint, similar to Ollama. Inference does not require an internet connection, and **New chat** clears the visible conversation without changing models or keys. The background API can answer while the web page and desktop app are closed. If weights were unloaded during idle time, the first new request loads them again and may take longer.

Every local inference receives an immutable Orbit/YUNSH identity instruction. Identity questions and identity-overwrite attempts such as “you are Doubao” are answered with Orbit's canonical identity, and old or user-edited model metadata cannot replace it. Training data can change capabilities, not the product identity.

Orbit currently uses a byte-level experimental tokenizer and architecture. A very small or short training run validates the workflow but will not produce a generally capable assistant.

## Export and deployment

The Models page provides two explicit export paths:

- **Server export** creates a ZIP with the selected checkpoint, Orbit identity metadata, source, macOS/Linux and Windows launch scripts, and the local OpenAI-compatible server. It can be moved to the user's own server.
- **Ollama export** creates a GGUF-based `Modelfile` package only when `~/.orbit/models/<model-id>.gguf` exists and is compatible with Ollama. Orbit's experimental native `.pt` hybrid checkpoint is not mislabeled or converted as GGUF; use Server export for that format.

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
| Training history and exact run content | `~/.orbit/training-runs` |
| Portable model exports | `~/.orbit/exports` |
| Random API keys | `~/.orbit/api-keys.json` |
| Saved teacher API settings | `~/.orbit/teacher-api.json` |
| Community contributions and review records | `~/.orbit/community` |
| Isolated Python runtime | `~/.orbit/runtime` |

Orbit does not upload local training text, checkpoints, or chat messages. Users explicitly move a remote training bundle if they choose to train on another machine.

## Updates

Open **Settings** and enable **Automatically update Orbit**. Orbit checks the official release channel in the background. If training is active, the update is queued and waits for the current checkpoint to be saved before restarting the local service. Releases can be applied directly from an older updater-capable version to the latest release; intermediate versions are not required.

The original 0.5.0 runtime predates the updater. It needs one manual installation of a newer Orbit version before in-app updates can take over.

## Private development workspace

Invited developers participate only through the private Orbit project workspace for implementation, review, and testing. The workspace address and access are provided separately by the project owner.

This public GitHub page is not a developer entry point. Developers do not receive access to the GitHub repository, repository secrets, deploy keys, collaborators, GitHub Apps, or release permissions. Changes are reviewed and explicitly accepted by the project owner before publication.

## License

Orbit is available under the [MIT License](LICENSE).
