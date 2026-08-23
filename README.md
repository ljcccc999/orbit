<p align="center">
  <img src="orbit/static/orbit-logo.png" width="180" alt="Orbit logo">
</p>

<p align="center">
  <strong>English</strong> | <a href="README.zh-CN.md">中文</a>
</p>

# Orbit

> **Everyone can train their own local model.**

Orbit is a local AI studio that puts model training, checkpoint management, chat, and an OpenAI-compatible API in one interface. Training data, models, and conversations stay on the user's computer by default.

## Architecture

The self-developed `orbit-hybrid-moe-v1` architecture:

```text
Plain text
  ↓
UTF-8 byte input
  ↓
256-byte vocabulary Embedding
  ↓
Orbit Backbone
  ├─ DepthResidual
  ├─ DeltaAttention
  ├─ GatedMLA
  ├─ RMSNorm
  └─ LatentMoE
      ├─ 8 routed experts
      ├─ Top-2 experts activated per token
      ├─ 1 shared expert
      └─ SiTUGLU gated feed-forward network
  ↓
Shared Embedding language-model head
  ↓
Predict the next UTF-8 byte
```

The public model boundary is summarized in
[`Orbit模型文档/Orbit模型文档.md`](Orbit模型文档/Orbit模型文档.md). Detailed
internal architecture material is kept outside this public repository.

## Project boundaries

This is the public **Orbit** project. **Orbit-PC** is its desktop mainline for
local training, inference, the desktop app, models, and OpenAI-compatible API.
**Orbit-XR** is the YUNSH OS embedded system integration and is documented separately in
[`Orbit-XR`](https://github.com/ljcccc999/yunsh-os); **Orbit-Phone** is the
iPhone/HarmonyOS mobile surface. Orbit-XR and Orbit-Phone are used together
with YUNSH OS and are maintained with its project boundaries. Their source,
models, releases, logos, and hardware test results are not included in this
public Orbit repository as one mixed implementation; each surface keeps its own
project boundary.

Orbit is organized into three layers: **Architecture** (OCA and
`orbit-hybrid-moe-v1`), **Training** (data, parameters and validation), and
**Surface** (Orbit-PC, Orbit-Phone and Orbit-XR).
See the [three-layer project document](Orbit三层架构项目文档.md) for the file and
project boundaries.

## Orbit layers

- **Architecture:** [OCA](Orbit模型文档/OCA-项目文档.md) (unimplemented world-model
  research) and
  [`orbit-hybrid-moe-v1`](Orbit模型文档/orbit-hybrid-moe-v1-项目文档.md).
- **Training:** [training methods and parameter data](Orbit训练方式与参数数据.md).
- **Surface:** **Orbit-PC** is the desktop mainline;
  [Orbit-XR](https://github.com/ljcccc999/yunsh-os) and Orbit-Phone are
  surfaces used together with YUNSH OS and maintained with the YUNSH OS project.

## What Orbit does

- Opens a draggable desktop workspace with separate Chat, Models, Training, Training History, and API areas.
- Follows the operating-system language by default and can be switched between English and Chinese.
- Supports architecture presets from approximately 300M to 38B parameters.
- Makes local training the primary workflow: every user can start with their own text, choose a model scale, train on their computer, and continue training later.
- Checks local memory before allocating a model.
- Runs a crash-recovering background API on macOS, Linux, and Windows.
- Keeps a native menu-bar or system-tray controller running when its window is closed; only **Quit Orbit** stops the local service.
- Loads weights on demand and automatically unloads an idle model after five minutes.
- Can use DeepSeek or another OpenAI-compatible API to generate a parameter-aware supervised dataset and then start local training. Each provider keeps its own local model, URL, and API key; replacing one key does not affect another provider.
- Shows the complete teacher-generated corpus in the Training page before or during local training, with a scrollable preview and copy action; the same corpus remains inspectable in Training History.
- Supports Chinese, English, or balanced bilingual training data.
- Creates portable community contribution packages for shared models. Contributions are quarantined or held for review and cannot enter a dataset until a local human reviewer approves them.
- Automatically tunes advanced parameters for the selected model size, device, available memory, and amount of local or teacher-generated data. Unsafe local configurations are blocked before allocation and the UI recommends a remote GPU bundle.
- The optimization selector shows concrete recommendations for balanced, time-saving, memory-saving, and quality-first training, including steps, sequence length, estimated time, peak memory, and sample count.
- Uses the same locally generated GPU training package for human-authored and AI-assisted training. The package can be downloaded or directly imported into the user's configured Orbit training server.
- Saves locally trained checkpoints under `~/.orbit/models`.
- Supports custom model names, secondary training from an existing checkpoint, parent-model lineage, and inspectable content/configuration for every training run.
- Keeps the user's custom display name separate from the internal unique checkpoint ID, so timestamp suffixes never replace the name shown in Orbit.
- Preserves the immutable product identity **Orbit, developed by YUNSH**, independently of a user's custom model name or training text, including before the first model is trained and in exported server packages.
- Creates self-contained CUDA GPU training bundles locally, with dataset, configuration, source, identity rules, and `run.sh`.
- Exports a trained model as a portable Orbit server with the OpenAI-compatible API. An Ollama package can also be generated when a compatible same-name GGUF file is present.
- Serves local checkpoints through `GET /v1/models`, `POST /v1/chat/completions`, and `POST /v1/responses`.
- Creates multiple random, revocable API keys that can be restricted to one local model.
- Lets local agents connect through an OpenAI-compatible configuration.

## Orbit Code

Use the product switch at the upper left to move between **Orbit**, **Training**, and **Orbit Code**. Orbit Code is a local-first coding agent that explains its plan before acting, reports new findings while it works, and ends with a result and verification summary.

The desktop shell follows one contextual workspace model: the upper-left switch selects Orbit, Training, or Orbit Code. Each sidebar then shows one create action, that workspace's fixed destinations, and only that workspace's history. The default window is about 900×640. A single translucent sidebar forms the structural layer; the conversation remains a quiet, flat content canvas, while navigation, the composer, and transient menus use restrained Liquid Glass with continuous corners. The whole desktop UI uses PingFang, and the O/T/C workspace marks are plain letterforms without icon tiles. Collapsed execution stages show the live action or a tool-count summary; expansion reveals tools, and a third level shows commands, output, and red/green diffs. Selecting a changed or completed file temporarily creates a split workspace with conversation on the left and a full-file preview on the right; closing it restores the full-width conversation. New Chat always leaves the model editor and starts a fresh Orbit Code session. The model page uses one card: all existing API and local models are mixed in a single source-labelled list, while Add Model expands below that same list. Newly trained local checkpoints appear automatically. Models can be reordered by dragging the right handle; moving one to the top or choosing Set Default makes it the default for new tasks, while the composer can still override the model per task. The left action button expands into a content-sized Liquid Glass menu for default, edit, rename, or delete actions. API and local display names can both be edited without changing a local checkpoint ID. API setup still provides provider presets plus Manual entry for any custom model ID. Launch shows only the Orbit mark, a passing highlight, and a deliberately spaced **By YUNSH** signature—never a loading message or spinner.

- The left rail puts **Plugins** directly below **New conversation**, keeps summarized conversation history, and shows a spinner beside running sessions. Completed sessions preserve an expandable execution record.
- The composer accepts files, images, and voice input. While a task runs, a new message is queued by default; promoting it changes the message into immediate guidance without stopping the current task. Queued messages require confirmation before cancellation, and promoted guidance cannot be undone.
- A liquid-glass progress pill shows completed and total steps, elapsed activity, and clickable green additions/red deletions. The right-side review drawer filters added, modified, and deleted files.
- Execution history expands through three levels without hiding the Agent's written update. While work is active, the first action row names the current operation; after completion it becomes a count such as files edited or web searches performed. Expanding it lists every completed/current action in that stage, and expanding an action reveals the full Shell output, search content, mouse/keyboard result, or red/green diff. A changed file can then open as a complete file in the right drawer; deleted files use the archived pre-change content.
- **Intelligence** is a real execution budget, not only a label. Higher levels allow more agent turns, searches, checks, verification, time, and output tokens. The system keeps the stable instruction prefix and append-only task history to improve provider prompt-cache reuse.
- Permission choices are **Ask for approval**, **Approve workspace actions**, and **Full access**. Destructive or out-of-scope actions still pass through the local permission flow. Mouse and keyboard control is a separate, default-off setting: the agent prefers files, APIs, and command-line tools and uses UI control only when no reliable programmatic route is available. Every attempted action appears in the execution timeline.
- Closing the desktop window leaves the Agent and local API running. Only **Quit Orbit (Stops Agent & API)** in the status/tray menu shuts down the Agent and releases the local API port. Settings can prevent sleep while Orbit works and enable or disable login-time background startup.
- Every task starts with a stable Agent system prompt that identifies Orbit Code as an Orbit product developed by YUNSH and defines its planning, evidence, safety, execution, verification, and reporting behavior.
- Model connections include local Orbit checkpoints plus international and Chinese API providers: OpenAI, Anthropic, Gemini, xAI, Groq, Mistral, OpenRouter, DeepSeek, Kimi, Zhipu GLM, Alibaba Cloud Model Studio/Qwen, MiniMax, and a single custom OpenAI-compatible option for other services. Only providers with a straightforward compatible API and stable public model IDs receive dedicated presets; endpoint-based or account-specific services stay under the custom option. Orbit Code always lists verified local checkpoints alongside saved API profiles, keeps its workspace path in Advanced, and never exposes complete saved keys.
- The visual system uses optical Apple system typography, layered translucent materials, and interruptible spring transitions, with explicit Reduce Motion and Reduce Transparency fallbacks.
- Local plugins are imported from a `plugin.json` manifest, enabled per device, and added to new-task context. Their actions remain subject to the same Orbit Code permissions.

## Requirements

- More than 10 GB of system memory. More memory is required for larger presets.
- macOS, Linux, or Windows. CUDA is recommended for serious training.
- No preinstalled Python is required by the installers. Orbit uses a suitable existing Python or prepares a private Python 3.11 runtime.
- Internet access is needed once to download the installer and dependencies. Training, inference, and the local API can run offline afterward.

## One-line install and launch

### Desktop apps

Download the current [macOS universal app](https://github.com/ljcccc999/orbit/releases/latest/download/Orbit-macOS-universal.zip), [Windows x64 app](https://github.com/ljcccc999/orbit/releases/latest/download/Orbit-Windows-x64.exe), [Windows ARM64 app](https://github.com/ljcccc999/orbit/releases/latest/download/Orbit-Windows-arm64.exe), [Linux x64 AppImage](https://github.com/ljcccc999/orbit/releases/latest/download/Orbit-Linux-x64.AppImage), or [Linux ARM64 AppImage](https://github.com/ljcccc999/orbit/releases/latest/download/Orbit-Linux-arm64.AppImage). Open the app and it prepares the local runtime without opening a terminal.

After the first launch, Orbit starts with the user's login. Closing or minimizing the window leaves Orbit in the macOS menu bar, Windows notification area, or Linux system tray and keeps the local API available. Choose **Quit Orbit** from that menu to stop the API and disable automatic startup; manually opening Orbit again re-enables it.

The lower-left update controls have separate actions: **Check for updates** only checks the official release channel, while **Install now** downloads, verifies, and queues the latest release from inside Orbit. If training is active, installation waits for a safe checkpoint and then restarts the local service; models, conversations, API keys, and training history remain in the user's local data directory.

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

Open the **Training** page, choose a preset, optionally name the model or select a parent checkpoint, paste your own text, and choose how to train:

- **Pretraining (from scratch)** starts with random Orbit weights. **Fine-tuning** can continue an already-trained local Orbit checkpoint or a downloaded, validated Orbit-compatible checkpoint; the Training page separates local and download sources and preserves the parent lineage. Orbit never silently changes one mode into the other.
- Manual documents are shown with an estimated sample count based on UTF-8 corpus size and sequence length. The page warns about small datasets, overfitting, excessive context, unsafe memory pressure, and unrealistic pretraining goals, then suggests safer settings.
- The four optimization choices recommend configuration from the actual token budget, model scale, device and memory. AI assistance accepts 1–10,000,000 samples per round; this is an operational safety cap, not a complete pretraining corpus. A manually entered count is never silently replaced by a recommendation. Generate larger corpora in reviewed rounds, and keep manual text and AI-generated text separately viewable. See [`Orbit训练方式与参数数据.md`](Orbit训练方式与参数数据.md) for the unified training reference.
- After a reset, the advanced fields use one internally consistent **Balanced** profile instead of mixing a placeholder sample count with a different step recommendation. Manual advanced-parameter edits do not alter the four saved recommendations; selecting a profile again, or pressing **Apply this recommendation again**, restores that profile.

Orbit reports three different units instead of treating them as interchangeable: a **sample** is one independently delimited document/task accepted from the teacher, a **training token** is one UTF-8 byte in the current `orbit-byte-v1` tokenizer, and a **step** is one optimizer update after batch and gradient accumulation. Character count, byte count, token count and optimizer steps are therefore shown separately. Teacher batches must contain exact sample boundaries, pass language/length/control-character checks, and survive exact and near-duplicate filtering; malformed batches are retried and only accepted samples increase progress. A deterministic 5% sample-level split is held out from weight updates. Structural checks do not independently prove factual correctness, so the UI labels generated facts as not independently reviewed.

For scale: a model trained from random weights is a foundation-model project, not a normal local chat fine-tune. Orbit uses the Chinchilla planning reference of about 20 training tokens per parameter: approximately 6 billion tokens for 300M, 20 billion for 1B, 60 billion for 3B, 140 billion for 7B, 280 billion for 14B and 760 billion for 38B. These are planning references, not guarantees, and they do not prescribe fixed sample counts or optimizer steps. The UI also shows one Llama 3 data-rich scale example (about 37 tokens/parameter) as a comparison, not a universal target. The practical chat route is a genuinely pretrained compatible base followed by high-quality SFT; LIMA's 1,000 curated examples are a research starting point, not a universal minimum. Larger AI corpora are generated in reviewed rounds under the 10,000,000-per-round safety cap.

The Training page now separates the current run result from the research reference. Final optimizer-token coverage is calculated as `steps × sequence length × batch size × gradient accumulation`; for example, 2,000 steps × 512 × 1 × 1 = 1,024,000 tokens. The selected scale's reference is shown beside it (300M ≈ 6,000,000,000 tokens), together with the percentage deviation; closer to 0% means closer to that pretraining reference. The four goals remain available, and the default is Balanced. Editing an advanced parameter updates the displayed result without silently changing the selected architecture or the four recommendation baselines. Fine-tuning shows coverage but does not pretend that a universal pretraining token standard applies to SFT.

The page also shows separate writing guidance: pretraining rounds 1–N should be document- and code-first with only a small dialogue/identity tail; fine-tuning should dynamically mix specialized documents, single-turn labeled tasks and high-quality instruction dialogue according to validation results instead of enforcing a fixed percentage. Structured tasks include classification/sentiment, NER, code/SQL generation, summarization, and expansion/polishing; they are input→output examples, not chat bubbles. External or downloaded models cannot be trained from random initialization and are restricted to fine-tuning.

1. **Train on this computer** starts a local job only after the memory gate passes. The checkpoint stays in `~/.orbit/models`.
2. **Generate a GPU training package locally** creates a ZIP containing the dataset, configuration, Orbit source, identity rules, and a `run.sh` entry point for a CUDA machine. Human-authored and AI-assisted training use this same package format. You can download it or choose **Generate locally and import to server** after logging in to Orbit Hub.
3. **Generate data and automatically train** calls DeepSeek or another OpenAI-compatible teacher API, then starts training on this computer. Orbit tells the teacher the selected parameter count, context length, steps, parent model, user goal, and the immutable identity rule that the model is Orbit developed by YUNSH. Every generated dataset also receives real supervised identity examples before training. The sample count controls dataset coverage, generation time, and provider cost. The generated dataset stays local. Each provider's model, URL, and key is saved separately in `~/.orbit/teacher-api.json` with user-only file permissions, so switching back restores that provider and entering a new key replaces only its previous key. Keys are never copied into training history, server uploads, or exports. The goal leaves the computer and the provider may charge for usage, so this path requires explicit confirmation.

Every actual training run writes an inspectable record under `~/.orbit/training-runs`, including the exact dataset, parameters, status, loss, result, model name, and parent model. Selecting **Train again** creates a new checkpoint and a fresh optimizer instead of overwriting the parent.

The Training page offers two stop actions: **Safe stop and keep checkpoint** preserves a resumable partial model, while **Stop and delete unfinished model** removes the incomplete checkpoint and metadata but keeps the training history.

The advanced-parameter panel automatically recommends the safest configuration for the selected model size, device, available memory, sample count, and goal. It shows the estimated total time, time per step, peak memory, and the effect of changing steps, batch size, sequence length, gradient accumulation, and model scale. These are pre-training estimates; once a job starts, the measured step rate and remaining-time ETA replace them. On a typical 24 GB Apple Silicon Mac, the 300M preset with a small bilingual coding/world-knowledge run is the recommended starting point; larger presets may be blocked by the local memory gate or require a remote GPU.

On Apple Silicon, the local MPS path uses fused causal attention and sparse top-k expert evaluation. The automatic 300M starting profile is bounded to a 512-token context and one accumulation pass so a first run does not silently expand into an all-day job. MPS automatic precision stays FP32 because FP16 overflows the freshly initialized 300M output head on the tested stack; fused FP32 is still substantially faster than the old recurrent path. BF16 and FP16 remain explicit choices for hardware/configurations where they are stable.

## Collaborative training

The **Community** page lets anyone write an idea, factual source, or dialogue example and export it as an `.orbit-contribution.zip` package. Another user can import the package into a local review queue. Machine pre-screening can quarantine obvious dangerous instructions or personal information; factual material requires an HTTPS source and explicit reviewer verification. Only approved contributions can be assembled into training text.

This workflow does not claim that automatic screening can prove content true or legal. The final reviewer remains responsible for source, rights, privacy, and policy checks, and flagged content cannot be directly approved. Orbit currently exchanges portable packages; it does not silently upload contributions to a central server.

See the bilingual [Community Contribution Policy](COMMUNITY_POLICY.md) for the full review flow and official references.

## Optional Orbit Hub

`server/` contains an optional small-server Hub for accounts, administrator review, finished-model uploads, and locally generated GPU training packages. Orbit creates the package on the user's computer and imports it to the configured server with chunking and checksum verification. The Hub stores it for the user's GPU workflow but never executes or loads an uploaded file automatically. See [server/README.md](server/README.md) for deployment and security boundaries.

The 300M–38B choices describe Orbit architecture sizes. The Training page can download an HTTPS Orbit checkpoint, validate its architecture, and add it to the local base-model list. External Hugging Face/Qwen checkpoints are not directly compatible with the current byte-level Orbit trainer yet; a real tokenizer, architecture, and weight-format adapter is required before they can be offered as fine-tuning bases. Training a useful foundation model from scratch requires a large, carefully prepared dataset and substantial compute.

## Chat locally

After training finishes, open **Chat** and send a message—manual loading is not required. Orbit automatically chooses and loads the local checkpoint, similar to Ollama. Inference does not require an internet connection, and **New chat** clears the visible conversation without changing models or keys. The background API can answer while the web page and desktop app are closed. If weights were unloaded during idle time, the first new request loads them again and may take longer.

Orbit identity is inserted into every first and subsequent training corpus as actual identity and dialogue examples, and it is included in exports. The inference worker does **not** detect identity questions or replace answers with hard-coded text. Whether an exported checkpoint answers correctly therefore depends on what the model actually learned; a tiny or failed training run may still answer incorrectly.

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

Orbit does not upload local training text, checkpoints, or chat messages automatically. A GPU package is sent to a server only after the user explicitly chooses the import action; the teacher API key is never included in that package.

## Updates

The AI-generated training-content preview is independently scrollable with a mouse wheel or trackpad, including large teacher-generated corpora.

Open **Settings** and enable **Automatically update Orbit**. Orbit checks the official release channel in the background. If training is active, the update is queued and waits for the current checkpoint to be saved before restarting the local service. Releases can be applied directly from an older updater-capable version to the latest release; intermediate versions are not required.

The original 0.5.0 runtime predates the updater. It needs one manual installation of a newer Orbit version before in-app updates can take over.

After replacing the runtime and desktop bundle, the updater forces one final background-service reload and waits for the new service to become healthy. This prevents an older KeepAlive process from serving stale in-memory UI code under the new version number.

## Private development workspace

Invited developers participate only through the private Orbit project workspace for implementation, review, and testing. The workspace address and access are provided separately by the project owner.

This public GitHub page is not a developer entry point. Developers do not receive access to the GitHub repository, repository secrets, deploy keys, collaborators, GitHub Apps, or release permissions. Changes are reviewed and explicitly accepted by the project owner before publication.

## License

Orbit is available under the [MIT License](LICENSE).
