<p align="center">
  <img src="orbit/static/orbit-logo-transparent.png" width="112" alt="Orbit">
</p>

<p align="center"><a href="README.md">English</a> | <a href="README.zh-CN.md">中文</a></p>

# Orbit 2.0

Orbit is YUNSH's local AI workspace for training your own models, running them locally, and collaborating with an agent that can turn a request into verified work.

## Highlights

- **Train your own model** — pretrain from scratch, continue training, or fine-tune a compatible Orbit checkpoint with local or teacher-assisted datasets.
- **One model library** — organize, rename, reorder, load and unload local models alongside mainstream or custom OpenAI-compatible API models. The first model is the default for Orbit and Orbit Code.
- **Orbit Code agent** — plans before acting, reports discoveries while working, searches files or the web, edits code, runs commands, verifies results and ends with a concise summary.
- **Inspectable execution** — live Thinking/Searching/Editing/Running states, expandable tool calls, elapsed time, step progress and a three-level path from activity to diff to full-file preview.
- **Reviewable changes** — green additions, red removals, per-file counts, completed-answer change cards, review filters and conflict-checked undo with confirmation.
- **Guidance without interruption** — messages sent during a run queue above the composer; promote one to immediate guidance when needed. Queued messages can be removed with confirmation, while consumed guidance cannot be undone.
- **Workspace and permissions** — one reusable Orbit workspace by default, optional project folders, Ask for approval, Auto approve and Full access modes, plus a separate computer-control permission.
- **Local context and memory** — optional project context and local long-term memory. Stable facts can be added automatically or manually, inspected and deleted; secrets are redacted.
- **Files, images and voice** — attach images or audio, use voice input, and preview generated or modified files beside the conversation.
- **Background local service** — closing the window keeps the lightweight agent and OpenAI-compatible API available; explicitly quitting Orbit stops both.
- **Collaboration and plugins** — reviewed contribution packages, optional Orbit Hub workflows and local plugins with the same permission boundary as the agent.

## Product modes

- **Orbit** — local/API conversation, histories, memory and the shared model library.
- **Training** — datasets, pretraining, fine-tuning, checkpoints, local-model operations and collaboration.
- **Orbit Code** — project work, execution history, approvals, diffs, file review and computer tools.

Each mode owns its own history and fixed sidebar actions. The macOS interface uses SF Pro/PingFang typography, a restrained frosted sidebar, translucent controls, concentric squircle geometry and reduced-motion support.

## Local API

Orbit serves an OpenAI-compatible local API on `127.0.0.1` and supports `/v1/chat/completions` and `/v1/responses`. Training data, checkpoints, conversations, configuration and memory remain local by default.

## Platform packages

Orbit 2.0 targets macOS, Windows x64/ARM64 and Linux x64/ARM64. Platform verification and available artifacts are listed in each GitHub Release.

## Architecture and research

- **orbit-hybrid-moe-v1** — Orbit's experimental byte-level hybrid-MoE architecture.
- **OCA (Orbit Continuum Architecture)** — a research direction toward persistent world models; it is not presented as a shipped capability.

## Source availability

Orbit's model implementation and detailed proprietary engineering documentation are YUNSH internal assets. This repository provides the product distribution and public documentation.
