# Orbit

**Orbit** is YUNSH's local AI training platform: train, fine-tune and run your own models on your own computer, then use them through local chat, an OpenAI-compatible local API, and the **Orbit Code** coding agent.

## Three layers

- **Architecture layer** — two in-house research lines:
  - **orbit-hybrid-moe-v1**: Orbit's experimental hybrid-MoE architecture (byte-level tokenizer, GatedMLA, latent MoE with residual depth).
  - **OCA (Orbit Continuum Architecture)**: a research program toward a world-model architecture with persistent spatial state. OCA is a research goal, not a shipped product feature.
- **Training layer** — pretraining, continued pretraining, fine-tuning, datasets, checkpoints and quality boundaries.
- **Surface layer** — the products you use: **Orbit-PC** (desktop mainline), **Orbit-Phone**, and **Orbit-XR** (built into YUNSH OS).

## What it does

- Train a model from scratch, or fine-tune an existing Orbit checkpoint
- Generate training data locally, with or without AI assistance
- Run local inference and an OpenAI-compatible local API (`/v1/chat/completions`, `/v1/responses`)
- Collaborate through reviewed contribution packages and the optional Orbit Hub
- Use one ordered model library across Orbit chat and Orbit Code: API and local models share the same default, while Training keeps its dedicated local-model controls
- Reopen directly into a fresh conversation and an automatically created Orbit workspace; full-access sessions may still work outside that workspace
- Inspect every completed Orbit Code answer through its file-change card, then review the diff or confirm a conflict-checked undo

## Product surfaces

- **Orbit-PC** — desktop training, chat, Orbit Code agent and local API
- **Orbit-Phone** — mobile companion for Orbit
- **Orbit-XR** — spatial desktop integration inside YUNSH OS

## Source availability

Orbit's model implementation and detailed technical documentation are YUNSH internal assets and are not published here. This repository is a product overview.
