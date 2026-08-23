# Orbit Continuum Architecture (OCA)
## 持续世界状态与因果想象架构

**研究预印本 / Research preprint**  
**Version 0.1 · 2026-08-03**  
**Project:** Orbit  

**作者 / Authors:** YUNSH · 刘家成（中国，14 岁初中生） / YUNSH · Liu Jiacheng (China, age 14, middle-school student)  

**实现声明 / Implementation statement:** 我们已经在 Apple Silicon Mac mini 上完成了 Orbit Continuum Architecture（OCA）7B 配置的 FP16 权重初始化、4-bit 量化和 Metal 前向验证。该 checkpoint 用于验证架构和部署链路；它尚未经过语言预训练，因此不等同于一个已经具备成熟知识和对话能力的基础模型。  
We have completed FP16 weight initialization, 4-bit quantization, and Metal forward validation of the Orbit Continuum Architecture (OCA) 7B configuration on an Apple Silicon Mac mini. The checkpoint validates the architecture and deployment path; it has not undergone language pretraining and is therefore not a finished foundation model with mature knowledge or conversational ability.

> 本文描述一个已经实现了可运行原型的研究架构。文中“7B”表示模型配置的逻辑参数规模；当前 7B checkpoint 是随机初始化后量化的结构验证权重，尚未完成语言预训练，因此不能被描述为已经具备成熟对话能力。

> This paper describes a research architecture with a working implementation. “7B” denotes the logical parameter scale of the model configuration. The current 7B checkpoint contains quantized random initialization for structural validation and has not undergone language pretraining; it must not be described as a finished conversational model.

---

## 摘要 / Abstract

### 中文

当前主流语言模型通常把世界信息压缩到上下文序列和注意力缓存中，并以预测下一个 token 作为主要训练目标。这种方案能够产生流畅语言，但并不要求模型显式维护对象、时间、隐藏状态、动作后果或观察不确定性。本文提出 Orbit Continuum Architecture（OCA），一种面向持续世界状态的语言模型架构。OCA 将模型拆分为感知、对象槽位绑定、连续状态更新、动作条件状态转移、未来状态想象和语言解码六个部分。其核心状态变量独立于 token KV cache，用门控更新机制维护跨观察时刻的世界状态；竞争式对象槽位为对象身份和属性提供可检查的潜在瓶颈；因果转移模块在给定动作后滚动预测多个未来状态。

本文给出 OCA 的数学定义、工程实现、Mac/MLX 部署方式、7B 配置和验证标准。当前原型已在 Mac mini 的 Apple Silicon 统一内存上完成 OCA 7B 配置的 FP16 初始化、4-bit 量化、Metal 前向推理以及连续状态传递测试。本文不声称 OCA 已经解决通用世界理解，而是提出一套可证伪的架构假设和实验路线：如果显式的持久状态、对象绑定和因果想象确实有价值，则在参数规模匹配的隐藏物体追踪、干预预测和多步规划任务上，应优于只使用语言预测的基线模型。

### English

Mainstream language models typically compress information about the world into a context sequence and an attention KV cache, with next-token prediction as the dominant training objective. This produces fluent language, but does not require an explicit account of objects, time, hidden state, action consequences, or observational uncertainty. We propose the Orbit Continuum Architecture (OCA), a language-model architecture centered on persistent world state. OCA separates perception, object-slot binding, continuum-state updating, action-conditioned state transition, future-state imagination, and language decoding. Its central state variable is distinct from the token KV cache and is maintained across observations through gated updates. Competitive object slots provide an inspectable latent bottleneck for object identity and attributes. A causal transition module rolls the state forward under an explicit action to produce multiple imagined futures.

This paper defines OCA mathematically, describes its implementation, documents Apple Silicon/MLX deployment, specifies the 7B configuration, and proposes validation criteria. On an Apple Silicon Mac mini, the current prototype has completed FP16 initialization of the OCA 7B configuration, 4-bit quantization, Metal inference, and persistent-state handoff tests. We do not claim that OCA has solved general world understanding. Instead, we provide falsifiable architectural hypotheses and an experimental program: if explicit persistent state, object binding, and causal imagination are useful, OCA should outperform a parameter-matched language-only baseline on hidden-object tracking, intervention prediction, and multi-step planning.

**关键词 / Keywords:** Orbit, OCA, world model, persistent state, object slots, causal imagination, Apple Silicon, MLX

---

## 1. 引言 / Introduction

### 中文

语言是世界的一个投影，而不是世界本身。一个人说“杯子被挡住了”，隐含表达了至少三类信息：杯子仍然存在；它的可见性发生了变化；如果移开遮挡物，杯子可能重新出现。只优化下一个 token 的模型可以通过统计相关性回答类似问题，但它没有被强制要求保存一个稳定的对象状态，也没有被强制要求区分“物体消失”和“物体暂时不可见”。

OCA 的出发点不是否定 Transformer，而是重新划分其职责。注意力适合处理当前观察中的关系，语言解码器适合表达和交互；然而，持续对象、时间变化和动作后果需要更明确的内部变量。OCA 因此保留 Transformer block，同时加入一个可传递、可重置、可观察的 continuum state（连续世界状态）。

本文将“理解世界”定义为一组可测试能力，而不是一句宣传语：在遮挡下保持对象存在性；理解事件顺序；预测动作引起的状态变化；保留未受干预对象的状态；在新观察与旧记忆冲突时更新信念；在不确定信息下表达校准后的置信度。只有在这些测试中取得可复现结果，才可以进一步讨论更强的世界模型能力。

### English

Language is a projection of the world, not the world itself. When a person says “the cup is occluded,” the statement implies at least three facts: the cup continues to exist, its visibility has changed, and it may reappear if the occluder is removed. A next-token model can answer such questions through statistical correlation, but it is not required to maintain a stable object state or distinguish disappearance from temporary unobservability.

OCA does not reject the Transformer. It reallocates responsibilities. Attention is useful for relations within a current observation, and a language decoder is useful for expression and interaction. Persistent objects, temporal change, and action consequences require more explicit internal variables. OCA therefore retains Transformer blocks while adding a transferable, resettable, and inspectable continuum state.

We define “world understanding” as a set of testable abilities rather than a marketing phrase: preserve object permanence under occlusion; represent event order; predict action-induced state changes; preserve unaffected objects after interventions; revise beliefs when new observations conflict with memory; and calibrate confidence under incomplete information. Reproducible success on these tests is required before making stronger claims.

---

## 2. 架构概述 / Architecture Overview

OCA 的数据流如下：

```text
观察 token O_t
      ↓
感知 Transformer P
      ↓
竞争式对象槽位绑定 B
      ↓
连续状态更新 U(S_{t-1}, ·)
      ↓
当前世界状态 S_t ─────→ 语言解码器 D → 输出 token
      ↓
动作条件转移 T(S_t, A_t)
      ↓
未来状态 Ŝ_{t+1}, Ŝ_{t+2}, …, Ŝ_{t+k}
```

The OCA data flow is:

```text
observation tokens O_t
        ↓
perception Transformer P
        ↓
competitive object-slot binding B
        ↓
continuum update U(S_{t-1}, ·)
        ↓
present world state S_t ─────→ language decoder D → output tokens
        ↓
action-conditioned transition T(S_t, A_t)
        ↓
future states Ŝ_{t+1}, Ŝ_{t+2}, …, Ŝ_{t+k}
```

### 2.1 感知 / Perception

给定 token 序列 `X_t`，感知模块加入 token embedding 和位置 embedding，并通过非因果 self-attention 聚合整个观察窗口：

Given token sequence `X_t`, the perception module adds token and positional embeddings and applies non-causal self-attention over the observation window:

\[
H_t = P(X_t) = P_L(\cdots P_2(P_1(E(X_t) + R))).
\]

非因果感知并不等于模型可以偷看未来世界；它只表示当前输入窗口内部的 token 可以互相读取。时间上的未来由独立的状态转移模块负责。

Non-causal perception does not allow the model to read the future world. It only permits tokens within the current observation window to interact. Temporal futures are produced by the separate transition module.

### 2.2 对象槽位绑定 / Object-slot Binding

OCA 维护固定数量的 latent slots。每个 slot 不是预先指定的物体类别，而是一个可学习的对象假设。模型通过多轮竞争式注意力让 observation features 竞争 slots：

OCA maintains a fixed number of latent slots. A slot is not assigned a predefined object category; it is a learnable object hypothesis. Repeated competitive attention lets observation features compete for slots:

\[
A^{(i)} = \operatorname{softmax}_{slots}
\left(\frac{Q(S^{(i)})K(H_t)^T}{\sqrt d}\right),
\]

\[
S^{(i+1)} = G^{(i)} \odot S^{(i)} + (1-G^{(i)}) \odot
\tanh\left(C([S^{(i)}; A^{(i)}V(H_t)])\right).
\]

槽位数量是一个明确的容量假设：太少会导致对象混叠，太多会带来冗余和训练不稳定。槽位可以被读取和单独监督，因此比完全分散在 token 表示中的对象信息更容易进行诊断。

The number of slots is an explicit capacity assumption: too few slots cause object entanglement, while too many can create redundancy and instability. Because slots can be read and supervised individually, they are easier to diagnose than object information distributed entirely across token representations.

### 2.3 连续世界状态 / Continuum State

设上一时刻状态为 `S_{t-1}`，当前观察绑定出的槽位为 `B_t`。连续状态更新器使用更新门、重置门和候选状态：

Let the previous state be `S_{t-1}` and the currently bound slots be `B_t`. The continuum updater uses an update gate, reset gate, and candidate state:

\[
z_t = \sigma(W_z[S_{t-1};B_t]),
\quad r_t = \sigma(W_r[S_{t-1};B_t]),
\]

\[
\tilde S_t = \tanh(W_c[r_t\odot S_{t-1};B_t]),
\quad S_t = (1-z_t)\odot S_{t-1}+z_t\odot\tilde S_t.
\]

该状态和 KV cache 有不同生命周期。KV cache 服务于一次上下文中的高效注意力；continuum state 表达跨观察时刻的世界假设。应用可以在会话开始时初始化它，在独立任务之间显式重置它，也可以将它序列化保存为工作区状态。

This state has a different lifecycle from the KV cache. The KV cache serves efficient attention within a context, while the continuum state represents a world hypothesis across observations. An application can initialize it at session start, explicitly reset it between independent tasks, or serialize it as workspace state.

### 2.4 因果想象 / Causal Imagination

给定当前状态 `S_t` 和动作表示 `A_t`，转移模块生成未来状态：

Given current state `S_t` and an action representation `A_t`, the transition module generates future states:

\[
\hat S_{t+1} = T(S_t,A_t),
\qquad
\hat S_{t+k} = T(\hat S_{t+k-1},A_t).
\]

每次转移都包含一个 confidence gate，用于限制不确定的预测对状态的修改幅度。当前实现默认进行六步想象。多步 rollout 的意义不是生成好看的内部动画，而是让模型在输出或执行动作前比较可能的后果。

Each transition includes a confidence gate that limits how much an uncertain prediction can modify the state. The current 7B configuration uses six imagination steps by default. The purpose of rollout is not to create visually pleasing internal animation; it is to compare consequences before producing an answer or executing an action.

### 2.5 语言解码 / Language Decoding

语言解码器接收感知特征，并注入当前 continuum state 的 pooled context：

The language decoder receives perception features and injects a pooled context derived from the current continuum state:

\[
Y_t = D(H_t + W_s\operatorname{pool}(S_t)),
\qquad
\text{logits}_t = W_o\operatorname{RMSNorm}(Y_t).
\]

这使语言输出能够读取世界状态，但不要求语言序列本身承担全部状态存储责任。

This allows language generation to read the world state without requiring the language sequence itself to carry the entire state representation.

---

## 3. 训练目标 / Training Objective

OCA 的目标函数设计为多任务形式：

OCA is intended to use a multi-objective loss:

\[
\mathcal L = \mathcal L_{lang}
+ \lambda_s\mathcal L_{state}
+ \lambda_d\mathcal L_{dyn}
+ \lambda_c\mathcal L_{cycle}
+ \lambda_u\mathcal L_{uncertainty}.
\]

1. `L_lang`：标准 next-token 与指令响应损失。 / Standard next-token and instruction-response loss.
2. `L_state`：从 slots 重建对象属性、位置和可见性。 / Reconstruct object attributes, locations, and visibility from slots.
3. `L_dyn`：在给定动作后预测真实下一状态。 / Predict the true next state after an action.
4. `L_cycle`：动作只应改变相关对象，避免无关状态漂移。 / Penalize changes to unrelated objects.
5. `L_uncertainty`：在遮挡或冲突观察下校准置信度。 / Calibrate confidence under occlusion or conflicting observations.

当前仓库已经具备 `L_lang` 的 MLX 冒烟训练入口和合成世界生成器；`L_state`、`L_dyn`、`L_cycle` 和 `L_uncertainty` 是下一阶段训练实现重点。这个边界很重要：结构已经能运行，完整世界模型训练尚未完成。

The repository currently contains an MLX language-loss smoke trainer and a synthetic-world generator. Implementing `L_state`, `L_dyn`, `L_cycle`, and `L_uncertainty` is the next training milestone. This boundary matters: the architecture runs, but full world-model training is not complete.

---

## 4. 当前实现 / Current Implementation

### 中文

原型使用 Apple Silicon 上的 MLX 实现，支持从 Tiny 到 7B 的配置。7B 配置使用 4096 hidden width、10 个 perception blocks、12 个 decoder blocks、5 个 transition blocks、32 个 object slots、32 个 attention heads 和 6 步 imagination rollout。逻辑参数估算为 7,306,608,640。FP16 初始化后可以通过 MLX 4-bit 量化，当前本地量化 checkpoint 约 3.9GB。

当前已验证：

- 7B OCA 模型可以在 Metal 上完成量化前向传播；
- logits 形状为 `(batch, sequence, 65536)`；
- continuum state 形状为 `(batch, 32, 4096)`；
- imagined states 形状为 `(batch, 6, 32, 4096)`；
- 状态可以从一个 observation 传递到下一个 observation；
- Tiny 配置的配置校验、世界动作生成和基础测试通过。

### English

The prototype is implemented in MLX for Apple Silicon and supports Tiny through 7B configurations. The 7B configuration uses a 4096-wide latent space, 10 perception blocks, 12 decoder blocks, 5 transition blocks, 32 object slots, 32 attention heads, and six imagination steps. The logical parameter estimate is 7,306,608,640. After FP16 initialization, the model can be quantized to 4-bit; the current local quantized checkpoint is approximately 3.9 GB.

Verified properties include:

- the quantized 7B OCA model completes a Metal forward pass;
- logits have shape `(batch, sequence, 65536)`;
- continuum state has shape `(batch, 32, 4096)`;
- imagined states have shape `(batch, 6, 32, 4096)`;
- state can be handed from one observation to the next;
- configuration checks, synthetic action generation, and basic Tiny tests pass.

These results validate software execution and tensor structure, not intelligence or generalization.

### 4.1 具体实现方法 / Concrete Implementation Method

#### 中文

OCA 的第一版实现位于 `orbit/model.py`，使用 MLX `nn.Module` 编写，以便直接使用 Apple Silicon 的统一内存和 Metal kernel。实现采用如下模块映射：

| 理论模块 | 实现类 | 输入 / 输出 |
|---|---|---|
| 感知 `P` | `AttentionBlock`，`causal=False` | `[B,T,W] → [B,T,W]` |
| 对象绑定 `B` | `SlotBinder` | `[B,T,W] → [B,K,W]` |
| 连续更新 `U` | `ContinuumUpdater` | `[B,K,W] × [B,K,W] → [B,K,W]` |
| 因果转移 `T` | `CausalTransition` | `[B,K,W] × [B,W] → [B,K,W]` |
| 语言解码 `D` | `AttentionBlock`，`causal=True` | `[B,T,W] → [B,T,W]` |

其中 `B` 是 batch size，`T` 是 token 数，`K` 是 slots 数，`W` 是 hidden width。模型调用的实际顺序如下：

```python
features = token_embedding(tokens) + position_embedding(positions)
for block in perception:
    features = block(features)
slots = binder(features)
state = state_update(previous_state, slots)
future = state
imagined = []
for _ in range(imagination_steps):
    future = transition(future, action)
    imagined.append(future)
decoded = features + state_to_decoder(mean(state, axis=1))
for block in decoder:
    decoded = block(decoded)
logits = output(final_norm(decoded))
```

`SlotBinder` 使用 `Q/K/V` 投影和 slot 维竞争式 softmax。每轮绑定先计算每个 token 对各 slot 的 affinity，再聚合 value，最后用 gate 在旧 slot 与候选 slot 之间插值。这样 slot 不会被硬编码为“红色物体”或“位置一”，而是从数据中学习稳定的潜在角色。

`ContinuumUpdater` 不直接覆盖旧状态，而是计算 update/reset/candidate 三个门。状态默认只在观察提供证据时改变；在遮挡观察中，训练目标应鼓励它保持旧对象属性。应用层必须显式传入 `previous_state`，新会话传 `None`，避免不同任务之间发生隐式记忆泄漏。

`CausalTransition` 接受 pooled observation/action embedding，并对所有 slots 施加共享动作条件。每一步先用 transition attention 建模槽位间关系，再预测 `delta` 和 `confidence`，最终使用 `state + confidence * tanh(delta)`。因此模型既可以表达动作后果，也可以降低低置信度预测对世界状态的破坏。

当前 tokenizer 尚未绑定到真实语料；原型使用整数 token 张量验证结构。正式训练需要固定 tokenizer vocabulary、special tokens、state reset token 和 action span，并将每个样本组织为：

```text
[RESET] observation_1 [OBS] observation_2 [ACTION] action
[TARGET_STATE] ... [TARGET_FUTURE] ... [ANSWER] ...
```

训练时必须同时保存 `tokens`、`previous_state`、`action`、`target_state`、`target_future` 和 `object_mask`。`object_mask` 用于 `L_cycle`，确保未被动作影响的对象不会被错误地惩罚或更新。

#### English

The first implementation is in `orbit/model.py` and uses MLX `nn.Module`, allowing direct use of Apple Silicon unified memory and Metal kernels. The implementation maps theory to code as follows:

| Theoretical module | Implementation class | Input / output |
|---|---|---|
| Perception `P` | `AttentionBlock`, `causal=False` | `[B,T,W] → [B,T,W]` |
| Object binding `B` | `SlotBinder` | `[B,T,W] → [B,K,W]` |
| Continuum update `U` | `ContinuumUpdater` | `[B,K,W] × [B,K,W] → [B,K,W]` |
| Causal transition `T` | `CausalTransition` | `[B,K,W] × [B,W] → [B,K,W]` |
| Language decoder `D` | `AttentionBlock`, `causal=True` | `[B,T,W] → [B,T,W]` |

Here `B` is batch size, `T` is token count, `K` is slot count, and `W` is hidden width. The actual model call follows this order:

```python
features = token_embedding(tokens) + position_embedding(positions)
for block in perception:
    features = block(features)
slots = binder(features)
state = state_update(previous_state, slots)
future = state
imagined = []
for _ in range(imagination_steps):
    future = transition(future, action)
    imagined.append(future)
decoded = features + state_to_decoder(mean(state, axis=1))
for block in decoder:
    decoded = block(decoded)
logits = output(final_norm(decoded))
```

`SlotBinder` uses Q/K/V projections and competitive softmax over the slot dimension. At each iteration, token-to-slot affinities are computed, values are aggregated, and a gate interpolates between the previous slot and a candidate slot. Slots are therefore not hard-coded as “red object” or “location one”; their latent roles are learned from data.

`ContinuumUpdater` does not overwrite the old state. It computes update, reset, and candidate gates. The intended behavior is to change state when observations provide evidence and to preserve object attributes during occlusion. At the application boundary, `previous_state` must be explicit: pass `None` for a new episode to prevent implicit memory leakage across tasks.

`CausalTransition` consumes a pooled observation/action embedding and applies a shared action condition to all slots. Each step models slot interactions, predicts a `delta` and a `confidence`, and returns `state + confidence * tanh(delta)`. This allows action consequences to be represented while limiting the damage caused by low-confidence predictions.

The tokenizer is not yet connected to a real corpus; the prototype currently uses integer token tensors to validate the architecture. A production training format must fix the vocabulary, special tokens, state-reset token, and action spans. Each sample should contain:

```text
[RESET] observation_1 [OBS] observation_2 [ACTION] action
[TARGET_STATE] ... [TARGET_FUTURE] ... [ANSWER] ...
```

Training batches must retain `tokens`, `previous_state`, `action`, `target_state`, `target_future`, and `object_mask`. `object_mask` supports `L_cycle` by identifying objects that should remain unchanged after an intervention.

### 4.2 训练实施与 Mac 内存策略 / Training Procedure and Mac Memory Strategy

#### 中文

训练采用分阶段策略。第一阶段只使用 Tiny 配置，batch size 由统一内存动态决定，先验证 loss 是否下降以及 slots 是否能重建对象状态。第二阶段同时训练语言损失和状态/动力学损失。第三阶段再扩大 width、sequence length 和 imagination depth。每个阶段都保存 optimizer state、模型配置、tokenizer hash、数据集版本和随机种子。

在 24GB Mac 上，7B 只适合作为 4-bit 推理和结构验证模型。训练时使用 Tiny/Small 的 FP16 或 BF16；不要在 7B 上保留 Adam 全量 optimizer state，因为它会额外占用数倍权重内存。可行的 7B 训练路线是外部多 GPU 预训练、LoRA/QLoRA 适配，或先训练较小 OCA 再进行蒸馏。Apple Silicon 上的 MLX 入口是 `orbit.train`，7B 结构构建入口是 `scripts/build_7b.py`，Metal 前向检查入口是 `scripts/smoke_7b.py`。

4-bit checkpoint 使用 MLX `nn.quantize(model, bits=4, group_size=64)`，随后写入 `weights.safetensors` 和 `model.json`。`model.json` 必须同时记录 logical parameter estimate 与 packed storage element count，不能把量化后的 `uint32` 存储元素数量误写成实际模型参数量。

#### English

Training should proceed in stages. Start with the Tiny configuration, choose batch size according to unified-memory pressure, and verify that language loss decreases and slots can reconstruct object state. The second stage jointly trains language, state, and dynamics losses. Only the third stage increases width, sequence length, and imagination depth. Every stage should save optimizer state, model configuration, tokenizer hash, dataset version, and random seed.

On a 24 GB Mac, 7B is appropriate for 4-bit inference and structural validation, not full Adam pretraining. Train Tiny or Small in FP16/BF16. A full Adam optimizer state for 7B would require several times the weight memory. Practical 7B training routes are external multi-GPU pretraining, LoRA/QLoRA adaptation, or distillation from a smaller trained OCA model. The Apple Silicon MLX entry point is `orbit.train`; 7B construction is `scripts/build_7b.py`; and the Metal forward check is `scripts/smoke_7b.py`.

The 4-bit checkpoint uses `nn.quantize(model, bits=4, group_size=64)` and stores `weights.safetensors` plus `model.json`. `model.json` must record both the logical parameter estimate and the packed storage-element count; quantized `uint32` storage elements must never be mislabeled as the logical model parameter count.

### 4.3 可复现命令 / Reproduction Commands

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# Tiny Metal training smoke test
python -m orbit.train --preset tiny --steps 200

# Generate the 7B architecture manifest
python scripts/build_7b.py

# Allocate FP16 weights and produce a 4-bit structural checkpoint
python scripts/build_7b.py --initialize --quantize \
  --output checkpoints/orbit-7b-random-4bit

# Load the checkpoint and run one forward pass
python scripts/smoke_7b.py checkpoints/orbit-7b-random-4bit

# Run dependency-free tests
python -m unittest discover -s tests -v
```

The commands above reproduce the software artifact. They do not reproduce a trained assistant until a dataset, tokenizer, optimization schedule, and completed training run are supplied.


---

### 4.4 形式化前向算法 / Formal Forward Algorithm

为了让实现与架构描述一一对应，下面给出 OCA 单次前向传播的伪代码。它与 Transformer 论文中对 encoder/decoder stack 的描述类似，但额外显式传递 `state` 和 `imagined_states`。

To make the implementation directly correspond to the architecture, the following pseudocode specifies one OCA forward pass. Like the encoder/decoder description in the Transformer paper, it defines the stack explicitly while adding `state` and `imagined_states` as first-class outputs.

```text
Algorithm 1: Orbit Continuum Architecture Forward
Input: tokens X, previous state S_prev, action A
Parameters: embedding E, perception blocks P, binder B,
            updater U, transition T, decoder D, output head O

H ← E(X) + positional_embedding(X)
for l = 1 ... Lp do
    H ← AttentionBlock(H, causal = false)
end for

Z ← B(H)                                  # object slots [B, K, W]
S ← U(S_prev, Z)                          # present state [B, K, W]
F ← S
for i = 1 ... R do
    F ← T(F, A)                            # imagined future state
    imagined_states[i] ← F
end for

C ← H + project(mean_slots(S))
for l = 1 ... Ld do
    C ← AttentionBlock(C, causal = true)
end for

logits ← O(RMSNorm(C))
return logits, S, Z, imagined_states
```

The algorithm has two distinct time axes. The token axis is processed by attention inside the current observation; the state axis is advanced by the recurrent updater and transition rollout. This separation is the central architectural choice of OCA.

该算法包含两个不同的时间轴。token 轴由当前观察窗口内的 attention 处理；state 轴由循环更新器和转移 rollout 推进。两条时间轴的分离是 OCA 的核心架构选择。

### 4.5 计算复杂度 / Computational Complexity

设当前观察长度为 `T`、隐藏宽度为 `W`、感知层数为 `Lp`、解码层数为 `Ld`、槽位数为 `K`、想象步数为 `R`。标准 self-attention 的主要复杂度为 `O(T²W)`。OCA 额外增加：

Let observation length be `T`, hidden width `W`, perception depth `Lp`, decoder depth `Ld`, slot count `K`, and imagination depth `R`. Standard self-attention contributes the dominant `O(T²W)` term. OCA adds:

- slot binding: `O(IKTW)` for `I` binding iterations;
- continuum update: `O(KW²)`;
- transition rollout: `O(RK²W + RKW²)` when transition attention is applied over slots;
- state-to-decoder conditioning: `O(TW²)` for the projection and `O(TW)` for the broadcast addition.

当 `K << T` 时，槽位和想象分支不会替代 token attention，而是以较小的结构化状态成本增加时间和因果能力。实际部署时可以减少 `R`、减少 slots、缓存静态 action embedding，或仅在需要规划时启用 imagination 分支。

When `K << T`, the slot and imagination branches do not replace token attention; they add temporal and causal capacity with a smaller structured-state cost. At inference time, `R` and `K` can be reduced, action embeddings can be cached, and the imagination branch can be enabled only for planning-sensitive requests.

### 4.6 与标准 Transformer 的结构差异 / Structural Difference from a Standard Transformer

| 组件 Component | Decoder-only Transformer | OCA |
|---|---|---|
| 当前观察 Current observation | token sequence | token sequence + perception stack |
| 长期状态 Long-term state | implicit in context/KV cache | explicit continuum state |
| 对象表示 Object representation | distributed token features | competitive object slots |
| 动作后果 Action consequences | learned implicitly in logits | action-conditioned transition |
| 内部规划 Internal planning | optional prompting/tool use | multi-step latent imagination |
| 状态生命周期 State lifecycle | context dependent | explicit initialize/update/reset |
| 诊断方式 Diagnostics | token/logit inspection | token + slots + state + futures |

这个表描述的是架构接口，不是效果结论。只有通过参数匹配的基线和消融实验，才能判断这些额外组件是否带来真实收益。

This table describes architectural interfaces, not performance claims. Only parameter-matched baselines and ablations can determine whether the additional components produce real gains.

---

## 5. 实验设计 / Experimental Design

为了判断 OCA 是否真的贡献了世界状态能力，必须使用参数量匹配的 decoder-only Transformer 作为基线，并进行模块消融。建议的实验集合如下：

To determine whether OCA contributes world-state capability, use a parameter-matched decoder-only Transformer baseline and perform component ablations. The proposed experiment suite is:

| 任务 Task | 测量内容 Measurement |
|---|---|
| 隐藏物体追踪 Hidden-object tracking | 遮挡后对象身份和属性是否保持 Object identity and attributes under occlusion |
| 干预预测 Intervention prediction | 动作后的目标状态 Action-conditioned target state |
| 非相关对象保护 Unaffected-object preservation | 未被动作影响的对象是否保持 Unchanged state of unrelated objects |
| 长序列记忆 Long-horizon memory | 状态在长观察序列中的稳定性 State stability over long sequences |
| 多步规划 Multi-step planning | imagination depth 对成功率的影响 Effect of rollout depth on planning |
| 不确定性校准 Uncertainty calibration | 遮挡和冲突观察时的置信度 Confidence under ambiguity |

必要的消融包括：移除 slots、移除 persistent state、移除 transition model、将 imagination steps 设为 1，以及把 state 直接拼接到 token embedding。只有当增益在多个随机种子、未见过的对象组合和未见过的动作组合上保持，才能认为架构假设得到支持。

Required ablations include removing slots, removing persistent state, removing the transition model, setting imagination depth to one, and concatenating state directly into token embeddings. An architectural hypothesis is supported only if gains persist across random seeds and unseen object/action combinations.

---

## 6. 与相关工作的关系 / Relation to Prior Work

### 中文

OCA 不是声称独立发明了所有组成部件。Transformer 提供了注意力和序列建模基础；Slot Attention 展示了无监督对象槽位绑定的可行性；World Models、Dreamer 等工作研究了潜在动力学和想象；循环网络长期以来也维护隐藏状态。OCA 的研究问题是：能否将这些思想组织成一个同时面向语言输出、对象持续性和动作条件想象的统一结构，并通过明确状态生命周期改善可诊断性。

任何专利或原创性结论都需要系统的既有技术检索、权利要求对照和实验数据。本文只是一份工程研究预印本，不构成专利有效性判断。

### English

OCA does not claim independent invention of every component. Transformers provide the attention and sequence-modeling foundation; Slot Attention demonstrates unsupervised object-slot binding; World Models and Dreamer study latent dynamics and imagination; recurrent models have maintained hidden state for decades. The research question is whether these ideas can be organized into one system serving language output, object permanence, and action-conditioned imagination, while improving diagnosability through an explicit state lifecycle.

Any patent or originality claim requires systematic prior-art search, claim comparison, and experimental evidence. This paper is an engineering research preprint and is not an opinion on patentability.

---

## 7. 局限与诚实边界 / Limitations and Honest Boundaries

### 中文

第一，当前 7B 权重是随机初始化的，不能进行有意义的自然语言对话。第二，合成世界数据不能代表真实视觉、物理和社会环境。第三，slots 不自动保证语义对象对应关系，可能产生 slot permutation、对象混合或状态坍缩。第四，当前实现尚未完成 tokenizer、真实多模态数据管线、状态监督损失和长时序训练。第五，Mac 可以承载量化推理和小规模实验，但无法在一天内从零训练出达到成熟开源模型水平的 7B 基础模型。

### English

First, the current 7B weights are randomly initialized and cannot conduct meaningful natural-language conversation. Second, synthetic worlds are not substitutes for real visual, physical, or social environments. Third, slots do not automatically guarantee semantic object correspondence and may suffer from slot permutation, object entanglement, or state collapse. Fourth, tokenizer integration, real multimodal data pipelines, state-supervision losses, and long-horizon training remain incomplete. Fifth, a Mac can support quantized inference and small experiments, but cannot pretrain a competitive 7B foundation model from scratch in one day.

---

## 8. 结论与路线图 / Conclusion and Roadmap

### 中文

OCA 提出一种清晰的工程分解：感知负责读取当前观察，slots 负责绑定对象，continuum state 负责持续世界假设，transition model 负责动作后果，language decoder 负责表达。它的价值不应由架构名称或参数规模决定，而应由可复现实验决定。

下一步路线是：完成可逆 tokenizer 和训练语料格式；加入状态、动力学和不确定性损失；训练 Tiny OCA 与参数匹配基线；在合成世界通过验证门后扩展到 300M/1B；最后再考虑 7B 的大规模训练或蒸馏。只有这些实验显示持续稳定的收益，Orbit 才能谨慎地声称自己正在接近“理解世界”。

### English

OCA provides a clear engineering decomposition: perception reads the current observation, slots bind objects, the continuum state maintains a world hypothesis, the transition model predicts consequences, and the language decoder communicates. Its value must be determined by reproducible experiments, not by its name or parameter count.

The next steps are to complete a reversible tokenizer and training format; add state, dynamics, and uncertainty losses; train Tiny OCA against a parameter-matched baseline; scale to 300M/1B after passing synthetic-world gates; and only then consider large-scale 7B training or distillation. If these experiments show stable gains, Orbit can cautiously claim progress toward world understanding.

---

## 参考文献 / References

1. Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
2. Locatello, F. et al. (2020). *Object-Centric Learning with Slot Attention.* NeurIPS.
3. Ha, D. and Schmidhuber, J. (2018). *World Models.* NeurIPS Workshop.
4. Hafner, D. et al. (2020). *Dream to Control: Learning Behaviors by Latent Imagination.* ICLR.
5. Hafner, D. et al. (2023). *Mastering Diverse Domains through World Models.* arXiv preprint.
6. Chung, J. et al. (2014). *Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.* NeurIPS Workshop.

---

## 版权 / Copyright

© 2026 YUNSH and Liu Jiacheng (刘家成), a 14-year-old middle-school student from China. All rights reserved unless a separate license is provided with a specific software file.

© 2026 YUNSH 与来自中国的 14 岁初中生刘家成。除非具体软件文件另有许可，本论文与相关研究材料保留全部权利。
