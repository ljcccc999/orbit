# Orbit Continuum Architecture（OCA）
## 持续世界状态与因果想象架构

**研究预印本 · Version 0.1 · 2026-08-03**  
**项目：** Orbit  
**作者：** YUNSH · 刘家成（中国，14 岁初中生）

> 本文描述一个已经实现了可运行原型的研究架构。“7B”表示模型配置的逻辑参数规模。当前 7B checkpoint 是随机初始化后量化的结构验证权重，尚未完成语言预训练，因此不能被描述为已经具备成熟对话能力的基础模型。

## 摘要

当前主流语言模型通常把世界信息压缩到上下文序列和注意力缓存中，并以预测下一个 token 作为主要训练目标。这种方案能够产生流畅语言，但并不要求模型显式维护对象、时间、隐藏状态、动作后果或观察不确定性。

本文提出 Orbit Continuum Architecture（OCA），一种面向持续世界状态的语言模型架构。OCA 将模型拆分为感知、对象槽位绑定、连续状态更新、动作条件状态转移、未来状态想象和语言解码六个部分。其核心状态变量独立于 token KV cache，用门控更新机制维护跨观察时刻的世界状态；竞争式对象槽位为对象身份和属性提供可检查的潜在瓶颈；因果转移模块在给定动作后滚动预测多个未来状态。

我们已经在 Apple Silicon Mac mini 上完成 OCA 7B 配置的 FP16 权重初始化、4-bit 量化、Metal 前向推理和连续状态传递测试。本文不声称 OCA 已经解决通用世界理解，而是提出一套可证伪的架构假设和实验路线：如果显式的持久状态、对象绑定和因果想象确实有价值，则在参数规模匹配的隐藏物体追踪、干预预测和多步规划任务上，应优于只使用语言预测的基线模型。

**关键词：** Orbit；OCA；世界模型；持续状态；对象槽位；因果想象；Apple Silicon；MLX

---

## 1. 引言

语言是世界的一个投影，而不是世界本身。一个人说“杯子被挡住了”，隐含表达了至少三类信息：杯子仍然存在；它的可见性发生了变化；如果移开遮挡物，杯子可能重新出现。只优化下一个 token 的模型可以通过统计相关性回答类似问题，但它没有被强制要求保存一个稳定的对象状态，也没有被强制要求区分“物体消失”和“物体暂时不可见”。

OCA 的出发点不是否定 Transformer，而是重新划分其职责。注意力适合处理当前观察中的关系，语言解码器适合表达和交互；然而，持续对象、时间变化和动作后果需要更明确的内部变量。OCA 因此保留 Transformer block，同时加入一个可传递、可重置、可观察的 continuum state（连续世界状态）。

本文将“理解世界”定义为一组可测试能力，而不是一句宣传语：

1. 在遮挡下保持对象存在性；
2. 理解事件顺序；
3. 预测动作引起的状态变化；
4. 保留未受干预对象的状态；
5. 在新观察与旧记忆冲突时更新信念；
6. 在不确定信息下表达校准后的置信度。

只有在这些测试中取得可复现结果，才可以进一步讨论更强的世界模型能力。

## 2. 架构概述

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

### 2.1 感知模块

给定 token 序列 `X_t`，感知模块加入 token embedding 和位置 embedding，并通过非因果 self-attention 聚合整个观察窗口：

\[
H_t = P(X_t) = P_L(\cdots P_2(P_1(E(X_t) + R))).
\]

非因果感知并不等于模型可以偷看未来世界；它只表示当前输入窗口内部的 token 可以互相读取。时间上的未来由独立的状态转移模块负责。

### 2.2 对象槽位绑定

OCA 维护固定数量的 latent slots。每个 slot 不是预先指定的物体类别，而是一个可学习的对象假设。模型通过多轮竞争式注意力让 observation features 竞争 slots：

\[
A^{(i)} = \operatorname{softmax}_{slots}
\left(\frac{Q(S^{(i)})K(H_t)^T}{\sqrt d}\right),
\]

\[
S^{(i+1)} = G^{(i)} \odot S^{(i)} + (1-G^{(i)}) \odot
\tanh\left(C([S^{(i)}; A^{(i)}V(H_t)])\right).
\]

槽位数量是一个明确的容量假设：太少会导致对象混叠，太多会带来冗余和训练不稳定。槽位可以被读取和单独监督，因此比完全分散在 token 表示中的对象信息更容易进行诊断。

### 2.3 连续世界状态

设上一时刻状态为 `S_{t-1}`，当前观察绑定出的槽位为 `B_t`。连续状态更新器使用更新门、重置门和候选状态：

\[
z_t = \sigma(W_z[S_{t-1};B_t]),
\quad r_t = \sigma(W_r[S_{t-1};B_t]),
\]

\[
\tilde S_t = \tanh(W_c[r_t\odot S_{t-1};B_t]),
\quad S_t = (1-z_t)\odot S_{t-1}+z_t\odot\tilde S_t.
\]

该状态和 KV cache 有不同生命周期。KV cache 服务于一次上下文中的高效注意力；continuum state 表达跨观察时刻的世界假设。应用可以在会话开始时初始化它，在独立任务之间显式重置它，也可以将它序列化保存为工作区状态。

### 2.4 因果想象

给定当前状态 `S_t` 和动作表示 `A_t`，转移模块生成未来状态：

\[
\hat S_{t+1} = T(S_t,A_t),
\qquad
\hat S_{t+k} = T(\hat S_{t+k-1},A_t).
\]

每次转移都包含一个 confidence gate，用于限制不确定的预测对状态的修改幅度。当前 7B 配置默认进行六步想象。多步 rollout 的意义不是生成内部动画，而是让模型在输出或执行动作前比较可能的后果。

### 2.5 语言解码

语言解码器接收感知特征，并注入当前 continuum state 的 pooled context：

\[
Y_t = D(H_t + W_s\operatorname{pool}(S_t)),
\qquad
\text{logits}_t = W_o\operatorname{RMSNorm}(Y_t).
\]

这使语言输出能够读取世界状态，但不要求语言序列本身承担全部状态存储责任。

## 3. 具体实现方法

第一版实现位于 `orbit/model.py`，使用 Apple Silicon 上的 MLX `nn.Module` 编写。

| 理论模块 | 实现类 | 输入 / 输出 |
|---|---|---|
| 感知 `P` | `AttentionBlock`，`causal=False` | `[B,T,W] → [B,T,W]` |
| 对象绑定 `B` | `SlotBinder` | `[B,T,W] → [B,K,W]` |
| 连续更新 `U` | `ContinuumUpdater` | `[B,K,W] × [B,K,W] → [B,K,W]` |
| 因果转移 `T` | `CausalTransition` | `[B,K,W] × [B,W] → [B,K,W]` |
| 语言解码 `D` | `AttentionBlock`，`causal=True` | `[B,T,W] → [B,T,W]` |

其中 `B` 是 batch size，`T` 是 token 数，`K` 是 slots 数，`W` 是 hidden width。

单次前向传播顺序如下：

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

`SlotBinder` 使用 `Q/K/V` 投影和 slot 维竞争式 softmax。每轮绑定先计算每个 token 对各 slot 的 affinity，再聚合 value，最后用 gate 在旧 slot 与候选 slot 之间插值。

`ContinuumUpdater` 不直接覆盖旧状态，而是计算 update/reset/candidate 三个门。新会话传入 `None` 初始化状态；连续观察则显式传入上一步 `state`，避免不同任务之间发生隐式记忆泄漏。

`CausalTransition` 接受 pooled observation/action embedding，并对所有 slots 施加共享动作条件。每一步先用 transition attention 建模槽位间关系，再预测 `delta` 和 `confidence`，最终使用：

```text
next_state = state + confidence * tanh(delta)
```

当前 tokenizer 尚未绑定到真实语料；原型使用整数 token 张量验证结构。正式训练需要固定 tokenizer vocabulary、special tokens、state reset token 和 action span，并将每个样本组织为：

```text
[RESET] observation_1 [OBS] observation_2 [ACTION] action
[TARGET_STATE] ... [TARGET_FUTURE] ... [ANSWER] ...
```

训练批次需要保存 `tokens`、`previous_state`、`action`、`target_state`、`target_future` 和 `object_mask`。

## 4. 训练目标

OCA 的目标函数设计为多任务形式：

\[
\mathcal L = \mathcal L_{lang}
+ \lambda_s\mathcal L_{state}
+ \lambda_d\mathcal L_{dyn}
+ \lambda_c\mathcal L_{cycle}
+ \lambda_u\mathcal L_{uncertainty}.
\]

- `L_lang`：标准 next-token 与指令响应损失；
- `L_state`：从 slots 重建对象属性、位置和可见性；
- `L_dyn`：在给定动作后预测真实下一状态；
- `L_cycle`：动作只应改变相关对象，避免无关状态漂移；
- `L_uncertainty`：在遮挡或冲突观察下校准置信度。

当前仓库已经具备 `L_lang` 的 MLX 冒烟训练入口和合成世界生成器；其余状态与动力学损失是下一阶段训练重点。

## 5. 形式化前向算法

```text
算法 1：Orbit Continuum Architecture Forward
输入：tokens X，previous state S_prev，action A
参数：embedding E，perception P，binder B，updater U，
      transition T，decoder D，output head O

H ← E(X) + positional_embedding(X)
for l = 1 ... Lp do
    H ← AttentionBlock(H, causal = false)
end for

Z ← B(H)                                  # 对象槽位 [B, K, W]
S ← U(S_prev, Z)                          # 当前状态 [B, K, W]
F ← S
for i = 1 ... R do
    F ← T(F, A)                            # 想象未来状态
    imagined_states[i] ← F
end for

C ← H + project(mean_slots(S))
for l = 1 ... Ld do
    C ← AttentionBlock(C, causal = true)
end for

logits ← O(RMSNorm(C))
返回 logits, S, Z, imagined_states
```

OCA 包含两个不同的时间轴。token 轴由当前观察窗口内的 attention 处理；state 轴由循环更新器和转移 rollout 推进。两条时间轴的分离是 OCA 的核心架构选择。

## 6. 计算复杂度

设当前观察长度为 `T`、隐藏宽度为 `W`、感知层数为 `Lp`、解码层数为 `Ld`、槽位数为 `K`、想象步数为 `R`。标准 self-attention 的主要复杂度为 `O(T²W)`。OCA 额外增加：

- slot binding：`O(IKTW)`，其中 `I` 是绑定迭代次数；
- continuum update：`O(KW²)`；
- transition rollout：`O(RK²W + RKW²)`；
- state-to-decoder conditioning：投影为 `O(TW²)`，广播加法为 `O(TW)`。

当 `K << T` 时，槽位和想象分支不会替代 token attention，而是以较小的结构化状态成本增加时间和因果能力。部署时可以减少 `R`、减少 slots，或仅在需要规划时启用 imagination 分支。

## 7. 与标准 Transformer 的结构差异

| 组件 | Decoder-only Transformer | OCA |
|---|---|---|
| 当前观察 | token sequence | token sequence + perception stack |
| 长期状态 | 隐式存在于 context/KV cache | 显式 continuum state |
| 对象表示 | 分散在 token features 中 | competitive object slots |
| 动作后果 | 隐式学习到 logits 中 | action-conditioned transition |
| 内部规划 | 依赖提示词或外部工具 | multi-step latent imagination |
| 状态生命周期 | 依赖 context | 显式 initialize/update/reset |
| 诊断方式 | token/logit 检查 | token + slots + state + futures |

这个表描述的是架构接口，不是效果结论。只有通过参数匹配的基线和消融实验，才能判断这些额外组件是否带来真实收益。

## 8. 7B 配置与 Mac mini 实现结果

7B 配置使用：

- hidden width：4096；
- perception blocks：10；
- decoder blocks：12；
- transition blocks：5；
- attention heads：32；
- object slots：32；
- imagination steps：6；
- vocabulary size：65,536；
- 逻辑参数估算：7,306,608,640。

我们已经在 Apple Silicon Mac mini 上完成：

1. OCA 7B FP16 权重初始化；
2. MLX 4-bit 量化；
3. 约 3.9GB 的本地量化 checkpoint；
4. Metal 前向传播；
5. continuum state 跨 observation 传递。

验证输出为：

```text
logits: (1, 8, 65536)
continuum state: (1, 32, 4096)
imagined states: (1, 6, 32, 4096)
```

这些结果证明软件结构、权重格式和 Metal 推理链路已经能够运行，但不证明模型已经获得语言知识或通用世界理解能力。当前 7B checkpoint 是随机初始化后量化的结构验证模型。

## 9. 训练实施与内存策略

训练采用分阶段策略。第一阶段只使用 Tiny 配置，先验证 loss 是否下降以及 slots 是否能重建对象状态。第二阶段同时训练语言损失和状态/动力学损失。第三阶段再扩大 width、sequence length 和 imagination depth。每个阶段都保存 optimizer state、模型配置、tokenizer hash、数据集版本和随机种子。

在 24GB Mac mini 上，7B 适合作为 4-bit 推理和结构验证模型，不适合使用 Adam 从零完成大规模预训练。可行的 7B 训练路线是外部多 GPU 预训练、LoRA/QLoRA 适配，或先训练较小 OCA 再进行蒸馏。

4-bit checkpoint 使用：

```python
nn.quantize(model, bits=4, group_size=64)
mx.save_safetensors("weights.safetensors", weights)
```

同时保存 `model.json`，记录 logical parameter estimate 与 packed storage element count，不能把量化后的 `uint32` 存储元素数量误写成实际模型参数量。

## 10. 实验设计

必须使用参数量匹配的 decoder-only Transformer 作为基线，并进行模块消融。建议实验包括：

| 任务 | 测量内容 |
|---|---|
| 隐藏物体追踪 | 遮挡后对象身份和属性是否保持 |
| 干预预测 | 动作后的目标状态 |
| 非相关对象保护 | 未被动作影响的对象是否保持 |
| 长序列记忆 | 状态在长观察序列中的稳定性 |
| 多步规划 | imagination depth 对成功率的影响 |
| 不确定性校准 | 遮挡和冲突观察时的置信度 |

必要的消融包括：移除 slots、移除 persistent state、移除 transition model、将 imagination steps 设为 1，以及把 state 直接拼接到 token embedding。只有当增益在多个随机种子、未见过的对象组合和未见过的动作组合上保持，才能认为架构假设得到支持。

## 11. 与相关工作的关系

OCA 不声称独立发明了所有组成部件。Transformer 提供了注意力和序列建模基础；Slot Attention 展示了无监督对象槽位绑定的可行性；World Models、Dreamer 等工作研究了潜在动力学和想象；循环网络长期以来也维护隐藏状态。

OCA 的研究问题是：能否将这些思想组织成一个同时面向语言输出、对象持续性和动作条件想象的统一结构，并通过明确状态生命周期改善可诊断性。任何专利或原创性结论都需要系统的既有技术检索、权利要求对照和实验数据。本文只是一份工程研究预印本，不构成专利有效性判断。

## 12. 局限与诚实边界

第一，当前 7B 权重是随机初始化的，不能进行有意义的自然语言对话。第二，合成世界数据不能代表真实视觉、物理和社会环境。第三，slots 不自动保证语义对象对应关系，可能产生 slot permutation、对象混合或状态坍缩。第四，当前实现尚未完成 tokenizer、真实多模态数据管线、状态监督损失和长时序训练。第五，Mac 可以承载量化推理和小规模实验，但无法在一天内从零训练出达到成熟开源模型水平的 7B 基础模型。

## 13. 结论与路线图

OCA 提出一种清晰的工程分解：感知负责读取当前观察，slots 负责绑定对象，continuum state 负责持续世界假设，transition model 负责动作后果，language decoder 负责表达。它的价值不应由架构名称或参数规模决定，而应由可复现实验决定。

下一步路线是：完成 tokenizer 和训练语料格式；加入状态、动力学和不确定性损失；训练 Tiny OCA 与参数匹配基线；在合成世界通过验证门后扩展到 300M/1B；最后再考虑 7B 的大规模训练或蒸馏。只有这些实验显示持续稳定的收益，Orbit 才能谨慎地声称自己正在接近“理解世界”。

## 参考文献

1. Vaswani, A. 等（2017）。《Attention Is All You Need》。NeurIPS。
2. Locatello, F. 等（2020）。《Object-Centric Learning with Slot Attention》。NeurIPS。
3. Ha, D. 与 Schmidhuber, J.（2018）。《World Models》。NeurIPS Workshop。
4. Hafner, D. 等（2020）。《Dream to Control: Learning Behaviors by Latent Imagination》。ICLR。
5. Hafner, D. 等（2023）。《Mastering Diverse Domains through World Models》。arXiv。
6. Chung, J. 等（2014）。《Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling》。NeurIPS Workshop。

---

## 版权

© 2026 YUNSH 与来自中国的 14 岁初中生刘家成。除非具体软件文件另有许可，本论文与相关研究材料保留全部权利。

