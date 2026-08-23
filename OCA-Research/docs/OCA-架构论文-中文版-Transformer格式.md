# Orbit Continuum Architecture：持续世界状态与因果想象模型

**YUNSH · 刘家成（中国，14 岁初中生）**  
**研究预印本，Version 0.2 · 2026-08-03**

## 摘要

大多数语言模型通过 token 序列和注意力缓存间接表示世界，并以预测下一个 token 作为主要训练目标。这一目标足以产生流畅的语言，但不要求模型在遮挡下保持对象身份、不要求模型在多个观察之间维护状态，也不要求模型预测干预的后果。本文提出 Orbit Continuum Architecture（OCA），一种在 Transformer 语言系统中加入显式持续世界状态的模型架构。

OCA 由感知堆栈、竞争式对象槽位绑定、门控连续状态更新器、动作条件状态转移模型、多步想象模块和自回归语言头组成。该架构将 token 时间轴与状态时间轴分开：token 轴编码当前观察，state 轴维护并推进一个关于世界的结构化潜在假设。这样，状态生命周期、对象表示、反事实未来和不确定性成为显式接口，而不是语言解码器的隐式行为。

本文定义 OCA 的架构、计算过程、训练目标和与参数量匹配 Transformer 基线的验证方案。当前实现已经在 Apple Silicon Mac mini 上完成逻辑 7B 配置的结构初始化和 Metal 推理验证；这只是实现状态，不是语言质量实验结果。OCA 仍然是一个研究假设，其价值必须由对象持续性、干预预测和多步规划实验建立。

## 1. 引言

循环网络和注意力序列模型推动了语言建模与序列转换的发展。然而，语言模型可以在没有显式记录“哪些实体存在、哪些属性改变、哪些事件造成改变”的情况下生成合理文本。在当前观察不完整时，这种差异尤其重要：被遮挡的物体通常仍然属于世界状态，而被真正移除的物体不应继续存在于状态中。

本文研究的问题是：如果为语言模型增加一个生命周期不同于 token 上下文的显式状态变量，模型是否会获得更好的世界状态和规划能力？OCA 并不否定 Transformer，而是为不同计算路径分配不同职责：

- 感知 attention 表示当前观察中的关系；
- 对象 slots 提供紧凑、可检查的绑定空间；
- continuum state 在观察之间传递信息；
- transition model 预测动作条件下的未来；
- language head 表达选定的状态和预测。

本文将“世界理解”定义为可测试能力：保持对象持续性、表示事件顺序、预测干预、保持未受影响对象、在矛盾证据出现后更新信念，以及在不完整信息下校准置信度。我们不从流畅文本单独推断这些能力。

## 2. 背景

Transformer 通过堆叠 self-attention 与逐位置前馈层构成序列转换架构，并移除了主计算中的位置对齐循环。其 encoder 将输入序列映射为连续表示，autoregressive decoder 生成输出序列。OCA 保留这一有效分解，但增加一条持久状态路径。

在标准 decoder-only 语言模型中，早期事件的信息通常保存在隐藏激活和依赖上下文的 KV cache 中。这些表示非常强大，但模型没有被强制要求使用对应于对象、干预或世界状态的变量。OCA 因此把语言序列视为观察和通信通道，而不是世界唯一的记忆。

OCA 将 self-attention、门控更新、对象中心 slots 和潜在动力学组合成统一接口。本文不声称这些组成部分分别是全新的；研究问题是将它们分离并联合训练后，是否会改善可测量的状态和规划行为。

## 3. 模型架构

给定观察 token 序列 `X_t`、可选的前一状态 `S_{t-1}` 和动作表示 `A_t`，OCA 返回语言 logits、当前状态、对象槽位和想象未来：

\[
(Y_t,S_t,Z_t,\hat S_{t+1:t+R}) = f_\theta(X_t,S_{t-1},A_t).
\]

整体计算过程为：

```text
X_t → 感知 P → 特征 H_t → 槽位绑定 B → 槽位 Z_t
                                      ↓
                    前一状态 S_{t-1} → 更新 U → S_t
                                                        ↓
                                       转移 T(S_t, A_t)
                                                        ↓
                                      想象状态 Ŝ_{t+1:t+R}
                                                        ↓
                H_t + 状态上下文 → 解码 D → logits Y_t
```

### 3.1 感知堆栈

模型先加入 token embedding 和位置 embedding，再通过非因果 attention block：

\[
H_t=P(X_t)=P_{L_p}(\cdots P_2(P_1(E(X_t)+R))).
\]

感知模块可以使用标准 multi-head attention 与门控前馈层。由于输入窗口表示当前观察，非因果 attention 可以读取窗口内部的 token；它不会读取 transition model 生成的未来状态。

### 3.2 竞争式对象槽位绑定

设 slots 数量为 `K`。slots 由可学习种子初始化，并通过多轮迭代绑定观察特征。在第 `i` 轮：

\[
Q_i=W_qS_i,\quad K_t=W_kH_t,\quad V_t=W_vH_t,
\]

\[
A_i=\operatorname{softmax}_{slots}
\left(\frac{Q_iK_t^T}{\sqrt W}\right),
\qquad
M_i=A_iV_t.
\]

槽位更新为：

\[
G_i=\sigma(W_g[S_i;M_i]),
\]

\[
S_{i+1}=G_i\odot S_i+(1-G_i)\odot
\tanh(W_c[S_i;M_i]).
\]

槽位瓶颈鼓励模型表示少量持续实体，而不是把每个对象属性分散到互不相关的 token 位置中。槽位身份并不预设为固定语义，因此训练时必须处理槽位排列和对应关系。

### 3.3 连续状态更新

绑定后的槽位 `Z_t` 与上一状态通过门控循环更新合并：

\[
z_t=\sigma(W_z[S_{t-1};Z_t]),
\qquad r_t=\sigma(W_r[S_{t-1};Z_t]),
\]

\[
\tilde S_t=\tanh(W_c[r_t\odot S_{t-1};Z_t]),
\]

\[
S_t=(1-z_t)\odot S_{t-1}+z_t\odot\tilde S_t.
\]

该状态在应用层显式存在。新 episode 传入空状态或学习到的初始状态，连续 episode 传入 `S_{t-1}`，独立 episode 必须重置状态。它与 KV cache 的生命周期不同：KV cache 是上下文内高效 attention 的实现结构，而 continuum state 是跨观察的世界假设。

### 3.4 动作条件状态转移

转移模型预测一个动作如何改变当前状态。动作 embedding 被投影到状态宽度，并与槽位表示合并：

\[
U_0=S_t+W_aA_t.
\]

转移 block 建模槽位关系，并输出状态变化量与置信度：

\[
\Delta_i=\tanh(W_\Delta F_i),
\qquad c_i=\sigma(W_cF_i),
\]

\[
\hat S_{t+1}=S_t+c_i\odot\Delta_i.
\]

置信度门限制不确定动力学预测的影响。重复应用转移得到：

\[
\hat S_{t+j}=T(\hat S_{t+j-1},A_t),\quad j=1,\ldots,R.
\]

### 3.5 语言头

语言 decoder 接收感知特征，并注入当前状态的 pooled context：

\[
C_t=H_t+W_s\operatorname{pool}(S_t),
\]

\[
Y_t=D(C_t),\qquad
P(y_i|y_{<i},X_t,S_t)=\operatorname{softmax}(W_o\operatorname{RMSNorm}(Y_{t,i})).
\]

语言头保持 autoregressive 和 causal。状态路径不是为了替代 token 语言建模，而是提供一个语言可以读取和描述的持久世界假设。

### 3.6 形式化前向算法

```text
算法 1：OCA 前向传播
输入：tokens X，previous state S_prev，action A

H ← E(X) + positional_embedding(X)
for l = 1 ... Lp do
    H ← PerceptionBlock_l(H)
end for

Z ← SlotBinder(H)
S ← ContinuumUpdate(S_prev, Z)
F ← S
for j = 1 ... R do
    F ← Transition(F, A)
    futures[j] ← F
end for

C ← H + StateProjection(mean_slots(S))
for l = 1 ... Ld do
    C ← CausalDecoderBlock_l(C)
end for
logits ← OutputProjection(RMSNorm(C))
返回 logits, S, Z, futures
```

## 4. 为什么需要 Continuum 架构

### 4.1 持久状态与部分观察

如果一个实体没有出现在当前观察中，模型应该区分“没有观察到”和“不存在”。持久状态为这种区别提供了位置。观察缺失时，更新器可以保持槽位；证据矛盾时，它可以修改槽位。

### 4.2 对象绑定与诊断

token 表示可能包含对象信息，却无法说明信息存储在哪里。slots 提供了一个可以解码、监督和比较的接口。这不能自动保证可解释性，但能让架构假设可以被实验检查。

### 4.3 反事实 rollout

语言模型可以不显式模拟动作就生成一个答案。OCA 提供 transition path，使训练能够惩罚错误的动作后果，并保护不相关对象的状态。想象深度 `R` 可以配置，也可以在推理时降低。

### 4.4 状态生命周期

显式的 initialize/update/reset 接口可以避免 episode 之间意外携带记忆，也允许工作区状态被独立保存、检查或丢弃。

## 5. 训练

### 5.1 数据与批处理

初始课程应使用包含对象、位置、可见性、时间和显式动作的合成世界。一个样本包含观察、动作、真实当前状态、真实未来状态和答案目标：

```text
[RESET] observation_1 [OBS] observation_2 [ACTION] action
[TARGET_STATE] state_t [TARGET_FUTURE] state_t+1 ... [ANSWER] answer
```

每个 batch 保存 token IDs、`previous_state`、action embedding、目标槽位或属性、目标未来状态，以及标记未受影响对象的 intervention mask。

### 5.2 目标函数

\[
\mathcal L=\mathcal L_{lang}
+\lambda_s\mathcal L_{state}
+\lambda_d\mathcal L_{dyn}
+\lambda_c\mathcal L_{cycle}
+\lambda_u\mathcal L_{uncertainty}.
\]

`L_lang` 是 next-token loss；`L_state` 重建对象属性和可见性；`L_dyn` 预测动作后的状态；`L_cycle` 惩罚 intervention mask 之外对象的状态变化；`L_uncertainty` 校准遮挡和冲突观察下的置信度。

### 5.3 扩展与实现状态

当前实现使用 MLX。逻辑 7B 配置已经完成初始化和量化，并在 Apple Silicon Mac mini 上验证 Metal 前向路径。这是部署检查，不是训练结果。完整 7B 预训练需要更大的算力计划；本机适合进行架构实验、量化推理、小规模训练和蒸馏。

## 6. 实验

我们提出参数量匹配的 decoder-only Transformer 基线，并使用以下任务：

| 任务 | 测量内容 |
|---|---|
| 隐藏物体追踪 | 遮挡后的对象身份和属性 |
| 干预预测 | 显式动作后的目标状态 |
| 非相关对象保护 | intervention mask 之外的状态变化 |
| 长时记忆 | 长观察序列中的状态稳定性 |
| 多步规划 | imagination depth 对成功率的影响 |
| 不确定性校准 | 不完整证据下的置信度 |

必要的消融包括移除 slots、移除 persistent state、移除 transition model，以及只保留一步 imagination。有效结果必须泛化到未见过的对象与动作组合，并在多个随机种子下稳定。

当前阶段尚未完成这些比较实验。已验证结果仅限软件执行：7B 配置输出 logits 形状 `(1, 8, 65536)`、state 形状 `(1, 32, 4096)`，以及六步 imagined states 形状 `(1, 6, 32, 4096)`。

## 7. 结论

OCA 为 Transformer 语言系统增加了显式持续状态、对象槽位绑定和动作条件想象。它的中心主张是可测试的：将观察 token 与持久世界状态路径分开，可能改善对象持续性、干预预测和多步规划。

当前实现建立了架构和可运行的逻辑 7B 配置，但没有证明语言质量或通用世界理解。下一步应在 Tiny 规模上加入状态和动力学监督，完成参数量匹配的消融实验，再考虑更大规模训练。

## 参考文献

1. Vaswani, A. 等（2017）。*Attention Is All You Need*。NeurIPS。
2. Locatello, F. 等（2020）。*Object-Centric Learning with Slot Attention*。NeurIPS。
3. Ha, D. 与 Schmidhuber, J.（2018）。*World Models*。NeurIPS Workshop。
4. Hafner, D. 等（2020）。*Dream to Control: Learning Behaviors by Latent Imagination*。ICLR。
5. Chung, J. 等（2014）。*Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling*。NeurIPS Workshop。

---

## 版权

© 2026 YUNSH 与来自中国的 14 岁初中生刘家成。除非具体软件文件另有许可，本论文与相关研究材料保留全部权利。

