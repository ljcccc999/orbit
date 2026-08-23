# Orbit Continuum Architecture: A Persistent World-State and Causal-Imagination Model

**YUNSH · Liu Jiacheng (China, age 14, middle-school student)**  
**Research preprint, Version 0.2 · 2026-08-03**

## Abstract

Most language models represent the world indirectly through token sequences and attention caches, and are trained primarily with next-token prediction. This objective is sufficient for fluent continuation, but it does not require a model to preserve object identity under occlusion, maintain a state across observations, or predict the consequences of an intervention. We introduce the Orbit Continuum Architecture (OCA), a model architecture that adds an explicit persistent world state to a Transformer-based language system.

OCA consists of a perception stack, competitive object-slot binding, a gated continuum-state updater, an action-conditioned transition model, a multi-step imagination module, and an autoregressive language head. The architecture separates the token time axis from a state time axis. The token axis encodes the current observation, while the state axis maintains and rolls forward a structured latent hypothesis about the world. This separation makes state lifecycle, object representations, counterfactual futures, and uncertainty explicit interfaces rather than implicit behavior of a language decoder.

We define the architecture, its computation, its training objectives, and a validation program against parameter-matched Transformer baselines. The current implementation has completed structural initialization and Metal inference validation of a logical 7B configuration on an Apple Silicon Mac mini; this implementation note is not an empirical claim about language quality. OCA remains a research hypothesis. Its value must be established by controlled experiments on object permanence, intervention prediction, and multi-step planning.

## 1. Introduction

Recurrent and attention-based sequence models have made substantial progress in language modeling and sequence transduction. A language model, however, can produce a plausible continuation without maintaining an explicit account of which entities exist, which properties changed, or which events caused those changes. The distinction is important in situations where the latest observation is incomplete: an occluded object should usually remain part of the world state, while a genuinely removed object should not.

The central question of this work is whether a language model benefits from an explicit state variable whose lifecycle is different from the token context. We propose Orbit Continuum Architecture (OCA) to investigate this question. OCA does not discard attention. Instead, it assigns different responsibilities to different computation paths:

* perception attention represents relations in the current observation;
* object slots provide a compact, inspectable binding space;
* the continuum state carries information between observations;
* the transition model predicts action-conditioned futures; and
* the language head communicates a selected state and prediction.

We use “world understanding” operationally. A model should be evaluated on whether it can preserve object permanence, represent event order, predict interventions, preserve unaffected objects, revise a belief after contradictory evidence, and calibrate uncertainty. We do not infer these capabilities from fluent text alone.

## 2. Background

The Transformer introduced a sequence transduction architecture based on stacked self-attention and position-wise feed-forward layers, removing sequence-aligned recurrence from the main computation. Its encoder maps an input sequence to continuous representations and its autoregressive decoder produces an output sequence. OCA retains this useful decomposition but adds a persistent state path.

In a standard decoder-only language model, information about an earlier event is normally represented by hidden activations and a context-dependent KV cache. These representations are powerful, but the model has no mandatory variable corresponding to an object, an intervention, or a world state. OCA therefore treats the language sequence as an observation and communication channel, not as the only memory of the world.

The proposal combines familiar ingredients—self-attention, gated updates, object-centric slots, and latent dynamics—into a single interface. This paper does not claim that any ingredient is independently new. The research question is whether their separation and joint training improve measurable state and planning behavior.

## 3. Model Architecture

Given an observation token sequence `X_t`, an optional previous state `S_{t-1}`, and an action representation `A_t`, OCA returns language logits, the present state, bound object slots, and imagined future states:

\[
(Y_t,S_t,Z_t,\hat S_{t+1:t+R}) = f_\theta(X_t,S_{t-1},A_t).
\]

The computation is:

```text
X_t → perception P → features H_t → slot binding B → slots Z_t
                                             ↓
                         previous state S_{t-1} → update U → S_t
                                                               ↓
                                              transition T(S_t, A_t)
                                                               ↓
                                            imagined states Ŝ_{t+1:t+R}
                                                               ↓
                         H_t + state context → decoder D → logits Y_t
```

### 3.1 Perception Stack

Token and positional embeddings are added before a stack of non-causal attention blocks:

\[
H_t=P(X_t)=P_{L_p}(\cdots P_2(P_1(E(X_t)+R))).
\]

The perception stack may use standard multi-head attention and a gated feed-forward block. Non-causal attention is appropriate because the input window is the current observation. It does not expose future states generated by the transition module.

### 3.2 Competitive Object-Slot Binding

Let `K` be the number of slots. Slots are initialized from learned seeds and iteratively bind observation features. At iteration `i`:

\[
Q_i=W_qS_i,\quad K_t=W_kH_t,\quad V_t=W_vH_t,
\]

\[
A_i=\operatorname{softmax}_{slots}
\left(\frac{Q_iK_t^T}{\sqrt W}\right),
\qquad
M_i=A_iV_t.
\]

The slot update is:

\[
G_i=\sigma(W_g[S_i;M_i]),
\]

\[
S_{i+1}=G_i\odot S_i+(1-G_i)\odot
\tanh(W_c[S_i;M_i]).
\]

The slot bottleneck encourages the model to represent a small number of persistent entities rather than distributing every object property across unrelated token positions. Slot identity is not assumed to be semantically fixed; training must handle permutation and correspondence explicitly.

### 3.3 Continuum-State Update

The bound slots `Z_t` are merged with the previous state using a gated recurrent update:

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

The state is explicit at the application boundary. A new episode passes a null or learned initial state; a continuing episode passes `S_{t-1}`; and an independent episode must reset the state. This lifecycle is different from a KV cache, which is an implementation structure for attention within a context.

### 3.4 Action-Conditioned Transition

The transition model predicts how a proposed action changes the present state. An action embedding is projected into the state width and combined with slot representations:

\[
U_0=S_t+W_aA_t.
\]

Transition blocks model relations among slots and produce a state delta and confidence:

\[
\Delta_i=\tanh(W_\Delta F_i),
\qquad c_i=\sigma(W_cF_i),
\]

\[
\hat S_{t+1}=S_t+c_i\odot\Delta_i.
\]

The confidence gate limits the effect of uncertain dynamics. Rollout applies the same transition family repeatedly:

\[
\hat S_{t+j}=T(\hat S_{t+j-1},A_t),\quad j=1,\ldots,R.
\]

### 3.5 Language Head

The language decoder receives perception features plus a pooled state context:

\[
C_t=H_t+W_s\operatorname{pool}(S_t),
\]

\[
Y_t=D(C_t),\qquad
P(y_i|y_{<i},X_t,S_t)=\operatorname{softmax}(W_o\operatorname{RMSNorm}(Y_{t,i})).
\]

The language head is autoregressive and causal. The state path is not intended to replace token-level language modeling; it supplies a persistent hypothesis that language can inspect and describe.

### 3.6 Formal Forward Algorithm

```text
Algorithm 1: OCA forward pass
Input: tokens X, previous state S_prev, action A

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
return logits, S, Z, futures
```

## 4. Why a Continuum Architecture?

### 4.1 Persistent State and Partial Observation

If an entity is not present in the current observation, a model should distinguish “not observed” from “does not exist.” A persistent state provides a location for this distinction. The state update can preserve a slot when evidence is absent and revise it when evidence contradicts the previous hypothesis.

### 4.2 Object Binding and Diagnostic Access

Token representations may contain object information without exposing where it is stored. Slots provide a small interface that can be decoded, supervised, and compared before and after an intervention. This does not guarantee interpretability; it makes a hypothesis testable.

### 4.3 Counterfactual Rollout

A language answer can be produced without explicitly simulating an action. OCA makes a transition path available so that training can penalize incorrect action consequences and preserve unrelated state. Imagination depth `R` is configurable and can be reduced at inference time.

### 4.4 State Lifecycle

The explicit initialize/update/reset interface prevents accidental cross-episode memory. It also allows a workspace state to be serialized, inspected, or discarded independently from the language context.

## 5. Training

### 5.1 Data and Batching

The initial curriculum should use synthetic worlds with objects, locations, visibility, time, and explicit actions. A sample contains observations, an action, the true present state, the true future state, and an answer target:

```text
[RESET] observation_1 [OBS] observation_2 [ACTION] action
[TARGET_STATE] state_t [TARGET_FUTURE] state_t+1 ... [ANSWER] answer
```

Each batch retains token IDs, `previous_state`, action embeddings, target slots or attributes, target future states, and an intervention mask identifying unaffected objects.

### 5.2 Objective

\[
\mathcal L=\mathcal L_{lang}
+\lambda_s\mathcal L_{state}
+\lambda_d\mathcal L_{dyn}
+\lambda_c\mathcal L_{cycle}
+\lambda_u\mathcal L_{uncertainty}.
\]

`L_lang` is next-token loss. `L_state` reconstructs object attributes and visibility. `L_dyn` predicts the state after an action. `L_cycle` penalizes changes to objects outside the intervention mask. `L_uncertainty` calibrates confidence under occlusion and conflicting observations.

### 5.3 Scaling and Implementation Status

The implementation is written in MLX. A logical 7B configuration has been initialized and quantized, and its Metal forward path has been verified on an Apple Silicon Mac mini. This is a deployment check, not a training result. Full 7B pretraining requires a larger compute plan; the local machine is appropriate for architecture experiments, quantized inference, small-scale training, and distillation.

## 6. Experiments

We propose a parameter-matched decoder-only Transformer baseline and the following tasks:

| Task | Measurement |
|---|---|
| Hidden-object tracking | object identity and attributes after occlusion |
| Intervention prediction | target state after an explicit action |
| Unaffected-object preservation | state change outside the intervention mask |
| Long-horizon memory | stability across long observation sequences |
| Multi-step planning | success as imagination depth changes |
| Uncertainty calibration | confidence under incomplete evidence |

Required ablations remove object slots, persistent state, the transition model, or all but one imagination step. A useful result must generalize to unseen object/action combinations and remain stable across random seeds.

At the current stage, these comparative experiments have not yet been run. The verified results are limited to software execution: the 7B configuration produces logits of shape `(1, 8, 65536)`, state of shape `(1, 32, 4096)`, and six imagined states of shape `(1, 6, 32, 4096)`.

## 7. Conclusion

OCA adds an explicit persistent state, object-slot binding, and action-conditioned imagination to a Transformer-based language system. Its central claim is architectural and testable: separating observation tokens from a persistent world-state path may improve object permanence, intervention prediction, and multi-step planning.

The current implementation establishes the architecture and a runnable logical 7B configuration. It does not establish language quality or general world understanding. The next step is controlled Tiny-scale training with state and dynamics supervision, followed by parameter-matched ablations and only then larger-scale training.

## References

1. Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
2. Locatello, F. et al. (2020). *Object-Centric Learning with Slot Attention.* NeurIPS.
3. Ha, D. and Schmidhuber, J. (2018). *World Models.* NeurIPS Workshop.
4. Hafner, D. et al. (2020). *Dream to Control: Learning Behaviors by Latent Imagination.* ICLR.
5. Chung, J. et al. (2014). *Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.* NeurIPS Workshop.

---

## Copyright

© 2026 YUNSH and Liu Jiacheng (刘家成), a 14-year-old middle-school student from China. All rights reserved unless a separate license is provided with a specific software file.

