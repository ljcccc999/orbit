# Orbit Continuum Architecture (OCA)
## A Persistent World-State and Causal-Imagination Architecture

**Research Preprint · Version 0.1 · 2026-08-03**  
**Project:** Orbit  
**Authors:** YUNSH · Liu Jiacheng (China, age 14, middle-school student)

> This paper describes a working research prototype. “7B” denotes the logical parameter scale of the model configuration. The current 7B checkpoint contains quantized random initialization for structural validation and has not completed language pretraining; it must not be described as a mature conversational foundation model.

## Abstract

Mainstream language models typically compress information about the world into a context sequence and an attention cache, with next-token prediction as the dominant training objective. This produces fluent language, but does not require an explicit account of objects, time, hidden state, action consequences, or observational uncertainty.

We propose the Orbit Continuum Architecture (OCA), a language-model architecture centered on persistent world state. OCA separates perception, object-slot binding, continuum-state updating, action-conditioned state transition, future-state imagination, and language decoding. Its central state variable is distinct from the token KV cache and is maintained across observations through gated updates. Competitive object slots provide an inspectable latent bottleneck for object identity and attributes. A causal transition module rolls the state forward under an explicit action to produce multiple imagined futures.

We have completed FP16 weight initialization, 4-bit quantization, Metal forward inference, and persistent-state handoff validation for the OCA 7B configuration on an Apple Silicon Mac mini. We do not claim that OCA has solved general world understanding. Instead, we provide falsifiable architectural hypotheses and an experimental program: if explicit persistent state, object binding, and causal imagination are useful, OCA should outperform a parameter-matched language-only baseline on hidden-object tracking, intervention prediction, and multi-step planning.

**Keywords:** Orbit; OCA; world model; persistent state; object slots; causal imagination; Apple Silicon; MLX

---

## 1. Introduction

Language is a projection of the world, not the world itself. When a person says “the cup is occluded,” the statement implies at least three facts: the cup continues to exist, its visibility has changed, and it may reappear if the occluder is removed. A next-token model can answer such questions through statistical correlation, but it is not required to maintain a stable object state or distinguish disappearance from temporary unobservability.

OCA does not reject the Transformer. It reallocates responsibilities. Attention is useful for relations within a current observation, and a language decoder is useful for expression and interaction. Persistent objects, temporal change, and action consequences require more explicit internal variables. OCA therefore retains Transformer blocks while adding a transferable, resettable, and inspectable continuum state.

We define “world understanding” as a set of testable abilities:

1. preserve object permanence under occlusion;
2. represent event order;
3. predict action-induced state changes;
4. preserve unaffected objects after interventions;
5. revise beliefs when new observations conflict with memory; and
6. calibrate confidence under incomplete information.

Reproducible success on these tests is required before making stronger claims about world models.

## 2. Architecture Overview

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

### 2.1 Perception

Given token sequence `X_t`, the perception module adds token and positional embeddings and applies non-causal self-attention over the current observation window:

\[
H_t = P(X_t) = P_L(\cdots P_2(P_1(E(X_t) + R))).
\]

Non-causal perception does not allow the model to read the future world. It only permits tokens within the current input window to interact. Temporal futures are produced by the separate transition module.

### 2.2 Object-slot Binding

OCA maintains a fixed number of latent slots. A slot is not assigned a predefined object category; it is a learnable object hypothesis. Repeated competitive attention lets observation features compete for slots:

\[
A^{(i)} = \operatorname{softmax}_{slots}
\left(\frac{Q(S^{(i)})K(H_t)^T}{\sqrt d}\right),
\]

\[
S^{(i+1)} = G^{(i)} \odot S^{(i)} + (1-G^{(i)}) \odot
\tanh\left(C([S^{(i)}; A^{(i)}V(H_t)])\right).
\]

The number of slots is an explicit capacity assumption: too few slots cause object entanglement, while too many can create redundancy and instability. Because slots can be read and supervised individually, they are easier to diagnose than object information distributed entirely across token representations.

### 2.3 Continuum State

Let the previous state be `S_{t-1}` and the currently bound slots be `B_t`. The continuum updater uses an update gate, reset gate, and candidate state:

\[
z_t = \sigma(W_z[S_{t-1};B_t]),
\quad r_t = \sigma(W_r[S_{t-1};B_t]),
\]

\[
\tilde S_t = \tanh(W_c[r_t\odot S_{t-1};B_t]),
\quad S_t = (1-z_t)\odot S_{t-1}+z_t\odot\tilde S_t.
\]

This state has a different lifecycle from the KV cache. The KV cache serves efficient attention within a context, while the continuum state represents a world hypothesis across observations. An application can initialize it at session start, explicitly reset it between independent tasks, or serialize it as a workspace state.

### 2.4 Causal Imagination

Given current state `S_t` and action representation `A_t`, the transition module generates future states:

\[
\hat S_{t+1} = T(S_t,A_t),
\qquad
\hat S_{t+k} = T(\hat S_{t+k-1},A_t).
\]

Each transition includes a confidence gate that limits how much an uncertain prediction can modify the state. The current 7B configuration uses six imagination steps by default. The purpose of rollout is not to create an internal animation; it is to compare consequences before producing an answer or executing an action.

### 2.5 Language Decoding

The language decoder receives perception features and injects a pooled context derived from the current continuum state:

\[
Y_t = D(H_t + W_s\operatorname{pool}(S_t)),
\qquad
\text{logits}_t = W_o\operatorname{RMSNorm}(Y_t).
\]

This allows language generation to read the world state without requiring the language sequence itself to carry the entire state representation.

## 3. Concrete Implementation

The first implementation is in `orbit/model.py` and uses MLX `nn.Module` for Apple Silicon unified memory and Metal kernels.

| Theoretical module | Implementation class | Input / output |
|---|---|---|
| Perception `P` | `AttentionBlock`, `causal=False` | `[B,T,W] → [B,T,W]` |
| Object binding `B` | `SlotBinder` | `[B,T,W] → [B,K,W]` |
| Continuum update `U` | `ContinuumUpdater` | `[B,K,W] × [B,K,W] → [B,K,W]` |
| Causal transition `T` | `CausalTransition` | `[B,K,W] × [B,W] → [B,K,W]` |
| Language decoder `D` | `AttentionBlock`, `causal=True` | `[B,T,W] → [B,T,W]` |

Here `B` is batch size, `T` is token count, `K` is slot count, and `W` is hidden width.

The forward pass is implemented in this order:

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

`SlotBinder` uses Q/K/V projections and competitive softmax over the slot dimension. At each iteration, token-to-slot affinities are computed, values are aggregated, and a gate interpolates between the previous slot and a candidate slot.

`ContinuumUpdater` does not overwrite the old state. It computes update, reset, and candidate gates. A new episode passes `None` as `previous_state`; a continuing observation explicitly passes the previous state, preventing implicit memory leakage across tasks.

`CausalTransition` consumes a pooled observation/action embedding and applies a shared action condition to all slots. Each step models slot interactions, predicts a `delta` and a `confidence`, and returns:

```text
next_state = state + confidence * tanh(delta)
```

The tokenizer is not yet connected to a real corpus; the prototype currently uses integer token tensors to validate the architecture. A production training format must fix the vocabulary, special tokens, state-reset token, and action spans:

```text
[RESET] observation_1 [OBS] observation_2 [ACTION] action
[TARGET_STATE] ... [TARGET_FUTURE] ... [ANSWER] ...
```

Training batches must retain `tokens`, `previous_state`, `action`, `target_state`, `target_future`, and `object_mask`.

## 4. Training Objective

OCA is intended to use a multi-objective loss:

\[
\mathcal L = \mathcal L_{lang}
+ \lambda_s\mathcal L_{state}
+ \lambda_d\mathcal L_{dyn}
+ \lambda_c\mathcal L_{cycle}
+ \lambda_u\mathcal L_{uncertainty}.
\]

- `L_lang`: standard next-token and instruction-response loss;
- `L_state`: reconstruct object attributes, locations, and visibility from slots;
- `L_dyn`: predict the true next state after an action;
- `L_cycle`: penalize changes to unrelated objects;
- `L_uncertainty`: calibrate confidence under occlusion or conflicting observations.

The repository currently contains an MLX language-loss smoke trainer and a synthetic-world generator. Implementing the state and dynamics losses is the next training milestone.

## 5. Formal Forward Algorithm

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

## 6. Computational Complexity

Let observation length be `T`, hidden width `W`, perception depth `Lp`, decoder depth `Ld`, slot count `K`, and imagination depth `R`. Standard self-attention contributes the dominant `O(T²W)` term. OCA adds:

- slot binding: `O(IKTW)` for `I` binding iterations;
- continuum update: `O(KW²)`;
- transition rollout: `O(RK²W + RKW²)`;
- state-to-decoder conditioning: `O(TW²)` for projection and `O(TW)` for broadcast addition.

When `K << T`, the slot and imagination branches do not replace token attention; they add temporal and causal capacity with a smaller structured-state cost. At inference time, `R` and `K` can be reduced, and the imagination branch can be enabled only for planning-sensitive requests.

## 7. Structural Difference from a Standard Transformer

| Component | Decoder-only Transformer | OCA |
|---|---|---|
| Current observation | token sequence | token sequence + perception stack |
| Long-term state | implicit in context/KV cache | explicit continuum state |
| Object representation | distributed token features | competitive object slots |
| Action consequences | learned implicitly in logits | action-conditioned transition |
| Internal planning | optional prompting/tool use | multi-step latent imagination |
| State lifecycle | context dependent | explicit initialize/update/reset |
| Diagnostics | token/logit inspection | token + slots + state + futures |

This table describes architectural interfaces, not performance claims. Only parameter-matched baselines and ablations can determine whether the additional components produce real gains.

## 8. 7B Configuration and Mac mini Result

The 7B configuration uses:

- hidden width: 4096;
- perception blocks: 10;
- decoder blocks: 12;
- transition blocks: 5;
- attention heads: 32;
- object slots: 32;
- imagination steps: 6;
- vocabulary size: 65,536;
- logical parameter estimate: 7,306,608,640.

On an Apple Silicon Mac mini, we have completed:

1. OCA 7B FP16 weight initialization;
2. MLX 4-bit quantization;
3. a local quantized checkpoint of approximately 3.9 GB;
4. Metal forward inference; and
5. continuum-state handoff across observations.

The verified output shapes are:

```text
logits: (1, 8, 65536)
continuum state: (1, 32, 4096)
imagined states: (1, 6, 32, 4096)
```

These results validate software structure, weight format, and the Metal inference path. They do not demonstrate language knowledge or general world understanding. The current 7B checkpoint is a structural model produced from random initialization and quantization.

## 9. Training Procedure and Memory Strategy

Training should proceed in stages. Start with Tiny, verify that language loss decreases and slots reconstruct object state, then jointly train language, state, and dynamics losses. Only afterward should width, sequence length, and imagination depth be increased. Every stage should save optimizer state, model configuration, tokenizer hash, dataset version, and random seed.

On a 24 GB Mac mini, 7B is appropriate for 4-bit inference and structural validation, not full Adam pretraining. Practical 7B training routes are external multi-GPU pretraining, LoRA/QLoRA adaptation, or distillation from a smaller trained OCA model.

The 4-bit checkpoint uses:

```python
nn.quantize(model, bits=4, group_size=64)
mx.save_safetensors("weights.safetensors", weights)
```

`model.json` records both the logical parameter estimate and the packed storage-element count. Quantized `uint32` storage elements must never be mislabeled as the logical model parameter count.

## 10. Experimental Design

Use a parameter-matched decoder-only Transformer baseline and perform component ablations.

| Task | Measurement |
|---|---|
| Hidden-object tracking | object identity and attributes under occlusion |
| Intervention prediction | target state after an action |
| Unaffected-object preservation | unchanged state of unrelated objects |
| Long-horizon memory | state stability over long sequences |
| Multi-step planning | effect of imagination depth on success |
| Uncertainty calibration | confidence under ambiguity |

Required ablations include removing slots, removing persistent state, removing the transition model, setting imagination depth to one, and concatenating state directly into token embeddings. An architectural hypothesis is supported only if gains persist across random seeds and unseen object/action combinations.

## 11. Relation to Prior Work

OCA does not claim independent invention of every component. Transformers provide the attention and sequence-modeling foundation; Slot Attention demonstrates unsupervised object-slot binding; World Models and Dreamer study latent dynamics and imagination; recurrent models have maintained hidden state for decades.

The research question is whether these ideas can be organized into one system serving language output, object permanence, and action-conditioned imagination, while improving diagnosability through an explicit state lifecycle. Any patent or originality claim requires systematic prior-art search, claim comparison, and experimental evidence. This paper is an engineering research preprint and is not an opinion on patentability.

## 12. Limitations and Honest Boundaries

First, the current 7B weights are randomly initialized and cannot conduct meaningful natural-language conversation. Second, synthetic worlds are not substitutes for real visual, physical, or social environments. Third, slots do not automatically guarantee semantic object correspondence and may suffer from slot permutation, object entanglement, or state collapse. Fourth, tokenizer integration, real multimodal data pipelines, state-supervision losses, and long-horizon training remain incomplete. Fifth, a Mac can support quantized inference and small experiments, but cannot pretrain a competitive 7B foundation model from scratch in one day.

## 13. Conclusion and Roadmap

OCA provides a clear engineering decomposition: perception reads the current observation, slots bind objects, the continuum state maintains a world hypothesis, the transition model predicts consequences, and the language decoder communicates. Its value must be determined by reproducible experiments, not by its name or parameter count.

The next steps are to complete the tokenizer and training format; add state, dynamics, and uncertainty losses; train Tiny OCA against a parameter-matched baseline; scale to 300M/1B after passing synthetic-world gates; and only then consider large-scale 7B training or distillation. If these experiments show stable gains, Orbit can cautiously claim progress toward world understanding.

## References

1. Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
2. Locatello, F. et al. (2020). *Object-Centric Learning with Slot Attention.* NeurIPS.
3. Ha, D. and Schmidhuber, J. (2018). *World Models.* NeurIPS Workshop.
4. Hafner, D. et al. (2020). *Dream to Control: Learning Behaviors by Latent Imagination.* ICLR.
5. Hafner, D. et al. (2023). *Mastering Diverse Domains through World Models.* arXiv.
6. Chung, J. et al. (2014). *Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.* NeurIPS Workshop.

---

## Copyright

© 2026 YUNSH and Liu Jiacheng (刘家成), a 14-year-old middle-school student from China. All rights reserved unless a separate license is provided with a specific software file.

