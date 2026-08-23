# Orbit Continuum Architecture (OCA)

## Research hypothesis

A decoder trained only to predict the next token has no architectural need to
maintain a stable account of objects, time, hidden state, or interventions.
OCA adds those requirements explicitly while retaining transformer components
where they are useful.

Its central variable is the **continuum state**:

\[
S_t = U(S_{t-1}, B(P(O_t)))
\]

where `P` perceives observation `O`, `B` binds features into object slots, and
`U` selectively updates the prior state. An action-conditioned transition model
predicts counterfactual futures:

\[
\hat S_{t+k} = T^k(S_t, A_t)
\]

The language decoder is conditioned on `S_t`; it is not the only store of world
information.

## What is actually new in the prototype

- Persistent state is distinct from attention KV cache and has an explicit
  lifecycle.
- Competitive object slots impose a small, inspectable state bottleneck.
- State transition is action-conditioned and reusable for multi-step rollout.
- Perception, dynamics, and language losses can be trained and ablated
  independently.
- Confidence gates limit how much a speculative transition can modify state.

These are research hypotheses, not a claim that OCA already understands the
real world. Similar ingredients exist in world models, slot attention, recurrent
models, and model-based reinforcement learning. OCA's contribution must be
judged by the combined system and experiments, followed by a prior-art review
before any originality or patent claim.

## Training objectives

The intended objective is:

\[
L = L_{language} + \lambda_s L_{state} + \lambda_d L_{dynamics}
  + \lambda_c L_{cycle} + \lambda_u L_{uncertainty}
\]

- `language`: next-token and instruction response accuracy.
- `state`: reconstruct known object attributes from slots.
- `dynamics`: predict state after an explicit intervention.
- `cycle`: avoid changing unrelated objects during an intervention.
- `uncertainty`: calibrate predictions when observations are incomplete.

## Validation gates before scaling

1. Beat a parameter-matched transformer on hidden-object tracking.
2. Generalize to unseen object/action combinations.
3. Preserve unaffected object state after interventions.
4. Improve multi-step planning as imagination depth increases.
5. Show through ablation that slots and persistent state each contribute.
6. Remain stable over long sequences without state collapse.

Only after these gates pass should Orbit move from tiny to 300M/1B and finally
to the 7B specification. A 7B model can be quantized for inference on a 24 GB
Mac, but serious pretraining will require external compute.

## 7B scaling specification

The `7b` preset uses a 4096-wide latent space, 27 total attention/dynamics
blocks, 32 object slots, grouped-query-ready head geometry, and six imagination
steps. Exact parameter count will change as the efficient attention backend,
tokenizer, tied embeddings, and mixture-of-experts option are finalized. The
configuration is therefore a target envelope, not a misleading exact claim.
