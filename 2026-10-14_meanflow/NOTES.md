# MeanFlow: fact sheet + Hacker prep (Wed 14 Oct)

Geng, Deng, Bai, Kolter, He. "Mean Flows for One-step Generative Modeling". arXiv 2505.13447
(CMU + MIT). PDF: [`doc/08_MeanFlow_2505.13447.pdf`](../doc/08_MeanFlow_2505.13447.pdf). Every
number below was checked against that PDF.

## The idea in four lines

- Flow Matching learns the **instantaneous** velocity v(z_t, t). Sampling integrates it, and
  the marginal paths are curved (§3), so 1 step is bad.
- MeanFlow learns the **average** velocity over [r, t]:
  u(z_t, r, t) = 1/(t−r) ∫_r^t v(z_τ, τ) dτ (Eq. 3). One step: z_0 = z_1 − u(z_1, 0, 1).
- The integral is intractable, so differentiate (t−r)u = ∫ v w.r.t. t to get the
  **MeanFlow identity** (Eq. 6): u = v − (t−r) du/dt.
- The total derivative is du/dt = v ∂_z u + ∂_t u (Eq. 8). This is one JVP with tangent (v, 0, 1).

## Algorithm 1 (training), verbatim shape

```
t, r = sample_t_r()
e = randn_like(x)
z = (1 - t) * x + t * e
v = e - x
u, dudt = jvp(fn, (z, r, t), (v, 0, 1))
u_tgt = v - (t - r) * dudt
error = u - stopgrad(u_tgt)
loss = metric(error)
```

- With r = t, the second term vanishes and this is exactly Flow Matching.
- The stop-gradient means no double backprop. JVP overhead is < 20% of training time (JAX).

## Defaults (Tab. 4) and ablations (Tab. 1, B/4, 80 ep, 1-NFE FID)

| knob | values → FID | default |
| --- | --- | --- |
| ratio r≠t (1a) | 0% (=FM) 328.91 · **25% 61.06** · 50% 63.14 · 100% 67.32 | 25% |
| JVP tangent (1b) | **(v,0,1) 61.06** · (v,0,0) 268.06 · (v,1,0) 329.22 · (v,1,1) 137.96 | (v,0,1) |
| embedding (1c) | (t,r) 61.75 · **(t,t−r) 61.06** · (t,r,t−r) 63.98 · t−r only 63.13 | (t, t−r) |
| sampler (1d) | uniform 65.90 · **lognorm(−0.4,1.0) 61.06** · others 61.79 to 64.72 | lognorm(−0.4, 1.0) |
| loss power p (1e) | 0 (L2) 79.75 · 0.5 63.98 · **1.0 61.06** · 1.5 66.57 · 2.0 69.19 | 1.0, w = 1/(‖Δ‖²+c)^p |
| CFG ω (1f) | 1.0 61.06 · 1.5 33.33 · 2.0 20.15 · **3.0 15.53** · 5.0 20.75 | |

Headline (Tab. 2, ImageNet 256, from scratch, no distillation, no curriculum):

- MF-XL/2 1-NFE: **3.43** (B/2 6.17, M/2 5.01, L/2 3.84).
- 2-NFE: 2.93, and XL/2+ gets **2.20**.
- vs Shortcut 10.60 (1-NFE), IMM 7.77 (1×2 NFE).
- DiT-XL/2 2.27 and SiT-XL/2 2.06 use 250×2 NFE. The §5.2 text says SiT is 2.15. The paper is inconsistent with itself here; quote the table.
- CIFAR-10 uncond 1-NFE: 2.92.

**CFG is folded into the target** (Eq. 19): ṽ = ω v_t + (1−ω) u_θ(z_t, t, t). Sampling stays at 1 NFE.

## What my toy found (2D checkerboard, `meanflow.py`, `demo.ipynb`)

- Same net (MLP, 3 hidden layers of 512), same 20k steps, same everything. Only `ratio` or `tangent`
  changes. The shared settings are the paper's **CIFAR-10** ones (App. A): 75% r≠t,
  lognorm(−2, 2), p = 0.75. The ImageNet defaults (25%, lognorm(−0.4, 1), p = 1) gave a smeared
  1-step checkerboard: an X with smeared corners instead of squares.
- **Hacker gotcha:** with DiT-style high-frequency time embeddings (max freq ~1000), du/dt blows up
  through the JVP and training diverges (ratio 75% went to SW distance > 1000). Low-frequency
  sin/cos of t fixed it. Paper scale works because time is embedded then passed through an MLP; I
  did not test that.
- My first metric, sliced-Wasserstein, ranked a bad 1-step sample above a good 50-step one, so I
  switched to histogram overlap (1 − TV on a 40×40 grid). **Look at the pictures**, not just one number.
- Numbers (histogram overlap with the data, 1 = perfect; two fresh data samples score 0.92):

  | NFE | 1 | 2 | 4 | 8 | 16 | 50 |
  | --- | --- | --- | --- | --- | --- | --- |
  | Flow Matching | **0.02** | 0.50 | 0.62 | 0.67 | 0.69 | 0.68 |
  | MeanFlow | **0.56** | 0.63 | 0.67 | 0.68 | 0.68 | 0.68 |

- Tangent ablation, 1 step: (v,0,1) 0.56 · (v,1,1) 0.38 · (v,1,0) 0.28 · (v,0,0) 0.27. Same order as
  the paper's FIDs (61 < 138 < 268 < 329, with the last two swapped but both broken).
- (v,0,0) and (v,1,0) give a Gaussian ball: the jump barely moves the noise.
- JVP vs finite difference (float64): max error 2e-6, against a |du/dt| scale of 1.8.
- Training: 5 models × 20k steps, about 9 min in parallel on the M3 Pro CPU. The notebook runs in about 25 s from saved weights.

## Run of show, 5 min (your words; these are cues)

1. **30 s, the problem.** Flow Matching 1 step = a dot in the middle. Why: at t = 1, z = ε carries
   no information about x, so the ideal v(z, 1) = z − E[x]. One Euler step lands **exactly on the
   data mean**, for every sample.
2. **90 s, the one line.** Show `loss_fn`. The target is v minus (t−r)·dudt, and dudt comes from
   one `jvp` with tangent (v, 0, 1). Delete that term and you have Flow Matching.
3. **90 s, the result.** MF 1 step vs FM 1 step vs FM 50 steps. Then the NFE curve.
4. **60 s, the ablation.** Wrong tangent, 1 step falls apart. Same ordering as Tab. 1b: (v,1,1)
   hurts least.
5. **30 s, question to the room.** Pick one:
   - The target uses the network's own derivative under stop-grad. What stops it from
     converging to a wrong u?
   - Why does 25% r≠t beat 100%?

## Slack draft to the presenters (Leslie, Zihan)

> Hi Leslie and Zihan, I'm the Hacker for MeanFlow. I'll show a from-scratch toy (2D
> checkerboard): Flow Matching vs MeanFlow at 1 step, plus the JVP tangent ablation from Tab. 1b.
> To avoid overlap, I'll put the ~10 lines of Alg. 1 on screen, so you can keep the identity
> derivation short and point at my demo for the "why the JVP" part. Any slot works for me. After
> your method section is probably most natural. Juntang
