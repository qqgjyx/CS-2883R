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

## What my toy found (`meanflow.ipynb`, 2D checkerboard)

- One notebook, about 40 lines of method code.
  - Five models are identical except one knob: `ratio` (0 = Flow Matching) or the JVP tangent.
  - Shared settings are the paper's **CIFAR-10** ones (App. A): 75% r≠t, lognorm(−2, 2), p = 0.75.
  - Model: MLP on [z, t, t−r], 3 × 512, 10k steps, MPS.
- **Checked against the authors' PyTorch code.** On the same inputs, `loss_fn` gives identical
  loss and gradients to `forward_with_loss` in [Gsunshine/py-meanflow](https://github.com/Gsunshine/py-meanflow)
  (the first author's port, CIFAR-10). JVP vs finite difference (float64): 1e-8.
- Results: histogram overlap with the data, where 1 = identical and two fresh data samples score 0.92.

  | NFE | 1 | 2 | 4 | 8 | 16 | 50 |
  | --- | --- | --- | --- | --- | --- | --- |
  | Flow Matching | **0.03** | 0.51 | 0.60 | 0.68 | 0.70 | 0.68 |
  | MeanFlow | **0.61** | 0.63 | 0.65 | 0.64 | 0.64 | 0.63 |

  MeanFlow at 1 step ≈ Flow Matching at 4. With many steps Flow Matching ends slightly higher,
  as in the paper (3.43 at 1 step vs 2.06 for SiT at 250 steps).
- Tangent ablation, 1 step: (v,0,1) 0.61 > (v,1,1) 0.40 > (v,0,0) 0.28 > (v,1,0) 0.00. This is
  **exactly the paper's FID order** (61 < 138 < 268 < 329).
  - (v,0,0) leaves a Gaussian ball: the jump barely moves the noise.
  - (v,1,0) throws every sample off the plot.
- Hacker findings, if asked:
  - **Time input.** DiT-style sin/cos embeddings with high frequencies (~1000) blew up du/dt through
    the JVP, and training diverged. Feeding raw t and t−r was both simplest and best: 0.61 vs 0.56
    at 1 step.
  - **Settings.** The paper's ImageNet settings (25%, lognorm(−0.4, 1), p = 1) gave a smeared X on
    this toy. The CIFAR-10 settings fixed it.
  - **Metric.** Sliced-Wasserstein ranked a smeared 1-step sample above a clean 50-step one, so look
    at the pictures.
- Prior art I checked (12 Oct):
  - The authors' JAX code ([Gsunshine/meanflow](https://github.com/Gsunshine/meanflow)) and their
    PyTorch CIFAR-10 port.
  - [haidog-yaqub/MeanFlow](https://github.com/haidog-yaqub/MeanFlow): 1.2k★, MNIST to ImageNet.
  - A 2D tutorial comparing Shortcut and MeanFlow
    ([sdt201909](https://github.com/sdt201909/Flow_Matching_Tutorial_For_Beginner)).
  - None of them does FM vs MF at 1 step plus the tangent ablation.

## Run of show, 5 min (your words; these are cues)

1. **30 s, F1 left half: the problem.** Flow Matching 1 step = a dot. At t = 1 the noise carries
   no information about x, so v(z, 1) = z − E[x], and one step lands **on the data mean**.
2. **90 s, the `loss_fn` cell.** u_tgt = v − (t−r)·dudt, where dudt is one `jvp` with tangent
   (v, 0, 1). With r = t this is Flow Matching. It is the same as the authors' PyTorch loss, checked.
3. **60 s, F1 right + F2.** MeanFlow gets the board in 1 jump. The FM paths curve, and the MF jump
   is the chord.
4. **30 s, F3.** 1 MeanFlow step ≈ 4 Flow Matching steps. FM still wins with many steps, as in the
   paper.
5. **60 s, F4.** Wrong tangent: the order matches Tab. 1b exactly. Dropping ∂t u means the jump
   goes nowhere.
6. **30 s, question to the room.** Pick one:
   - The target uses the network's own derivative under stop-grad. What stops it from converging
     to a wrong u?
   - Why do the toy and CIFAR want 75% r≠t when ImageNet wants 25%?

## Slack draft to the presenters (Leslie, Zihan)

> Hi Leslie and Zihan, I'm the Hacker for MeanFlow. I'll show a from-scratch toy (2D
> checkerboard): Flow Matching vs MeanFlow at 1 step, plus the JVP tangent ablation from Tab. 1b.
> To avoid overlap, I'll put the ~10 lines of Alg. 1 on screen, so you can keep the identity
> derivation short and point at my demo for the "why the JVP" part. Any slot works for me. After
> your method section is probably most natural. Juntang
