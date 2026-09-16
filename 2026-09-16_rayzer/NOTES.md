# RayZer: fact sheet

> Deliverable (single source of truth): <https://claude.ai/artifact/6LC7D89KEjrcYCsbNXsjZ8>
> Protocol and answer keys: `interrogation/PROTOCOL.md`

Working notes for CS 2883R, 2026-09-16. Every number below was re-extracted with
`pdftotext -layout` from `2505.00702v1.pdf`, not read off the rendered page.

## Identity

| | |
| --- | --- |
| Title | RayZer: A Self-supervised Large View Synthesis Model |
| Authors | Hanwen Jiang¹, Hao Tan², Peng Wang², Haian Jin³, Yue Zhao¹, Sai Bi², Kai Zhang², Fujun Luan², Kalyan Sunkavalli², Qixing Huang¹, Georgios Pavlakos¹ |
| Affiliations | ¹UT Austin, ²Adobe Research, ³Cornell |
| arXiv | 2505.00702v1, 1 May 2025 |
| Published | **ICCV 2025, Best Student Paper Honorable Mention** (stated on the GitHub repo; *not* anywhere in the arXiv v1 PDF) |
| Project | <https://hwjiang1510.github.io/RayZer/> |
| Code | <https://github.com/hwjiang1510/RayZer> |

**Claim in one sentence.** A feed-forward multi-view model trained with zero 3D supervision (no
camera poses, no geometry) that takes unposed and uncalibrated images, predicts cameras,
reconstructs a latent scene, and renders novel views, matching or beating "oracle" methods that
use COLMAP poses in both training and inference.

## Method: a pose-first cascade, with no explicit 3D anywhere

1. **Tokenize.** K images, ViT patchify. Resolution 256, patch size 16, so 16x16 = 256 tokens
   per image, linear layer to `R^d`, d = 768. Add sinusoidal spatial p.e. **plus a sinusoidal
   image-index p.e.** shared across all tokens of the same image. For video, that index encodes
   temporal order. Remember this; it does a lot of work later.
2. **Camera Estimator `E_cam`.** 8 full self-attention layers over `{f, p}`, where `p ∈ R^{K×d}`
   is one learnable camera token repeated K times plus image-index p.e. Outputs `p*`. The updated
   image tokens `f*` are **discarded**, used only as context.
   - *Pose:* one view is chosen canonical (the **middle** frame, not the first). For each other
     view, a 2-layer MLP on `[p*_i, p*_c]` gives `p_i ∈ R^9` (6D continuous rotation + 3D
     translation), mapped to `SE(3)`.
   - *Intrinsics:* a single focal length from a 2-layer MLP on `p*_c`. Assumes `fx = fy`, all
     views share intrinsics, principal point at image center.
3. **Plücker ray maps.** Cameras become pixel-aligned Plücker ray maps `R ∈ R^{K×H×W×6}`. This
   is the **only** 3D prior in the entire model.
4. **Split.** Images are split into two disjoint sets: `I_A` (predicts the scene) and `I_B`
   (provides supervision). `I_A ∪ I_B = I`, `I_A ∩ I_B = ∅`.
5. **Scene Reconstructor `E_scene`.** Fuse image tokens `f_A` and ray tokens `r_A` with a 2-layer
   MLP. Note it uses the **raw** tokens `f`, not the pose transformer's `f*`, precisely to stop
   `I_B` information leaking in. Then 8 self-attention layers over `{z, x_A}` with `z ∈ R^{L×d}`
   learnable, **L = 3072 scene tokens**. Output `z*`.
6. **Rendering Decoder `D_render`.** 8 self-attention layers over `{r, z*}` where `r` are the
   target view's Plücker ray tokens; `MLP_rgb` on the updated `r*` regresses patch RGB.
7. **Loss.** `L = (1/K_B) Σ_{I ∈ I_B} (MSE(I, Î) + λ·Percep(I, Î))`, **λ = 0.2**. Pure 2D
   photometric. Target views are rendered with RayZer's **own predicted poses**.

**24 transformer layers total = 8 + 8 + 8, d = 768.** The scene is a latent token set, not a
mesh / NeRF / 3DGS. Rendering is a learned function, not a rendering equation.

## Training setup

32 A100 GPUs, total batch size 256, 50,000 steps. LR warms up linearly over 3000 steps from 0 to
`4e-4`, cosine decay to a final `1.5e-4`. BF16 mixed precision, FlashAttention-2 via xFormers,
gradient checkpointing, QK-Norm, gradient clipping at 1.0. No bias terms in linear and norm
layers, depth-wise init (both following LVSM). Trained and evaluated on **each dataset
separately**; no cross-dataset transfer is reported.

| Dataset | inputs (size of I_A) | targets (size of I_B) | frame index range | curriculum start range |
| --- | --- | --- | --- | --- |
| DL3DV | 16 | 8 | 64-96 | 48-64 |
| RealEstate10K | 5 | 5 | 128-192 | 96-128 |
| Objaverse | 12 | 8 | 50-65 | 24-32 |

Objaverse is rendered as video: ~70 frames, azimuth 0 to 360 degrees, elevation sampled randomly
in [-20, 60] degrees, constant distance on a unit sphere, about one azimuth cycle.

## Results, exact

**Table 1, DL3DV**, continuous video frames. Oracles use COLMAP cameras at train and test.

| Method | Train sup. | Inf. w. COLMAP cam | Even: PSNR / SSIM / LPIPS | Random: PSNR / SSIM / LPIPS |
| --- | --- | --- | --- | --- |
| GS-LRM | 2D + Camera | Yes | 23.49 / 0.712 / 0.252 | 23.02 / 0.705 / 0.266 |
| LVSM | 2D + Camera | Yes | 23.69 / 0.723 / 0.242 | 23.10 / 0.703 / 0.257 |
| **RayZer** | 2D | No | **24.36 / 0.757 / 0.209** | **23.72 / 0.733 / 0.222** |

**Table 2, RealEstate**, COLMAP annotations.

| Method | Even | Random |
| --- | --- | --- |
| GS-LRM | 24.25 / 0.770 / 0.227 | 23.21 / 0.748 / 0.251 |
| LVSM | 27.00 / 0.851 / 0.157 | 25.88 / 0.828 / 0.175 |
| **RayZer** | **27.48 / 0.861 / 0.146** | **26.32 / 0.835 / 0.164** |

**Table 3, Objaverse**, Blender ground-truth poses. This is the one RayZer loses.

| Method | Even | Random |
| --- | --- | --- |
| LVSM (oracle) | **32.34 / 0.950 / 0.050** | **32.34 / 0.949 / 0.051** |
| PF-LRM (supervised) | 25.48 / 0.882 / 0.110 | 25.43 / 0.881 / 0.111 |
| RayZer | 31.52 / 0.945 / 0.052 | 31.42 / 0.943 / 0.053 |

**Table 4, DL3DV, continuous vs unordered training inputs.** The big one.

| | Even | Random |
| --- | --- | --- |
| (1) continuous, temporal index p.e. | 24.36 / 0.757 / 0.209 | 23.72 / 0.733 / 0.222 |
| (2) randomly shuffled during training | 20.56 / 0.576 / 0.334 | 20.02 / 0.566 / 0.356 |

A **3.8 dB** drop, and LPIPS goes 0.209 to 0.334. Figure 1 advertises "Unordered Image Sets" as
an input modality; the headline number is the ordered-video one.

**Table 5, Objaverse, 3D awareness by pose interpolation.** Render novel views by interpolating
*predicted* poses of input views, with interpolation coefficients computed from GT poses.

| Method | Even | Random |
| --- | --- | --- |
| PF-LRM | 20.63 / 0.819 / 0.160 | 21.27 / 0.827 / 0.154 |
| RayZer-copy (copy nearest rendered input view) | 19.56 / 0.812 / 0.159 | 20.17 / 0.820 / 0.150 |
| **RayZer** | **27.01 / 0.900 / 0.075** | **26.87 / 0.896 / 0.078** |

**Table 6, probing the pose space.** A 2-layer MLP trained with pose supervision on frozen `p*`,
compared against the same architecture trained supervised from scratch.

| Dataset | Pose encoder | R@10° | R@20° | R@30° | t@0.1 | t@0.2 | t@0.3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DL3DV | supervised | 39.3 | 63.0 | 77.8 | 15.7 | 33.1 | 44.4 |
| DL3DV | self-supervised | 47.6 | 72.5 | 84.0 | 20.8 | 44.0 | 60.5 |
| RealEstate | supervised | 87.0 | 96.4 | 99.6 | 44.6 | 59.3 | 82.5 |
| RealEstate | self-supervised | 99.6 | 99.9 | 100 | 61.2 | 84.2 | 92.8 |
| Objaverse | supervised | 19.8 | 46.7 | 66.0 | 15.1 | 37.2 | 53.8 |
| Objaverse | self-supervised | 33.6 | 69.2 | 86.8 | 20.1 | 52.7 | 75.5 |

Self-supervised wins every cell, but note the *absolute* values: DL3DV translation at 0.1 is
20.8%, Objaverse 20.1%. And the comparison is against a from-scratch supervised model of the
same architecture, **not** against COLMAP, DUSt3R/MASt3R, or RelPose++.

**Table 7, design ablations**, DL3DV continuous.

| | Even | Random |
| --- | --- | --- |
| (0) RayZer | 24.36 / 0.757 / 0.209 | 23.72 / 0.733 / 0.222 |
| (1) Representation: 3DGS + rasterization | **failed** (did not converge) | failed |
| (2) Prior: no Plücker ray, use SE(3) pose tokens | 22.73 / 0.687 / 0.249 | 21.88 / 0.647 / 0.274 |
| (3) Prior: no explicit pose, use latent camera tokens | 23.13 / 0.700 / 0.251 | 22.36 / 0.668 / 0.272 |
| (4) Paradigm: scene first, not pose first | 13.31 / 0.338 / 0.732 | 13.12 / 0.337 / 0.729 |

(3) + (4) combined is conceptually close to RUST.

**Table 8, appendix, video-frame training techniques.** Read the Random column carefully.

| | Even | Random |
| --- | --- | --- |
| (0) RayZer | 24.36 / 0.757 / 0.209 | 23.72 / 0.733 / 0.222 |
| (1) first frame as canonical | 23.86 / 0.736 / 0.224 | **23.78** / **0.737** / 0.225 |
| (2) no curriculum | 23.87 / 0.734 / 0.226 | **23.87** / **0.735** / 0.226 |

The appendix text says "removing any of the previously discussed techniques leads to a degraded
performance." On Random Sample, both ablations **beat** the full model on PSNR and on SSIM. Only
LPIPS favors (0). The sentence is not supported by its own table.

## The evaluation protocol, which is the crux

RayZer renders target views using **its own predicted poses**; the baselines are given COLMAP or
GT poses. The paper defends this:

> "We note that the target views are used only for pose estimation and not for scene
> representation prediction, ensuring that no information leakage occurs." (Sec. 5.1)

Two things to hold onto:

- The camera estimator takes **all K images, targets included**, as input. So per target view,
  roughly 7 scalars (6-DoF relative pose plus the shared focal), computed *from that target
  image*, reach the renderer through the Plücker map. That is small, but it is not zero.
- The paper itself concedes the leakage axis exists. On Table 7(3): "the camera tokens
  `p* ∈ R^d` can leak target image information, while `SE(3)` poses used in (2) serve as an
  information bottleneck to enforce this disentanglement."

So the honest reading is that leakage is bounded by a 7-scalar bottleneck, not absent. Meanwhile
the baselines get *noisy COLMAP* poses for the same target views. The protocol is not symmetric,
and the asymmetry runs in RayZer's favor on exactly the two datasets where it wins.

Protocol follows RUST.

## Parameter fairness

> "For fair comparisons, we use 16 transformer layers in total for GS-LRM and LVSM. Thus, their
> number of parameters is the same as RayZer, except that RayZer has another camera estimator to
> handle unposed images."

The scene + render trunk is matched at 8 + 8. RayZer carries 8 extra layers on top. At d = 768 a
transformer layer is roughly `12d² ≈ 7.1M` parameters, so about **170M (24L) vs 113M (16L)**,
call it +50%. That estimate is mine, the paper gives no parameter counts. PF-LRM uses 24 layers.

## Failure cases the paper states (Appendix C, Fig. 7)

Fine-grained geometry (plants), complicated materials (specular surfaces, stacked glasses, silver
teapots), and occlusions. The paper notes GS-LRM and LVSM fail on these too.

## Concessions the paper makes

- **Appendix D:** "RayZer's pose space jointly learns the actual pose information and 3D-aware
  video frame interpolation at the same time." On large-baseline datasets (DL3DV, Objaverse) it
  "leverages video interpolation cues together with pose estimation to perform novel view
  synthesis." That is the paper saying part of the win is interpolation, not geometry.
- **Sec 5.3:** the predicted `SE(3)` poses "do not exactly match the real-world pose space", and
  the model is robust to any warping between learned and real poses. So the poses are not metric
  and are not a drop-in SfM replacement.
- Static scenes only: "We focus on the standard setting of modeling static scenes."

## Limitations the paper does not state

- **K is fixed between train and test.** From the repo: "The model can be sensitive to the number
  of views, as it uses image index embedding. Make sure your number of views are the same during
  training and testing." Nothing in the paper says this.
- No cross-dataset generalization experiment. One model per dataset.
- Intrinsics are one shared focal length, principal point at center, no distortion. "Uncalibrated"
  is real but bounded.
- No numeric comparison to DUSt3R / MASt3R (cited as [20, 43, 76, 77]) or to COLMAP itself on
  pose accuracy.
- RUST is never compared numerically ("does not have an official public implementation"), only
  ablated conceptually.
- Table 7(1) "failed" is a single non-converged run, used to support a general claim about "the
  optimization difficulty of explicit 3D representation."

## Context worth knowing in this classroom

- **Our instructor is cited twice in this paper.** [75] IBRNet (CVPR 2021) and [76] CUT3R,
  "Continuous 3D perception model with persistent state" (arXiv 2501.12387, 2025), both
  Qianqian Wang.
- **The follow-up is co-authored by our instructor.** E-RayZer, "Self-supervised 3D
  Reconstruction as Spatial Visual Pre-training", arXiv 2512.10950 (v1 11 Dec 2025, v2
  camera-ready 28 Mar 2026). Authors: Qitao Zhao, Hao Tan, **Qianqian Wang**, Sai Bi, Kai Zhang,
  Kalyan Sunkavalli, Shubham Tulsiani, Hanwen Jiang. It moves from latent-space view synthesis to
  **explicit** 3D reconstruction specifically to "eliminate shortcut solutions", and reports that
  it "significantly outperforms RayZer on pose estimation". In other words, the authors' own next
  paper concedes that RayZer's latent formulation admits a shortcut.
- Code is released with one checkpoint (RayZer-8-12-12-100K, DL3DV) on HuggingFace, but it is a
  re-implementation; the original was trained inside Adobe. Needs compute capability above 8.0
  because of xFormers. RE10K data prep scripts are marked TODO.
