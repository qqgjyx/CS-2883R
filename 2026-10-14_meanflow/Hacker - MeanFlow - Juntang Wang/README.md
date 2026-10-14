# Hacker: MeanFlow (Juntang Wang)

Takeaway: **learn the average, not the instant.** MeanFlow predicts the average velocity over a time
interval, so one jump turns noise into a sample, and one JVP makes that average trainable.

- `Hacker - MeanFlow - Juntang Wang.pdf`: the slides
- `meanflow.ipynb`: the demo. Open in Colab and click Run all; it retrains while you watch (about 3 min on a free T4)

Colab: https://colab.research.google.com/github/qqgjyx/CS-2883R/blob/main/2026-10-14_meanflow/meanflow.ipynb

What it shows, on a checkerboard, a Swiss roll, MNIST and Fashion-MNIST:

1. MeanFlow in 1 step against Flow Matching at 1, 2, 4 and 50 steps.
2. The 1-step jump lands where the 100-step path ends (same network, same noise).
3. Breaking the JVP tangent breaks the jump, in the paper's order (Tab. 1b).

Credits:

- Method: Geng, Deng, Bai, Kolter, He, *Mean Flows for One-step Generative Modeling*, arXiv 2505.13447.
  Fig. 3 on slide 2 is theirs (CC BY 4.0).
- Code: adapted from https://github.com/haidog-yaqub/MeanFlow (MIT). We switched the JVP to
  forward mode (`torch.func.jvp`) and added a tangent option for the ablation.
- ImageNet cat on the last slide: MF-XL/2 checkpoint from https://github.com/zhuyu-cs/MeanFlow, one step.
- Toy data: FFJORD (Grathwohl et al., 2019) and the Swiss roll of Sohl-Dickstein et al. (2015).
