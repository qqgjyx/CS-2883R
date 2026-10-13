# Hacker: MeanFlow (Juntang Wang)

A from-scratch toy MeanFlow on a 2D checkerboard. It compares Flow Matching and MeanFlow at 1
step and breaks the JVP tangent as in the paper's Tab. 1b.

- `demo.pdf` / `demo.html`: the executed notebook, readable without Python
- `demo.ipynb`: the notebook itself
- `meanflow.py`: all of the code (model, Alg. 1 loss, sampler, data, metric)
- `weights/`: the five trained models, so the notebook runs in seconds

Run it:

```
uv run --with torch --with matplotlib --with jupyter jupyter lab demo.ipynb
```

Delete a file in `weights/` to retrain that model, about 10 minutes on a laptop CPU.

Credits: the method and pseudocode are from Geng, Deng, Bai, Kolter, He, "Mean Flows for One-step
Generative Modeling" (arXiv 2505.13447), Alg. 1 and 2. I wrote the code from scratch in PyTorch
and used no existing MeanFlow implementation.
