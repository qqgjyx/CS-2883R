# Hacker: MeanFlow (Juntang Wang)

A from-scratch MeanFlow on a 2D checkerboard, in one notebook. It compares Flow Matching and
MeanFlow at 1 step and breaks the JVP tangent as in the paper's Tab. 1b.

- `meanflow.pdf`: the executed notebook, readable without Python
- `meanflow.ipynb`: the notebook (about 40 lines of method code)
- `figs/`: the four figures
- `weights/`: the five trained models, so the notebook runs in about 20 s

To run it:

```
uv run --with torch --with matplotlib --with jupyter jupyter lab meanflow.ipynb
```

Delete a file in `weights/` to retrain that model, about 2.5 min on an M3 Pro.

Credits: the method, pseudocode and settings are from Geng, Deng, Bai, Kolter, He, "Mean Flows for
One-step Generative Modeling" (arXiv 2505.13447). I wrote the code. I checked `loss_fn` against
the first author's PyTorch release, https://github.com/Gsunshine/py-meanflow
(`forward_with_loss`): the loss and gradients are identical on the same inputs.
