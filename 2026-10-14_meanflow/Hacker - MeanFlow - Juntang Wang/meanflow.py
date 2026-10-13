"""MeanFlow on 2D toys, written from Alg. 1 and Alg. 2 of Geng et al. 2025 (arXiv 2505.13447).

u(z, r, t) is the average velocity over [r, t]. The MeanFlow identity
    u = v - (t - r) * du/dt,   du/dt = v . d_z u + d_t u   (one JVP, tangent (v, 0, 1))
gives a training target that needs only v = eps - x. Setting r = t everywhere recovers Flow Matching.
"""
import math

import torch
import torch.nn as nn
from torch.func import jvp
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


def temb(t, dim=32, max_freq=1.0):
    """Sinusoidal embedding of a (B, 1) time in [0, 1].
    Keep max_freq low: du/dt goes through this, and high frequencies (DiT uses ~1000) blow up the JVP target."""
    freqs = torch.exp(torch.linspace(0, math.log(max_freq), dim // 2))
    return torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=1)


class Net(nn.Module):
    """u_theta(z, r, t). Conditioned on (t, t - r), the paper's default (Tab. 1c)."""

    def __init__(self, dim=2, width=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim + 64, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, dim),
        )

    def forward(self, z, r, t):
        return self.mlp(torch.cat([z, temb(t), temb(t - r)], dim=1))


def sample_t_r(n, ratio, mu=-2.0, sigma=2.0):
    """Two logit-normal draws, t = larger, r = smaller; r = t except on a `ratio` fraction (Tab. 1a, 1d)."""
    a, b = torch.sigmoid(mu + sigma * torch.randn(2, n, 1))
    t, r = torch.maximum(a, b), torch.minimum(a, b)
    r = torch.where(torch.rand(n, 1) < ratio, r, t)
    return t, r


def loss_fn(net, x, ratio=0.75, p=0.75, c=1e-3, tangent=(1, 0, 1), mu=-2.0, sigma=2.0):
    """Alg. 1. Defaults are the paper's CIFAR-10 setting (App. A); its ImageNet one (25%, lognorm(-0.4, 1), p = 1)
    gave a blurrier 1-step result on this toy. `tangent` scales the (v, dr, dt) JVP direction; (1, 0, 1) is correct, others are Tab. 1b ablations."""
    t, r = sample_t_r(len(x), ratio, mu, sigma)
    e = torch.randn_like(x)
    z = (1 - t) * x + t * e
    v = e - x
    kv, kr, kt = tangent
    u, dudt = jvp(net, (z, r, t), (kv * v, kr * torch.ones_like(r), kt * torch.ones_like(t)))
    u_tgt = v - (t - r) * dudt
    sq = (u - u_tgt.detach()).pow(2).sum(dim=1)
    w = 1 / (sq.detach() + c) ** p  # adaptive weight (Tab. 1e)
    return (w * sq).mean()


@torch.no_grad()
def sample(net, n, steps=1, instantaneous=False, z=None):
    """Alg. 2 generalised to `steps` jumps: z_r = z_t - (t - r) u(z_t, r, t).
    instantaneous=True queries u(z, t, t) = v instead, i.e. plain Euler on the Flow Matching field."""
    z = torch.randn(n, 2) if z is None else z.clone()
    ts = torch.linspace(1, 0, steps + 1)
    for t, r in zip(ts[:-1], ts[1:]):
        tt, rr = torch.full((n, 1), t.item()), torch.full((n, 1), r.item())
        z = z - (t - r) * net(z, tt if instantaneous else rr, tt)
    return z


def train(data_fn, ratio=0.75, steps=20000, batch=1024, lr=1e-3, seed=0, **kw):
    """Adam + cosine lr; returns the EMA copy (the paper samples from EMA weights too)."""
    torch.manual_seed(seed)
    net = Net()
    ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    losses = []
    for _ in range(steps):
        loss = loss_fn(net, data_fn(batch), ratio=ratio, **kw)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        ema.update_parameters(net)
        losses.append(loss.item())
    return ema.module, losses


def checkerboard(n):
    x1 = torch.rand(n) * 4 - 2
    x2 = torch.rand(n) - torch.randint(0, 2, (n,)) * 2 + torch.floor(x1) % 2
    return torch.stack([x1, x2], dim=1)


def moons(n, std=0.05):
    a = torch.rand(n) * math.pi
    upper = torch.rand(n) < 0.5
    x = torch.where(upper, torch.cos(a), 1 - torch.cos(a))
    y = torch.where(upper, torch.sin(a), 0.5 - torch.sin(a))
    return 1.5 * (torch.stack([x - 0.5, y - 0.25], dim=1) + std * torch.randn(n, 2))


def eight_gaussians(n, radius=2.0, std=0.15):
    angles = torch.randint(0, 8, (n,)) * (math.pi / 4)
    return radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) + std * torch.randn(n, 2)


def overlap(a, b, bins=40, lim=3.0):
    """1 - total variation between 2D histograms on [-lim, lim]^2, mass outside the box in one extra bin.
    1 = same distribution, 0 = disjoint."""
    h = []
    for x in (a, b):
        p = torch.histogramdd(x, bins=[bins, bins], range=[-lim, lim, -lim, lim]).hist.flatten() / len(x)
        h.append(torch.cat([p, (1 - p.sum()).clamp(min=0).view(1)]))
    return 1 - 0.5 * (h[0] - h[1]).abs().sum().item()


# Everything below is the same code with one knob changed. Tab. 1a / 1b of the paper.
VARIANTS = {
    "fm": dict(ratio=0.0),              # r = t always: plain Flow Matching
    "mf": dict(ratio=0.75),             # MeanFlow
    "tangent_v00": dict(tangent=(1, 0, 0)),  # drop d_t u
    "tangent_v10": dict(tangent=(1, 1, 0)),  # differentiate along r instead of t
    "tangent_v11": dict(tangent=(1, 1, 1)),  # both
}


def load_or_train(name, folder="weights"):
    path = f"{folder}/{name}.pt"
    net = Net()
    try:
        net.load_state_dict(torch.load(path))
    except FileNotFoundError:
        net, _ = train(checkerboard, **VARIANTS[name])
        torch.save(net.state_dict(), path)
    return net.eval()


if __name__ == "__main__":  # python meanflow.py mf   -> weights/mf.pt
    import os
    import sys
    import time

    os.makedirs("weights", exist_ok=True)
    t0 = time.time()
    load_or_train(sys.argv[1])
    print(sys.argv[1], f"{time.time() - t0:.0f}s")
