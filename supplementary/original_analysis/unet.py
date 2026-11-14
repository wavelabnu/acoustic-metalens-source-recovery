# -*- coding: utf-8 -*-
"""
unet.py
=======
The optimized 1D U-Net reconstructor + trainer.

Architecture (faithful in spirit to the lab's Wolfram NetTrain U-Net): double
(Conv1d + BatchNorm + ReLU) blocks, max-pool encoder, transposed-conv decoder,
channel-catenation skips, then Flatten -> 512 -> the 2*Ns drive vector. Generalized
to ARBITRARY depth so widths=(64,128,256,512,1024) (5 levels) works as well as the
original 4.

Optimizations over the old fixed-2000-round / batch-64 / fp32 NetTrain pipeline:
  * early stopping on a validation split (no wasted GPU rounds),
  * large batch (default 512) + DataLoader(pin_memory) instead of manual minibatch .to(),
  * mixed precision (autocast + GradScaler) on CUDA  [torch.cuda.amp, modernized to the
    torch.amp API when available so it is warning-free on torch>=2.7 / RTX 50-series],
  * train tensors are built ONCE (caller caches them; see data.make_dataset),
  * configurable loss (mse / l1 / huber / rel_l2) for the loss sweep,
  * checkpoint save/load for the parametrized sweeps.

I/O:  input  x (B, 2*Nmic, Nfreq) real ;  output (B, 2*Nspeaker) real = [Re a ; Im a].
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----- AMP compatibility (new torch.amp API, fall back to torch.cuda.amp) ---
def _autocast(enabled):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def _grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ----- architecture --------------------------------------------------------
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv1d(cin, cout, 3, padding=1), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
        nn.Conv1d(cout, cout, 3, padding=1), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
    )


class UNet1D(nn.Module):
    """Depth-generic 1D U-Net. len(widths) encoder stages, len(widths)-1 pools; the
    input length must be divisible by 2**(len(widths)-1)."""

    def __init__(self, cin, length, nout, widths=(64, 128, 256, 512, 1024)):
        super().__init__()
        pools = len(widths) - 1
        if length % (2 ** pools) != 0:
            raise ValueError(f"length={length} must be divisible by "
                             f"2**(len(widths)-1)={2 ** pools} for {len(widths)} levels")
        self.enc = nn.ModuleList()
        prev = cin
        for w in widths[:-1]:
            self.enc.append(conv_block(prev, w)); prev = w
        self.bott = conv_block(prev, widths[-1])
        self.pool = nn.MaxPool1d(2)
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        prev = widths[-1]
        for w in reversed(widths[:-1]):
            self.up.append(nn.ConvTranspose1d(prev, w, 2, stride=2))
            self.dec.append(conv_block(2 * w, w)); prev = w
        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(widths[0] * length, 512), nn.ReLU(inplace=True),
                                  nn.Linear(512, nout))

    def forward(self, x):
        skips = []
        for enc in self.enc:
            x = enc(x); skips.append(x); x = self.pool(x)
        x = self.bott(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], 1))
        return self.head(x)


# ----- loss zoo (for the loss sweep) ---------------------------------------
def make_loss(name="mse"):
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name in ("huber", "smoothl1"):
        return nn.SmoothL1Loss()
    if name == "rel_l2":
        def _rel(pred, tgt):
            num = torch.linalg.vector_norm(pred - tgt, dim=-1)
            den = torch.linalg.vector_norm(tgt, dim=-1) + 1e-8
            return (num / den).mean()
        return _rel
    raise ValueError(f"unknown loss '{name}'")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ----- training ------------------------------------------------------------
def train_unet(model, Xtr, Ytr, Xva, Yva, *, epochs=200, bs=512, lr=1.5e-3,
               patience=12, loss="mse", amp=True, num_workers=0, device=DEV,
               ckpt_path=None, verbose=True):
    """Train with early stopping + AMP + pinned DataLoader. Returns
    dict(model, best_val, history). Restores the best-val weights before returning."""
    model = model.to(device)
    use_amp = bool(amp) and str(device).startswith("cuda")
    pin = str(device).startswith("cuda")

    train_dl = DataLoader(
        TensorDataset(torch.as_tensor(Xtr), torch.as_tensor(Ytr)),
        batch_size=bs, shuffle=True, pin_memory=pin, num_workers=num_workers, drop_last=False)
    Xva_t = torch.as_tensor(Xva).to(device)
    Yva_t = torch.as_tensor(Yva).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = _grad_scaler(use_amp)
    lossf = make_loss(loss)

    best, best_state, bad, history = np.inf, None, 0, []
    for ep in range(epochs):
        t0 = time.time(); model.train()
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=pin)
            yb = yb.to(device, non_blocking=pin)
            opt.zero_grad(set_to_none=True)
            with _autocast(use_amp):
                l = lossf(model(xb), yb)
            scaler.scale(l).backward()
            scaler.step(opt)
            scaler.update()
        model.eval()
        with torch.no_grad(), _autocast(use_amp):
            vl = float(lossf(model(Xva_t), Yva_t))
        history.append(vl)
        if verbose:
            print(f"    epoch {ep:03d}  val[{loss}] {vl:.5f}  ({time.time() - t0:.1f}s)", flush=True)
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if ckpt_path:
                torch.save({"state_dict": best_state, "val": best, "epoch": ep,
                            "loss": loss}, ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stop @ epoch {ep} (no val gain in {patience})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return dict(model=model, best_val=best, history=history)


def load_checkpoint(model, ckpt_path, device=DEV):
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["state_dict"])
    return model.to(device)
