"""
Anchor-Mediated Image Restoration on MNIST
==========================================
Idea under test:
  Instead of learning a black-box map  y -> x  (which is free to hallucinate
  high-frequency detail), insert a CREDIBLE INTERMEDIATE z = LowPass(y):
  keep only the SNR-rich low-frequency components of the observation and
  discard the noise-dominated high frequencies. Then refine z -> x under a
  low-frequency consistency constraint that forbids the network from
  re-inventing the already-observed structure.

Methods compared:
  (A) Direct black-box:        ResNet(y) -> x, trained with MSE only.
  (B) Anchor only (no net):     z = LowPass(y).
  (C) Naive anchor refine:     ResNet(z) -> x, MSE only.
  (D) Ours (anchor-mediated):  ResNet(z) -> x, MSE + lambda*||LP(x_hat)-z||^2.
"""
import os, time, math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(0)
np.random.seed(0)
DEVICE = "cpu"
OUT = "/home/user/Doubao/chats/38451616413142530/anchor_mnist_out"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Data + degradation
# ----------------------------------------------------------------------------
IMG = 28
SIGMA = 0.5                      # noise std (strong enough to force hallucination)
LP_RADIUS = 6                    # FFT low-pass radius (SNR-rich anchor)

tf = transforms.ToTensor()
train_set = datasets.MNIST(OUT + "/data", train=True,  download=True, transform=tf)
test_set  = datasets.MNIST(OUT + "/data", train=False, download=True, transform=tf)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=2)

def make_lp_mask(r):
    m = torch.zeros(1, 1, IMG, IMG)
    c = IMG // 2
    for i in range(IMG):
        for j in range(IMG):
            if (i - c) ** 2 + (j - c) ** 2 <= r * r:
                m[0, 0, i, j] = 1.0
    return m

LP_MASK = make_lp_mask(LP_RADIUS)

def lowpass(x, mask=LP_MASK):
    X = torch.fft.fftshift(torch.fft.fft2(x))
    X = X * mask
    return torch.fft.ifft2(torch.fft.ifftshift(X)).real

def degrade(x):
    return (x + SIGMA * torch.randn_like(x)).clamp(0, 1)

# ----------------------------------------------------------------------------
# 2. Networks
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1))
    def forward(self, x):
        return x + self.net(x)

class Denoiser(nn.Module):
    def __init__(self, in_ch=1, ch=32):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(in_ch, ch, 3, padding=1), nn.ReLU(inplace=True))
        self.body = nn.Sequential(Block(ch), Block(ch), Block(ch))
        self.tail = nn.Conv2d(ch, 1, 3, padding=1)
    def forward(self, x):
        return self.tail(self.body(self.head(x)))

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.fc = nn.Linear(64 * 7 * 7, 10)
    def forward(self, x):
        h = self.feat(x).flatten(1)
        return self.fc(h)

# ----------------------------------------------------------------------------
# 3. Pretrain a frozen classifier on CLEAN digits (downstream utility probe)
# ----------------------------------------------------------------------------
print("[*] pretraining classifier on clean MNIST ...")
clf = Classifier().to(DEVICE)
opt = torch.optim.Adam(clf.parameters(), 1e-3)
for ep in range(2):
    clf.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = F.cross_entropy(clf(x), y)
        loss.backward(); opt.step()
clf.eval()
for p in clf.parameters():
    p.requires_grad_(False)
# classifier accuracy on clean test
correct = tot = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (clf(x).argmax(1) == y).sum().item(); tot += y.numel()
print(f"    clean-test classifier acc = {correct/tot:.4f}")

# ----------------------------------------------------------------------------
# 4. Train the four restoration methods
# ----------------------------------------------------------------------------
def train_denoiser(model, input_kind, epochs, lam=0.0):
    """input_kind: 'y' (noisy) or 'z' (anchor).
       lam>0 adds low-frequency consistency to the anchor."""
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    model.train()
    t0 = time.time()
    for ep in range(epochs):
        for x, _ in train_loader:
            x = x.to(DEVICE)
            y = degrade(x)
            z = lowpass(y)
            inp = y if input_kind == "y" else z
            opt.zero_grad()
            out = model(inp)
            loss = F.mse_loss(out, x)
            if lam > 0:
                loss = loss + lam * F.mse_loss(lowpass(out), z)
            loss.backward(); opt.step()
    return time.time() - t0

EPOCHS = 4
models = {}
print("[*] training (A) direct black-box y->x ...")
m = Denoiser(1).to(DEVICE); t = train_denoiser(m, "y", EPOCHS); models["A_direct"] = m
print(f"    done in {t:.1f}s")

print("[*] training (C) naive anchor refine z->x (MSE only) ...")
m = Denoiser(1).to(DEVICE); t = train_denoiser(m, "z", EPOCHS); models["C_naive"] = m
print(f"    done in {t:.1f}s")

print("[*] training (D) ours: anchor-mediated z->x + LP consistency ...")
m = Denoiser(1).to(DEVICE); t = train_denoiser(m, "z", EPOCHS, lam=0.5); models["D_ours"] = m
print(f"    done in {t:.1f}s")

# ----------------------------------------------------------------------------
# 5. Evaluate on test set
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict(method, y, z):
    if method in ("anchor_only", "B_anchor"):
        return z
    inp = {"A_direct": y, "C_naive": z, "D_ours": z}[method]
    return models[method](inp)

results = {k: dict(psnr=[], lp_mse=[], hf_mse=[], acc=[], residual_corr=[])
           for k in ["A_direct", "B_anchor", "C_naive", "D_ours"]}

# stash a few samples for the qualitative grid
GRID = {"y": [], "z": [], "x": [], "A": [], "C": [], "D": []}
n_grid = 8

for bi, (x, label) in enumerate(test_loader):
    x = x.to(DEVICE)
    y = degrade(x)
    z = lowpass(y)
    for key, mname in [("A", "A_direct"), ("B", "B_anchor"),
                       ("C", "C_naive"), ("D", "D_ours")]:
        out = predict(mname, y, z).clamp(0, 1)
        mse = F.mse_loss(out, x).item()
        lp_out, lp_x = lowpass(out), lowpass(x)
        hf_out, hf_x = out - lp_out, x - lp_x
        results[mname]["psnr"].append(10 * math.log10(1.0 / max(mse, 1e-10)))
        results[mname]["lp_mse"].append(F.mse_loss(lp_out, lp_x).item())
        results[mname]["hf_mse"].append(F.mse_loss(hf_out, hf_x).item())
        results[mname]["acc"].append((clf(out).argmax(1).cpu().numpy()))
        # residual-output correlation (whiteness): residual = y - out
        r = (y - out).flatten(1)
        o = out.flatten(1)
        r = r - r.mean(1, keepdim=True); o = o - o.mean(1, keepdim=True)
        corr = (r * o).sum(1) / (r.norm(dim=1) * o.norm(dim=1) + 1e-8)
        results[mname]["residual_corr"].append(corr.cpu().numpy())
    if bi == 0:
        idx = torch.randperm(x.shape[0])[:n_grid]
        GRID["y"].append(y[idx].cpu()); GRID["z"].append(z[idx].cpu())
        GRID["x"].append(x[idx].cpu())
        for kk, mm in [("A", "A_direct"), ("C", "C_naive"), ("D", "D_ours")]:
            GRID[kk].append(predict(mm, y, z).clamp(0, 1)[idx].cpu())

# aggregate: align predictions with true labels
all_labels = torch.cat([l for _, l in test_loader]).numpy()
summary = {}
for k, v in results.items():
    pred = np.concatenate(v["acc"])
    summary[k] = dict(
        psnr=float(np.mean(v["psnr"])),
        lp_mse=float(np.mean(v["lp_mse"])),
        hf_mse=float(np.mean(v["hf_mse"])),
        acc=float((pred == all_labels).mean()),
        residual_corr=float(np.mean(np.concatenate(v["residual_corr"]))),
    )

import json
with open(OUT + "/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

# ----------------------------------------------------------------------------
# 6. Save qualitative grid
# ----------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = ["clean x", "noisy y", "anchor z", "(A) direct", "(C) naive", "(D) ours"]
fig, axes = plt.subplots(len(rows), n_grid, figsize=(1.6 * n_grid, 1.7 * len(rows)))
data = {
    "clean x":  torch.cat(GRID["x"])[ :n_grid],
    "noisy y":  torch.cat(GRID["y"])[ :n_grid],
    "anchor z": torch.cat(GRID["z"])[ :n_grid],
    "(A) direct": torch.cat(GRID["A"])[ :n_grid],
    "(C) naive":  torch.cat(GRID["C"])[ :n_grid],
    "(D) ours":    torch.cat(GRID["D"])[ :n_grid],
}
for r, name in enumerate(rows):
    for c in range(n_grid):
        ax = axes[r, c]
        ax.imshow(data[name][c, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(name, fontsize=9)
plt.tight_layout()
plt.savefig(OUT + "/grid.png", dpi=130, bbox_inches="tight")
plt.close()

# bar chart of metrics
fig, axs = plt.subplots(1, 4, figsize=(15, 3.2))
names = ["A_direct", "B_anchor", "C_naive", "D_ours"]
labels = ["(A)\ndirect", "(B)\nanchor only", "(C)\nnaive refine", "(D)\nours"]
colors = ["#888", "#bbb", "#d9a066", "#4c72b0"]
for ax, key, title, better in [
    (axs[0], "psnr", "PSNR (dB)  higher=better", "hi"),
    (axs[1], "lp_mse", "Low-freq MSE  lower=better", "lo"),
    (axs[2], "hf_mse", "High-freq MSE  lower=better", "lo"),
    (axs[3], "acc", "Downstream digit acc  higher=better", "hi"),
]:
    vals = [summary[n][key] for n in names]
    ax.bar(labels, vals, color=colors)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(OUT + "/bars.png", dpi=130, bbox_inches="tight")
plt.close()

print("[*] artifacts in", OUT)
#（注：内容由AI生成）
