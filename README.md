# Anchor-Mediated Transport (AMT) for Credible Image Restoration

**Restricting restoration trajectories through an evidence-sufficient intermediate manifold.**

A controlled study showing that inserting a *credible intermediate* — built purely from observation-supported statistics — between the degraded observation `y` and the clean image `x` sharply reduces ungrounded hallucination, at a modest cost in PSNR.

---

## Abstract

Modern image restoration is cast as a probabilistic transport problem: a flow-matching ODE, diffusion SDE, or two-marginal Schrödinger Bridge (SB) learns a trajectory from a degraded observation `y` to a clean image `x`. Standard two-marginal SB pins only the two endpoints, leaving the intermediate trajectory free; the optimizer then takes statistically convenient shortcuts through the high-frequency texture manifold and invents details that `y` does not support.

We propose **Anchor-Mediated Transport (AMT)**: a mandatory intermediate anchor manifold `z = LowPass(y)` that retains only the SNR-rich low-frequency content of `y` and explicitly discards the noise-dominated high frequencies. Every restoration trajectory is factorized into two coupled segments, `y → z` (closed-form) and `z → x` (learned), and the second segment is trained with a low-frequency consistency term that forbids re-inventing the structure already observed in `y`.

On MNIST additive Gaussian denoising (σ = 0.5), AMT reduces the structured residual signal — our observable proxy for hallucination, measured by the residual–output correlation `Corr(y − x̂, x̂)` — by **41%** relative to an unconstrained direct black-box regressor, at the cost of only ~2.5 dB PSNR and ~1.7 points of downstream digit-classification accuracy.

---

## The idea in one picture

```
   unconstrained (vanilla regression / two-marginal SB):

        y ────────────────────────────────────────► x̂
        (black box; free to hallucinate high-freq texture)


   AMT:

        y ──► z = LowPass(y) ──► x̂
             (evidence-only,     (learned refiner,
              closed-form,        low-freq locked to z;
              near-zero           may sharpen but not
              hallucination)      re-invent structure)
```

The anchor `z` is a deterministic, closed-form function of the observation — no learned parameters, no natural-image prior. It is deliberately blurry: that *is* the point.

---

## Key results

| Method | Input | PSNR ↑ | LP-MSE ↓ | HF-MSE ↓ | Digit acc ↑ | Residual–output corr. (→0) |
|---|---|---|---|---|---|---|
| (A) Direct black-box | `y` | 20.01 | 0.0047 | 0.0053 | 96.04% | **−0.315** |
| (B) Anchor only | `z = LP(y)` | 12.39 | 0.0450 | 0.0126 | 64.11% | **+0.008** |
| (C) Naive refine | `z` | 18.76 | 0.0062 | 0.0071 | 95.09% | −0.288 |
| (D) **AMT (ours)** | `z` | 17.54 | 0.0103 | 0.0073 | 94.34% | **−0.186** |

- The rightmost column is the hallucination proxy: for a correct denoiser the residual `y − x̂` must be pure noise (correlation ≈ 0).
- The evidence-only anchor (B) sits at +0.008 — the theoretical ideal.
- The unconstrained black-box (A) wins on PSNR and accuracy, but its residual is strongly structured: it invokes prior texture that `y` does not support.
- **AMT pulls the structured residual back by 41% relative to (A), while keeping 94.3% downstream accuracy.**

---

## Method

**Segment 1 — observation → anchor (closed-form, no learning).**
A circular FFT low-pass of radius `r = 6` on 28×28 MNIST:

```python
def lowpass(img, mask):                       # img: (B,1,H,W)
    F = torch.fft.fftshift(torch.fft.fft2(img))
    return torch.fft.ifft2(torch.fft.ifftshift(F * mask)).real
```

**Segment 2 — anchor → clean (learned refiner).** A small 3-block residual CNN:

```python
def train_step(refiner, opt, x, sigma=0.5, lam=0.5):
    y = (x + sigma * torch.randn_like(x)).clamp(0, 1)
    z = lowpass(y, LP_MASK)                  # Segment 1

    x_hat = refiner(z)                       # Segment 2
    l_rec    = F.mse_loss(x_hat, x)
    l_anchor = F.mse_loss(lowpass(x_hat, LP_MASK), z)
    loss = l_rec + lam * l_anchor           # <-- the AMT regularizer

    opt.zero_grad(); loss.backward(); opt.step()
```

The `l_anchor` term locks `LP(x̂) = z`: the network may sharpen the observed low-frequency scaffold, but cannot change it.

---

## Quick start

```bash
# dependencies
pip install torch torchvision matplotlib reportlab

# run the full experiment (~15 min on CPU; ~2 min on GPU)
python run_anchor_mnist.py

# build the paper-style PDF
python build_paper_pdf.py
```

Artifacts:
- `anchor_mnist_out/summary.json` — all metrics
- `anchor_mnist_out/grid.png` — qualitative comparison
- `anchor_mnist_out/bars.png` — metric bars
- `AMT_paper.pdf` — write-up

---

## Repository structure

```
.
├── run_anchor_mnist.py        # trains (A)/(C)/(D), evaluates, saves figures
├── build_paper_pdf.py          # builds the paper-style PDF
├── build_pdf.py                # builds the shorter report PDF
├── anchor_mnist_out/
│   ├── summary.json
│   ├── grid.png
│   └── bars.png
└── AMT_paper.pdf
```

---

## Scope and non-claims

This is a **proof-of-concept**, not a production method:

- The anchor here is a *hand-designed* FFT low-pass. The intended next step — and the real link to Schrödinger Bridge / flow matching — is to replace it with a **learned intermediate marginal**: a two-segment SB whose middle marginal is constrained to be low-hallucination by construction.
- AMT trades pixel fidelity for faithfulness; it is the wrong choice for consumer-facing enhancement where plausible hallucination is desired.
- The low-/high-frequency split is natural for additive noise but less clean for blur or compression artifacts.

---

## Citation

```bibtex
@misc{amt2026,
  title  = {Anchor-Mediated Transport for Credible Image Restoration},
  author = {Zhuoran Zheng},
  year   = {2026},
  note   = {Preprint, under review}
}
```
