# Device viability: can Apple Silicon train this?

**Date:** 2026-09-12 · **Script:** [`device_viability.py`](device_viability.py) · **Verdict:** gate passes

## The question

StyleGAN2's R1 and path length terms each need a second derivative through the
generator, and the generator's convolutions are grouped — that is how P1 App. B
implements per-sample demodulated weights. NVIDIA's implementation ships a C++
shim, `conv2d_gradfix`, because the autograd engines of 2019 could not take a
second derivative through a grouped convolution.

A clean-room implementation cannot ship that shim. So the premise the whole
project rests on is that a modern engine no longer needs it. That is a claim to
measure, not to assert.

## Method

A two-layer proxy — affine → modulated convolution → LeakyReLU, twice — written
from the papers: P1 Eq. 1 (modulation), Eq. 2-3 (demodulation), App. B (the
grouped-convolution reshape), §3.2 (the Jacobian-vector identity). No reference
implementation was consulted; see PROVENANCE.md.

Run with `PYTORCH_ENABLE_MPS_FALLBACK=0`, so that an operation missing from the
backend fails loudly instead of falling back to the CPU and merely being slow.

## Result — Apple M4 Pro, 20-core GPU, 48 GB unified, torch 2.14.0

```
--- 1. Double-backward through a grouped convolution (the gate) ---
[PASS] R1 double-backward           — 7 param grads, all finite=True
[PASS] Path-length double-backward  — 7 param grads, all finite=True

--- 2. Batch ceiling, with the path-length second derivative ---
       256²: batches that fit = [4, 8, 16, 32], peak 2.05 GiB
       512²: batches that fit = [4, 8, 16, 32], peak 8.19 GiB
       recommended max memory = 37.44 GiB

--- 3. Reduced precision ---
         float16: double-backward ran, grads finite=False
        bfloat16: double-backward ran, grads finite=True

--- 4. Throughput, fixed workload (ch=128, fwd + bwd + Adam) ---
       128² batch 8:  238.76 img/s
       256² batch 8:   56.43 img/s
       512² batch 4:   15.97 img/s
```

Nothing raised under `MPS_FALLBACK=0`, so every operation on this path is native.

## What it decides

**The premise holds, on this backend.** Both shapes of second derivative — the R1
shape, differentiating the input gradient, and the path length shape,
differentiating the latent gradient — run and produce finite parameter gradients.
`conv2d_gradfix` is unnecessary here. The same check still has to be run on CUDA.

**Reduced precision means bfloat16, not float16.** float16 produced non-finite
gradients. An honest caveat: the proxy initialises weights `N(0, 1)` with no
equalized-learning-rate scaling, so its activations are far larger than a real
network's, and float16 overflowing under that stress is expected rather than
damning. But bfloat16 carries float32's exponent range, cost nothing here, and
removes the failure mode outright. It is the better default, and the weaker
evidence is enough to choose it.

**Memory is not the constraint on this machine.** The paper's minibatch of 32 fits
at 512² with a peak of 8.19 GiB against a 37.44 GiB ceiling — while 8.19 GiB alone
already exceeds an 8 GB discrete card in total.

## What is still open

Throughput here is a **device comparison, not a prediction**: the proxy is two
convolutions, where StyleGAN2 at 512² has roughly fifteen across eight
resolutions. The figures are meaningful only against the same script on another
device, which has not been run yet:

```
python device_viability.py cuda
```

Until that exists, any statement about how this device compares to a discrete GPU
is an estimate, and is labelled as one.

## Addendum, same day: the CUDA comparison could not be run

The two discrete cards — an RTX 3070 and an RTX 3060 Ti, alongside a GTX 1660
Super — sit in a mining rig running HiveOS. The GPUs are fine. The host is not:

```
model name : Intel(R) Celeron(R) CPU E3300 @ 2.50GHz
  sse4_2: no    avx: no    avx2: no    fma: no
```

A 2009 dual-core with no vector extensions past SSE3. `torch` installs and then
dies on import with `Illegal instruction` — the official wheels are built against
an AVX2 baseline, so the process takes a SIGILL before it reaches any GPU code.

Two consequences worth recording:

- **The rig cannot host this project**, whatever its GPUs. Building torch from
  source without AVX, on two 2009 cores and 5 GB of host RAM, is not a day's work
  and would produce an untrusted toolchain at the end of it.
- **bfloat16 would have excluded one card anyway.** The 1660 Super is Turing, and
  bfloat16 needs Ampere. Only the 3070 and 3060 Ti could have run the format §3
  selects.

So the CUDA check does not disappear, it moves: the first thing to run on a rented
instance, before any training, is this script. It costs a couple of dollars and
closes both open questions — whether the second derivative works on CUDA too, and
what the real throughput ratio is.

The design consequence is nil, because it points where the measurements already
pointed. The M4 Pro holds 48 GB of unified memory against the cards' 8 GB, fits
the paper's minibatch of 32 at 512² where they fit 2 to 4, and needs no second
machine. It was already the validation device.
