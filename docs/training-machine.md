# The training machine

What a machine needs to train this project's generator, and what to check before
trusting it. Written after a first attempt failed on a detail nobody checks.

## The trap that already cost us once

**The CPU must support AVX2.** Not the GPU — the CPU.

The first attempt used a mining rig with capable GPUs and a Celeron E3300, a 2009
part with no SSE4.2, AVX, AVX2 or FMA. `import torch` died with SIGILL before any
CUDA code ran. The GPUs were irrelevant; the host could not load the library.

Check before buying or assembling anything:

```sh
# Linux
grep -o 'avx2' /proc/cpuinfo | head -1        # must print avx2
# then, the real test
python -c "import torch; print(torch.__version__)"
```

Any Intel from Haswell (2013) or any AMD from Zen (2017) onward is fine. The
failure only happens on genuinely old hosts, which is exactly what mining rigs
tend to be.

## GPU

Either of the cards on hand works. Neither is comfortable.

| | RTX 3070 | RTX 3060 Ti |
|---|---|---|
| VRAM | 8 GB | 8 GB |
| Compute capability | 8.6 | 8.6 |

**8 GB is the binding constraint, not compute.** StyleGAN2 at 512² holds
activations for nine resolution blocks in both networks, and the R1 and
path-length terms each need a second derivative, which means keeping the forward
graph alive through a backward pass. Expect a per-step batch of 4 to 8, not the
32 the papers use.

That is workable — gradient accumulation reaches the paper's effective batch at
the cost of wall time — but `ogan/training.py` does not implement accumulation
today. It is a small change and it is not written yet.

**More VRAM is worth more than more cards here.** A single 16 GB or 24 GB card
would beat both 8 GB cards together, for two reasons given below.

## Two GPUs will not help yet

`ogan/` has no `DataParallel`, no `DistributedDataParallel`, and no device
sharding. Training runs on one device. A second card sits idle.

Adding distributed training is real work and is not on the path to a first
result. Plan on one card doing the job, and treat the second as a spare.

## RAM and storage

| | Minimum | Comfortable |
|---|---|---|
| System RAM | 16 GB | 32 GB |
| Corpus storage | 100 GB | 500 GB |
| Checkpoint storage | 50 GB | 200 GB |

**The corpus number depends on a decision worth making early.** PD12M source
images average a few megabytes; 500,000 of them is on the order of a terabyte.
Storing them is unnecessary — the pipeline should download an image, crop the
face to 512², write the crop, and discard the original. 500,000 crops at 512² is
roughly 50 GB, not 1 TB.

Checkpoints are about 1 GB each: generator, discriminator and the averaged
generator, plus two Adam moments for each of the trained pair. Keep several.

## Software

| | Version |
|---|---|
| OS | Linux. Ubuntu 22.04 or 24.04 LTS is the least surprising choice |
| NVIDIA driver | 550 or newer |
| CUDA | supplied by the PyTorch wheel; no separate toolkit install needed |
| Python | 3.11 — pinned in `.python-version` |
| PyTorch | 2.14.0 — pinned in `pyproject.toml` |

Setup is the same as on any other machine:

```sh
git clone <this repo> && cd OpenGFPGAN
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest          # expect 228 passed
```

The `corpus` extra is macOS-only — it pulls `pyobjc-framework-Vision`. Do not
install it on Linux. Corpus building happens on the Mac; training happens here.

## Run this first

Before trusting the machine, run the viability spike. It takes a minute and it
closes two questions that are still open on CUDA:

```sh
.venv/bin/python docs/spikes/device_viability.py cuda
```

It checks four things in order of how much they decide:

1. **Double-backward through a grouped convolution.** This is the gate. The whole
   clean-room premise is that a modern autograd engine no longer needs NVIDIA's
   `conv2d_gradfix` shim. Verified on MPS; **not yet verified on CUDA.** If this
   fails, training cannot proceed and the project has a real problem to solve.
2. **Reduced precision.** On MPS, `float16` produced non-finite gradients through
   the double-backward and `bfloat16` did not. Ampere supports bfloat16 natively,
   so expect the same answer, but confirm it — on 8 GB it is the difference
   between a batch of 4 and a batch of 8.
3. **Batch ceiling at 512².**
4. **Throughput**, for comparison against the M4 Pro baseline below.

Record the output in `docs/spikes/` alongside the existing Apple Silicon run.

## The baseline to beat

Apple M4 Pro, 20-core GPU, 48 GB unified, on a two-layer proxy rather than the
full network:

```
512²  batch 4:  15.97 img/s
256²  batch 8:  56.43 img/s
128²  batch 8: 238.76 img/s
512²: batches that fit = [4, 8, 16, 32], peak 8.19 GiB
```

Those are proxy figures for comparing devices, not predictions of training time.
The full network is far deeper and will be much slower.

## What this does not tell you

**How long training takes.** That number does not exist yet, and guessing it
would be worse than leaving it open. It is the product of the throughput this
machine actually delivers and the number of images the generator needs to see,
and the first factor is unmeasured until the spike above is run on this hardware.

Expect it to be measured in weeks on a single 8 GB card, and plan the machine
around running unattended for that long: no sleep, no automatic reboots, and
checkpoints written often enough that losing a day does not hurt.
