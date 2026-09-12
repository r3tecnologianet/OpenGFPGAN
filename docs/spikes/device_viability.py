"""Device viability spike: can this device train StyleGAN2 at all?

    python device_viability.py [mps|cuda|cpu]

Four questions, in order of how much they decide:

1. Does double-backward through a grouped convolution work here?  (the gate)
2. Does reduced precision survive it?
3. What batch fits at 512²?
4. What throughput does the device give on a fixed workload?

Question 1 is the clean-room premise. StyleGAN2's R1 and path length terms both
need a second derivative, and NVIDIA's implementation ships a C++ shim
(conv2d_gradfix) because the autograd engines of 2019 could not take a second
derivative through a grouped convolution. If a modern engine can, the shim is
unnecessary — and that is a claim to measure, not to assert.

Everything here is written from the papers: P1 Eq. 1 (modulation), Eq. 2-3
(demodulation), App. B (the grouped-convolution reshape), §3.2 (the
Jacobian-vector identity). See PROVENANCE.md.

The network below is a two-layer proxy, not StyleGAN2. Throughput figures are for
comparing devices with one script, not for predicting training time.
"""
import os, sys, time, math
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")   # fail loud, not slow
import torch
import torch.nn.functional as F

DEV = sys.argv[1] if len(sys.argv) > 1 else "mps"
torch.manual_seed(0)


def modulated_conv2d(x, weight, styles, demodulate=True, eps=1e-8):
    """P1 Eq. 1 (modulation), Eq. 2-3 (demodulation), App. B (grouped conv)."""
    N, C_in, H, W = x.shape
    C_out, _, kh, kw = weight.shape
    w = weight.unsqueeze(0) * styles.reshape(N, 1, C_in, 1, 1)            # Eq. 1
    if demodulate:                                                         # Eq. 2-3
        w = w * w.pow(2).sum(dim=[2, 3, 4], keepdim=True).add(eps).rsqrt()
    x = x.reshape(1, N * C_in, H, W)                                       # App. B
    w = w.reshape(N * C_out, C_in, kh, kw)
    o = F.conv2d(x, w, padding=kh // 2, groups=N)
    return o.reshape(N, C_out, *o.shape[-2:])


class Stack(torch.nn.Module):
    """A representative slice: affine -> modulated conv -> lrelu, twice."""
    def __init__(self, ch=128, w_dim=512):
        super().__init__()
        self.a1 = torch.nn.Linear(w_dim, ch)
        self.a2 = torch.nn.Linear(w_dim, ch)
        self.w1 = torch.nn.Parameter(torch.randn(ch, ch, 3, 3))
        self.w2 = torch.nn.Parameter(torch.randn(ch, ch, 3, 3))
        self.to_rgb = torch.nn.Parameter(torch.randn(3, ch, 1, 1))
        self.ch = ch

    def forward(self, x, w):
        h = F.leaky_relu(modulated_conv2d(x, self.w1, self.a1(w) + 1), 0.2)
        h = F.leaky_relu(modulated_conv2d(h, self.w2, self.a2(w) + 1), 0.2)
        return F.conv2d(h, self.to_rgb)


def dtype_check(dev, dt):
    """Does double-backward stay finite in reduced precision?"""
    try:
        net = Stack(ch=64).to(dev).to(dt)
        x = torch.randn(4, 64, 128, 128, device=dev, dtype=dt, requires_grad=True)
        w = torch.randn(4, 512, device=dev, dtype=dt)
        g = torch.autograd.grad(net(x, w).sum(), x, create_graph=True)[0]
        g.pow(2).sum().backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        finite = all(torch.isfinite(t).all().item() for t in grads)
        return f"double-backward ran, grads finite={finite}"
    except Exception as e:
        return f"FAILED — {type(e).__name__}: {str(e)[:70]}"


def memory_report(dev):
    if dev != "mps":
        return
    print(f"       torch allocated   : {torch.mps.current_allocated_memory()/2**30:6.2f} GiB")
    print(f"       recommended max   : {torch.mps.recommended_max_memory()/2**30:6.2f} GiB")


def result(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


# --- 1a. R1-shaped double-backward: grad wrt input, then backward through it ---
def test_r1(dev):
    net = Stack(ch=64).to(dev)
    x = torch.randn(4, 64, 64, 64, device=dev, requires_grad=True)
    w = torch.randn(4, 512, device=dev)
    try:
        out = net(x, w)
        g = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
        r1 = g.pow(2).sum()
        r1.backward()
    except Exception as e:
        return result("R1 double-backward", False, f"{type(e).__name__}: {e}")
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    finite = all(torch.isfinite(t).all().item() for t in grads)
    return result("R1 double-backward", finite and len(grads) > 0,
                  f"{len(grads)} param grads, all finite={finite}, r1={r1.item():.4g}")


# --- 1b. Path-length-shaped: grad wrt w with create_graph, then backward ---
def test_pl(dev):
    net = Stack(ch=64).to(dev)
    const = torch.randn(4, 64, 64, 64, device=dev)
    w = torch.randn(4, 512, device=dev, requires_grad=True)
    try:
        img = net(const, w)
        y = torch.randn_like(img) / math.sqrt(img.shape[-1] * img.shape[-2])
        s = (img * y).sum()
        gw = torch.autograd.grad(s, w, create_graph=True)[0]     # P1 §3.2 identity
        pl = (gw.norm(dim=1) - 0.0).pow(2).mean()
        pl.backward()
    except Exception as e:
        return result("Path-length double-backward", False, f"{type(e).__name__}: {e}")
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    finite = all(torch.isfinite(t).all().item() for t in grads)
    return result("Path-length double-backward", finite and len(grads) > 0,
                  f"{len(grads)} param grads, all finite={finite}, pl={pl.item():.4g}")


# --- 2. Throughput on a fixed workload ---
def sync(dev):
    if dev == "mps": torch.mps.synchronize()
    elif dev == "cuda": torch.cuda.synchronize()

def bench(dev, res, batch, ch=128, iters=8):
    net = Stack(ch=ch).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.0, 0.99))
    x = torch.randn(batch, ch, res, res, device=dev)
    w = torch.randn(batch, 512, device=dev)
    for _ in range(3):                                   # warmup
        opt.zero_grad(); net(x, w).square().mean().backward(); opt.step()
    sync(dev); t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(); net(x, w).square().mean().backward(); opt.step()
    sync(dev); dt = time.perf_counter() - t0
    return batch * iters / dt


# --- 3. Reported memory ceiling probe ---
def probe_memory(dev, res, ch=128):
    ok, peak = [], [0.0]
    for batch in (4, 8, 16, 32):
        try:
            net = Stack(ch=ch).to(dev)
            x = torch.randn(batch, ch, res, res, device=dev)
            w = torch.randn(batch, 512, device=dev, requires_grad=True)
            img = net(x, w)
            y = torch.randn_like(img)
            gw = torch.autograd.grad((img * y).sum(), w, create_graph=True)[0]
            gw.norm(dim=1).pow(2).mean().backward()      # the expensive path
            sync(dev)
            ok.append(batch)
            if dev == "mps":
                peak[0] = max(peak[0], torch.mps.current_allocated_memory() / 2**30)
            del net, x, w, img, gw
            if dev == "mps": torch.mps.empty_cache()
        except RuntimeError as e:
            print(f"       batch {batch:>2} at {res}²: FAILED ({str(e)[:70]})")
            break
    return ok, peak[0]


print(f"device={DEV}  torch={torch.__version__}  MPS_FALLBACK={os.environ['PYTORCH_ENABLE_MPS_FALLBACK']}\n")

print("--- 1. Double-backward through a grouped convolution (the gate) ---")
gate = test_r1(DEV) and test_pl(DEV)

print("\n--- 2. Batch ceiling, with the path-length second derivative ---")
for res in (256, 512):
    fits, peak = probe_memory(DEV, res)
    extra = f", peak {peak:.2f} GiB" if peak else ""
    print(f"       {res}²: batches that fit = {fits}{extra}")
if DEV == "mps":
    print(f"       recommended max memory = {torch.mps.recommended_max_memory()/2**30:.2f} GiB")

print("\n--- 3. Reduced precision ---")
for dt in (torch.float16, torch.bfloat16):
    print(f"       {str(dt).replace('torch.', ''):>9}: {dtype_check(DEV, dt)}")

print("\n--- 4. Throughput, fixed workload (ch=128, fwd + bwd + Adam) ---")
for res, batch in ((128, 8), (256, 8), (512, 4)):
    try:
        print(f"       {res}² batch {batch}: {bench(DEV, res, batch):7.2f} img/s")
    except RuntimeError as e:
        print(f"       {res}² batch {batch}: FAILED ({str(e)[:70]})")

print(f"\nGATE: {'PASS — double-backward works on ' + DEV if gate else 'FAIL'}")
