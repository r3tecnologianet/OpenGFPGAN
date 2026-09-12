"""Train on the toy distribution and report whether it learned it.

    python scripts/smoke_run.py [--steps N] [--device mps|cuda|cpu]

The unit suite checks that each component matches its paper. This checks that the
assembly learns, which is a different question and not implied by the first.

Augmentation is off. The data is generated fresh every step, so the discriminator
cannot overfit and there is nothing for the adaptive controller to respond to —
leaving it on would add a moving part to a test whose purpose is isolating one.
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from ogan.toy import draw_discs, measure_discs  # noqa: E402
from ogan.training import Trainer, TrainingConfig  # noqa: E402

REPORTED = ("validity", "disc_iou", "radius_mean", "radius_std", "centre_std", "mean_level")


def report(label: str, metrics: dict[str, float]) -> None:
    body = "  ".join(f"{key}={metrics[key]:6.3f}" for key in REPORTED)
    print(f"{label:>12}  {body}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    trainer = Trainer(
        TrainingConfig(
            resolution=args.resolution,
            batch_size=args.batch,
            channel_max=args.channels,
            z_dim=args.latent,
            w_dim=args.latent,
            augment=False,
        ),
        device=device,
    )

    ceiling = measure_discs(draw_discs(256, args.resolution).images)
    print(f"device={device}  steps={args.steps}  resolution={args.resolution}²")
    report("data", ceiling)
    report("noise", measure_discs(torch.randn(256, 3, args.resolution, args.resolution)))
    print("-" * 78, flush=True)

    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        real = draw_discs(args.batch, args.resolution, device=device).images
        trainer.step(real)

        if step % args.every == 0 or step == args.steps:
            with torch.no_grad():
                z = torch.randn(256, args.latent, device=device)
                samples = trainer.averaged(z).cpu()
            report(f"step {step}", measure_discs(samples))

    elapsed = time.perf_counter() - started
    print("-" * 78)
    print(f"{args.steps} steps in {elapsed:.0f}s  ({elapsed / args.steps:.2f}s/step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
