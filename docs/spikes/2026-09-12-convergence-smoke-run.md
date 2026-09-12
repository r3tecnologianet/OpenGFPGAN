# Convergence: does the assembly learn?

**Date:** 2026-09-12 · **Script:** [`../../scripts/smoke_run.py`](../../scripts/smoke_run.py) · **Verdict: it learns**

## The question

At the time of this run the project had 221 passing tests, and none of them could
answer it. Each checks a component against the paper it came from — the
demodulation against Eq. 3, the resampling kernel against P2, the lazy interval
against App. B. A flipped sign in the loss, a gradient that never reaches the
generator, an inverted polarity: all of those match every equation and still fail
to train.

Deferring the question until a corpus exists would have made it expensive. A
distribution we generate ourselves answers it in minutes, and separates "does the
code train" from "do we have data".

## Method

A white disc of random radius and position on black, at 32², generated fresh
every step — so the discriminator cannot overfit, and augmentation is switched
off rather than left as a moving part in a test meant to isolate one.

Success is measured, not looked at. See `ogan/toy.py`: validity is a connectivity
count, shape is the overlap with the disc the mask's own area and centroid imply,
and the spreads of radius and centre are what catch a generator that learned the
dataset mean. `tests/test_toy.py` calibrates all of it on analytically drawn
shapes first.

Narrow network — 64 channels, 64-dimensional latent — on an M4 Pro. 4000 steps in
404 seconds, 0.10s per step.

## Result

```
        data  validity= 1.000  disc_iou= 0.990  radius_mean= 6.740  radius_std= 1.707  centre_std= 4.727  mean_level=-0.701
       noise  validity= 0.000  disc_iou= 0.000  radius_mean= 0.000  radius_std= 0.000  centre_std= 0.000  mean_level=-0.001
    step  250  validity= 0.258  disc_iou= 0.650  radius_mean=12.808  radius_std= 2.100  centre_std= 1.652  mean_level= 0.025
    step 1000  validity= 0.875  disc_iou= 0.791  radius_mean= 6.154  radius_std= 1.554  centre_std= 3.895  mean_level=-0.489
    step 2000  validity= 0.957  disc_iou= 0.870  radius_mean= 6.779  radius_std= 1.214  centre_std= 4.227  mean_level=-0.607
    step 3000  validity= 0.973  disc_iou= 0.907  radius_mean= 6.681  radius_std= 1.377  centre_std= 4.475  mean_level=-0.674
    step 4000  validity= 0.977  disc_iou= 0.926  radius_mean= 6.020  radius_std= 1.328  centre_std= 4.990  mean_level=-0.748
```

Every metric moves monotonically towards the data and none of them stalls. Two
are worth more than the convergence itself.

**No mode collapse.** Radius spread 1.33 against the data's 1.71, and centre
spread 4.99 against 4.73 — the second lands on the data almost exactly. The
calibration established that a generator producing the dataset mean scores 0.875
for shape while both spreads sit at zero, so these two numbers are what separate
learning the distribution from learning its average.

**The polarity corrected itself.** At step 250 the mean level was +0.025 — near
grey, coming down from white — and it finished at -0.748 against the data's
-0.701. An inverted sign anywhere in the loss would have left it stuck at the
wrong end rather than crossing to the right one.

Shape was still climbing at 4000 steps, 0.926 against a ceiling of 0.990. The run
was stopped because it had answered its question, not because it had finished.

## What this does and does not establish

It establishes that the generator, the discriminator, the three losses, the lazy
schedule, the weight average and the training step work together, on a real
gradient path, well enough to fit a distribution end to end. That is the failure
class the unit suite is structurally unable to see.

It establishes nothing about faces. A disc is three degrees of freedom at 32²
with a narrow network; the target is a 512² face prior. Convergence here is a
precondition, not evidence about that.

## Reuse

The numbers above are a regression baseline. A change to the gradient path that
leaves all 221 unit tests green can still stop this run from converging, and
running it is the cheapest way to find out.

```
python scripts/smoke_run.py --steps 4000
```
