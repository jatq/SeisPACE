# SeisPACE-2D

SeisPACE-2D uses a same-data SCRN model as the direct baseline and the 2D
dual-sparse prior as the frozen auxiliary model.

## Protocol

1. Train SCRN on noisy-clean sections.
2. Consolidate SCRN without the prior to obtain `B*`.
3. Build a two-channel SCRN-style calibrator from `B*` and zero its new prior
   channel.
4. Fine-tune with the task loss and the frozen anchor.

```text
L = MSE(R(y, P(y)), s)
    + lambda_anchor MSE(R(y, P(y)), B*(y))
```

The selected anchor weight is `lambda_anchor = 0.5`. The 256 x 128 patches use
a whole-section normalization after 0.5/99.5 percentile clipping.

## Main result

Mean over 100 held-out sections, reported as achieved SNR in dB:

| SNR | SCRN baseline | No-prior continuation | SeisPACE-2D |
|---|---:|---:|---:|
| -10 dB | 15.81 | 16.66 | **16.67** |
| 0 dB | 22.39 | 23.25 | **23.34** |
| 10 dB | 26.24 | 27.11 | **27.16** |

## Commands

```bash
export SEISPACE_DATA_ROOT=/path/to/seispace/data
export SEISPACE_SCRN_ROOT=/path/to/SCRN

python scripts/prepare_seispace2d_data.py \
  --config configs/prior2d_wavelet.json \
  --checkpoint runs/prior2d_wavelet/best.pt \
  --mat-file "$SEISPACE_DATA_ROOT/simulate_wavelet/training_data.mat" \
  --output-dir data/seispace2d

python scripts/train_seispace2d.py \
  --scrn-root "$SEISPACE_SCRN_ROOT" \
  --data-dir data/seispace2d \
  --output-dir runs/seispace2d_baseline \
  --input-mode noisy --epochs 80

python scripts/train_seispace2d.py \
  --scrn-root "$SEISPACE_SCRN_ROOT" \
  --data-dir data/seispace2d \
  --output-dir runs/seispace2d_consolidated \
  --input-mode noisy \
  --init-checkpoint runs/seispace2d_baseline/best.pt \
  --epochs 80

python scripts/train_seispace2d.py \
  --scrn-root "$SEISPACE_SCRN_ROOT" \
  --data-dir data/seispace2d \
  --output-dir runs/seispace2d_main \
  --input-mode noisy_prior \
  --init-checkpoint runs/seispace2d_consolidated/best.pt \
  --anchor-checkpoint runs/seispace2d_consolidated/best.pt \
  --anchor-weight 0.5 --epochs 20

python scripts/evaluate_seispace2d.py \
  --scrn-root "$SEISPACE_SCRN_ROOT" \
  --prior-config configs/prior2d_wavelet.json \
  --prior-checkpoint runs/prior2d_wavelet/best.pt \
  --refiner-checkpoint runs/seispace2d_main/best.pt \
  --test-mat "$SEISPACE_DATA_ROOT/simulate_wavelet/testing_data.mat" \
  --output runs/eval_seispace2d/0db.json \
  --input-mode noisy_prior --num-samples 100 --snr-db 0
```
