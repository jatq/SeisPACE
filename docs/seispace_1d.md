# SeisPACE-1D

SeisPACE-1D uses a temporal U-Net as the direct baseline and the 1D
dual-sparse prior as the frozen auxiliary model.

## Protocol

1. Train the direct baseline `B` from scratch.
2. Consolidate `B` on paired noisy-clean traces without the prior to obtain
   `B*`.
3. Initialize a two-channel calibrator from `B*`.
4. Zero the weights of the new prior channel, so `R_0(y, 0) = B*(y)`.
5. Fine-tune the calibrator with the task loss and the frozen anchor.

The anchor objective is

```text
L = MSE(R(y, P(y)), s)
    + lambda_anchor MSE(R(y, P(y)), B*(y)).
```

The selected anchor weight is `lambda_anchor = 0.5`. At inference, the prior is
run once and the calibrator is run once; `B*` is not executed.

## Main result

| SNR | Learned prior only | Direct baseline | No-prior continuation | SeisPACE-1D |
|---|---:|---:|---:|---:|
| -10 dB | 25.74 | 31.28 | 32.66 | **33.20** |
| 0 dB | 26.88 | 28.83 | 30.72 | **31.52** |
| 10 dB | 23.77 | 23.38 | 25.37 | **26.45** |
| Mean over 11 SNRs | 26.03 | 28.28 | 30.07 | **30.91** |

Values are SNR gains in dB on 200 held-out traces.

## Commands

```bash
export SEISPACE_DATA_ROOT=/path/to/seispace/data

python scripts/prepare_seispace1d_data.py \
  --config configs/prior1d_wavelet_paper.json \
  --checkpoint runs/prior1d_wavelet/best.pt \
  --mat-file "$SEISPACE_DATA_ROOT/simulate_wavelet/training_data.mat" \
  --output-dir data/seispace1d

python scripts/train_seispace1d.py \
  --mode baseline \
  --config configs/seispace1d_calibrator.json \
  --data-dir data/seispace1d \
  --output-dir runs/seispace1d_baseline \
  --epochs 100

python scripts/train_seispace1d.py \
  --mode baseline \
  --config configs/seispace1d_calibrator.json \
  --data-dir data/seispace1d \
  --output-dir runs/seispace1d_consolidated \
  --init-checkpoint runs/seispace1d_baseline/best.pt \
  --epochs 80

python scripts/train_seispace1d.py \
  --mode calibrate \
  --config configs/seispace1d_calibrator.json \
  --data-dir data/seispace1d \
  --output-dir runs/seispace1d_main \
  --baseline-checkpoint runs/seispace1d_consolidated/best.pt \
  --anchor-weight 0.5 \
  --epochs 20

python scripts/evaluate_seispace1d.py \
  --prior-config configs/prior1d_wavelet_paper.json \
  --prior-checkpoint runs/prior1d_wavelet/best.pt \
  --refiner-checkpoint runs/seispace1d_main/best.pt \
  --test-mat "$SEISPACE_DATA_ROOT/simulate_wavelet/testing_data.mat" \
  --output runs/eval_seispace1d/0db.json \
  --direct-refiner --snr-db 0
```
