# SeisPACE

Official research code for

**SeisPACE: Two-Stage Baseline-Anchored Calibration with a Lightweight
Dual-Sparse Prior for Seismic Denoising**

Repository: <https://github.com/jatq/SeisPACE>

SeisPACE combines two components:

1. A lightweight dual-sparse seismic prior built from an orthogonal DCT,
   learnable analysis/synthesis dictionaries, and a shallow context denoising
   autoencoder.
2. A baseline-anchored calibration stage that freezes a consolidated denoiser
   and the prior, initializes the prior input channel to zero, and trains one
   two-channel calibrator with a task loss and a baseline anchor.

Inference evaluates the prior once and the calibrator once:

```text
prior = P(noisy)
output = R(noisy, prior)
```

The consolidated baseline is used only during training and is not run at
inference. No model ensemble is used.

## Repository layout

```text
src/seispace/                 Core Python package
  prior1d.py                  Dual-sparse prior for 1D traces
  prior2d.py                  Dual-sparse prior for 2D sections
  calibrator1d.py             1D SeisPACE calibrator
  prior_control.py            Correct/zero/shuffled prior controls
  baselines/                  DeepDenoiser, DeepSeg, and MSMHA models
scripts/                      Training, evaluation, preprocessing, and plotting
configs/                      Reproducible experiment configurations
tests/                        CPU-side unit and smoke tests
docs/                         Data provenance and protocol documentation
```

Data, checkpoints, and experiment outputs are not stored in Git.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

The 2D calibrator uses the official SCRN implementation as a backbone. Fetch
the public baseline repositories into the ignored `third_party/` directory:

```bash
bash scripts/prepare_external_baselines.sh
```

Set the paths used by the experiment scripts:

```bash
export SEISPACE_ROOT="$(pwd)"
export SEISPACE_DATA_ROOT=/path/to/seispace/data
export SEISPACE_SCRN_ROOT="$SEISPACE_ROOT/third_party/SCRN"
```

On Windows PowerShell:

```powershell
$env:SEISPACE_ROOT = (Get-Location).Path
$env:SEISPACE_DATA_ROOT = "D:\path\to\seispace\data"
$env:SEISPACE_SCRN_ROOT = "$env:SEISPACE_ROOT\third_party\SCRN"
```

## Data layout

The code accepts arbitrary data roots. The expected default layout is:

```text
$SEISPACE_DATA_ROOT/
  simulate_wavelet/
    training_data.mat
    testing_data.mat
  real_seismic_trace/
    high_60dB_HH.hdf5
    high_60dB_HH.csv
    noise_HH.hdf5
    low_HH.hdf5
    low_HH.csv
  native_forge2d/
  native_marmousi2/
```

The controlled wavelet data support the main 1D and synthetic 2D benchmarks.
FORGE is a native DAS field record with controlled STEAD noise injection.
Marmousi2 is a public synthetic SEGY section, not a field recording.

See:

- [docs/native2d_field.md](docs/native2d_field.md)
- [docs/marmousi2_source.md](docs/marmousi2_source.md)
- [docs/seispace_experiment_plan.md](docs/seispace_experiment_plan.md)

## Train the 1D prior

The paper prior uses a 256-sample DCT, 256-dimensional embedding, 15 context
segments, and a coefficient-domain MAE objective with inverse-consistency
regularization.

```bash
python scripts/train_prior1d.py \
  --config configs/prior1d_wavelet_paper.json \
  --data-root "$SEISPACE_DATA_ROOT/simulate_wavelet"
```

The output is written to `runs/prior1d_wavelet/`.

Evaluate it:

```bash
python scripts/evaluate_prior1d.py \
  --config configs/prior1d_wavelet_paper.json \
  --checkpoint runs/prior1d_wavelet/best.pt \
  --data-root "$SEISPACE_DATA_ROOT/simulate_wavelet"
```

## Train the 2D prior

```bash
python scripts/train_prior2d.py \
  --config configs/prior2d_wavelet.json \
  --mat-file "$SEISPACE_DATA_ROOT/simulate_wavelet/training_data.mat" \
  --output-dir runs/prior2d_wavelet
```

Evaluate it:

```bash
python scripts/evaluate_prior2d.py \
  --config configs/prior2d_wavelet.json \
  --checkpoint runs/prior2d_wavelet/best.pt \
  --test-mat "$SEISPACE_DATA_ROOT/simulate_wavelet/testing_data.mat" \
  --output runs/eval_prior2d_wavelet/summary_0db.json
```

## Two-stage SeisPACE training

### 1D traces

Generate frozen prior predictions for the calibration split:

```bash
python scripts/prepare_seispace1d_data.py \
  --config configs/prior1d_wavelet_paper.json \
  --checkpoint runs/prior1d_wavelet/best.pt \
  --mat-file "$SEISPACE_DATA_ROOT/simulate_wavelet/training_data.mat" \
  --output-dir data/seispace1d
```

Train the direct baseline, consolidate it without the prior, then calibrate:

```bash
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
```

Evaluate:

```bash
python scripts/evaluate_seispace1d.py \
  --prior-config configs/prior1d_wavelet_paper.json \
  --prior-checkpoint runs/prior1d_wavelet/best.pt \
  --refiner-checkpoint runs/seispace1d_main/best.pt \
  --test-mat "$SEISPACE_DATA_ROOT/simulate_wavelet/testing_data.mat" \
  --output runs/eval_seispace1d/0db.json \
  --direct-refiner --snr-db 0
```

### 2D sections

```bash
python scripts/prepare_seispace2d_data.py \
  --config configs/prior2d_wavelet.json \
  --checkpoint runs/prior2d_wavelet/best.pt \
  --mat-file "$SEISPACE_DATA_ROOT/simulate_wavelet/training_data.mat" \
  --output-dir data/seispace2d

python scripts/train_seispace2d.py \
  --scrn-root "$SEISPACE_SCRN_ROOT" \
  --data-dir data/seispace2d \
  --output-dir runs/seispace2d_baseline \
  --input-mode noisy \
  --epochs 80

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
  --anchor-weight 0.5 \
  --epochs 20
```

Evaluate:

```bash
python scripts/evaluate_seispace2d.py \
  --scrn-root "$SEISPACE_SCRN_ROOT" \
  --prior-config configs/prior2d_wavelet.json \
  --prior-checkpoint runs/prior2d_wavelet/best.pt \
  --refiner-checkpoint runs/seispace2d_main/best.pt \
  --test-mat "$SEISPACE_DATA_ROOT/simulate_wavelet/testing_data.mat" \
  --output runs/eval_seispace2d/0db.json \
  --input-mode noisy_prior --num-samples 100 --snr-db 0
```

## Prior controls

The prior-control experiment keeps the noisy input, checkpoint, and noise
realization fixed while changing only the prior channel:

- `correct`: same-sample prior prediction.
- `zero`: all-zero prior input.
- `shuffled`: deterministic cross-sample prior input.

Use `--prior-mode correct|zero|shuffled` in the SeisPACE evaluation scripts.

## Run inference on a NumPy array

```bash
python scripts/infer_seispace.py 1d \
  --input noisy_trace.npy \
  --output seispace_trace.npz \
  --prior-config configs/prior1d_wavelet_paper.json \
  --prior-checkpoint runs/prior1d_wavelet/best.pt \
  --calibrator-config configs/seispace1d_calibrator.json \
  --calibrator-checkpoint runs/seispace1d_main/best.pt
```

For 2D sections, use mode `2d` and also pass
`--scrn-root "$SEISPACE_SCRN_ROOT"`. The output archive contains the noisy
input, frozen prior prediction, and final calibrated estimate.

## Main results

The representative results reported in the manuscript are:

| Benchmark | SeisPACE | Strongest control |
|---|---:|---:|
| 1D, mean over 11 SNRs | 30.91 dB gain | 30.07 dB no-prior |
| Controlled 2D, 0 dB | 23.34 dB | 23.25 dB no-prior |
| FORGE field, 0 dB | 17.68 dB | 17.67 dB no-prior |
| Marmousi2, 0 dB | 28.29 dB | 27.52 dB direct baseline |

See `runs/results_report.md` in the experiment workspace for the complete
result tables and protocol notes.

## Legacy checkpoints

Models trained before the SeisPACE rename can still be loaded:

```python
from seispace.prior1d import DualSparsePrior1D

model = DualSparsePrior1D(...)
model.load_compatible_state_dict(state_dict)
```

The compatibility loader accepts both the released 1D prior keys and the
earlier 2023 checkpoint naming scheme. See
[docs/legacy_compatibility.md](docs/legacy_compatibility.md).

## Tests

```bash
python -m pytest -q tests
```

Tests that require PyTorch are skipped automatically when PyTorch is not
installed.

## Citation and license

Citation metadata is provided in `CITATION.cff`. This repository is released
under the MIT License; see `LICENSE`.
