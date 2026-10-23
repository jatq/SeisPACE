# SeisPACE

Official code for **SeisPACE: Two-Stage Baseline-Anchored Calibration with a
Lightweight Dual-Sparse Prior for Seismic Denoising**.

Repository: <https://github.com/jatq/SeisPACE>

This is a compact reference release. It contains the 1D and 2D model code,
training and inference entry points, a few CSV examples, and one bundled
checkpoint. All data exchanged with the code use the same CSV schema.

## Installation

```bash
python -m pip install -e .
```

## Inference

Run the released 1D or 2D checkpoint directly:

```bash
python infer.py --input samples/1d.csv --output predictions_1d.csv
python infer.py --input samples/2d.csv --output predictions_2d.csv
```

The default checkpoint is `checkpoints/seispace.pt`. It contains both the 1D
and 2D prior/calibrator weights. Inference runs one prior and one calibrator;
the training anchor is not evaluated.

## CSV format

Each row is one trace or one two-dimensional section:

```text
sample_id,dimension,shape,clean,noisy
trace_0001,1d,2048,0.1 0.2 ...,0.0 0.3 ...
section_0001,2d,128x64,0.1 0.2 ...,0.0 0.3 ...
```

- `dimension` is `1d` or `2d`.
- `shape` is a length such as `2048`, or `height x width` such as `128x64`.
- `clean` and `noisy` are space-separated flattened arrays. `clean` may be
  empty for inference-only data.
- The output is also CSV and contains `noisy`, `prior`, `estimate`, `snr_db`,
  and `gain_db` columns.

## Training

The minimal release trains the three parts in sequence: prior, direct
baseline, and prior-aware calibrator. The examples below are templates; use
your own CSV files and epoch counts for a full training run.

```bash
# 1D prior
python train.py --dimension 1d --stage prior \
  --train-csv samples/1d.csv --output runs/prior_1d.pt --epochs 100

# 1D direct baseline
python train.py --dimension 1d --stage baseline \
  --train-csv samples/1d.csv --output runs/baseline_1d.pt --epochs 100

# 1D prior-aware calibration and anchor
python train.py --dimension 1d --stage calibrate \
  --train-csv samples/1d.csv \
  --prior-checkpoint runs/prior_1d.pt \
  --baseline-checkpoint runs/baseline_1d.pt \
  --output runs/seispace_1d.pt --epochs 20 --anchor-weight 0.5
```

For 2D, change `--dimension 2d` and use `samples/2d.csv`. The released bundle
has the same layout as the calibrate-stage output. After training,
`runs/seispace_1d.pt` can be passed directly to `infer.py --checkpoint`.

## Repository layout

```text
train.py                    prior, baseline, and calibration training
infer.py                    unified 1D/2D inference
src/seispace/               model and CSV code
samples/1d.csv              a few 1D examples
samples/2d.csv              a few 2D examples
checkpoints/seispace.pt     released bundled checkpoint
tests/                      small CPU/GPU smoke tests
```
