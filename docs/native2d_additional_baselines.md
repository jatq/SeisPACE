# Additional baselines for external SEGY benchmarks

## Scope

The FORGE and Marmousi2 caches are evaluated with the same `256 x 128`
patches, test splits, and metrics as the SeisPACE comparison.

All learned baselines are retrained on each source. No external checkpoint is
used for the main comparison.

## Prior controls

The selected SeisPACE calibrator is evaluated three times with the same
checkpoint, test data, and noise realization:

| Condition | Prior input |
|---|---|
| Correct | `P(y)` for the same noisy section |
| Zero | an all-zero prior channel |
| Shuffled | a deterministic cyclic shift across samples |

The control scripts are:

```text
scripts/run_native2d_prior_controls.sh
```

The source-specific controls are in `eval_zero_prior` and
`eval_shuffled_prior`.

## Classical methods

Curvelet thresholding, causal f-x deconvolution, and per-trace wavelet
thresholding have no training stage. They are applied directly to the cached
noisy test sections with the same parameterization used in the controlled 2D
table.

```text
scripts/run_native2d_classical.sh
```

## PS-SGMs

The official PS-SGMs architecture is trained from random initialization on
clean `128 x 128` patches from each source for 10,000 iterations. Snapshots at
2.5k, 5k, 7.5k, and 10k are evaluated on 40 validation sections at `0 dB`.
Both sources select iteration 10,000.

This is a compute-limited reproduction, not the official 210,001-iteration
budget.

```text
scripts/run_native2d_ps_sgms.sh
scripts/run_native2d_ps_sgms_eval_parallel.sh
```

## SeisDN

SeisDN uses the official canonical STFT parameters and is retrained for 80
epochs on spectrogram samples from the source training split. At test time,
every trace in each section is denoised, the section is reconstructed, and the
same section-level SNR and SSIM metrics are computed.

```text
scripts/run_native2d_seisdn_prepare.sh
scripts/run_native2d_seisdn_train.sh
scripts/run_native2d_seisdn_eval.sh
```

## MSMHA

MSMHA remains a transparent PyTorch reimplementation because the original
TensorFlow/Keras stack is incompatible with the current environment. It is
retrained for 100 epochs on 16,000 noisy `48 x 48` patches per source.

```text
scripts/run_native2d_msmha.sh
scripts/run_native2d_msmha_eval.sh
```

## Aggregation

`scripts/summarize_native2d_comparison.py` combines all source-specific
results into `comparison_full.json`, `comparison_full.csv`, and comparison
figures.
