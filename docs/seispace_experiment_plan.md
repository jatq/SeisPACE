# SeisPACE experiment protocol

Paper title:

```text
SeisPACE: Baseline-Anchored Prior Calibration for Seismic Denoising
```

This file fixes the experimental definitions. It is not manuscript text.

## Method definitions

Let `y` be the noisy observation, `s` the clean reference, `P` a frozen learned
seismic prior, `B` a same-data direct baseline, and `R` the dual-input
calibrator:

```text
p = P(y)
s_hat = R(y, p)
```

The calibrator is initialized from `B*` by copying all shared weights and zeroing
the new prior channel. Therefore its initialization satisfies:

```text
R_0(y, 0) = B*(y)
```

The calibration objective is:

```text
L = L_task(R(y, P(y)), s)
    + lambda_anchor * L_anchor(R(y, P(y)), B*(y))
```

`B*` is frozen and is used only for the training anchor. Inference executes `P`
and `R` once each in series. It does not execute `B`, average outputs, or form
a model ensemble.

## Main protocol

| Component | 1D | 2D |
|---|---|---|
| Prior `P` | retrained dual-sparse prior (1D), frozen | retrained dual-sparse prior (2D), frozen |
| Baseline `B` | temporal U-Net, same-data training | SCRN, same-data training |
| Random/no-prior stage | 80 epochs | 80 epochs |
| Baseline consolidation | 80 prior-free paired epochs | 80 prior-free paired epochs |
| Late prior injection | initialize a two-channel calibrator from the consolidated baseline and zero the new prior channel | initialize a two-channel calibrator from the consolidated baseline and zero the new prior channel |
| Anchor stage | 20 epochs, `lambda=0.5` selected on validation | 20 epochs, `lambda=0.5` selected on validation |
| Inference | `P` then one `R` | `P` then one `R` |

The test set is never used to select an anchor weight or checkpoint.

The 1D result is repeated with seeds 42, 43, and 44. The synthetic 2D main
result is repeated with the same seeds. Seed 42 is used for the ablation matrix;
the extra seeds estimate training stability.

## Ablation matrix

| ID | Input | Initialization | Anchor | Purpose |
|---|---|---|---|---|
| 1D-B | noisy | random | none | direct baseline |
| 1D-R80 | noisy + prior | random | none | random two-input model |
| 1D-N80 | noisy | baseline warm start | none | extra training without prior |
| 1D-W80 | noisy + prior | original baseline warm start, prior channel zero | none | early prior injection |
| 1D-L20 | noisy + prior | consolidated baseline, prior channel zero | yes | proposed late prior injection |
| 1D-R80-A20 | noisy + prior | random at epoch 0, 80 epochs, then anchor fine-tune | yes | effect of warm start with anchor |
| 1D-W80-A20 | noisy + prior | baseline warm start, 80 prior-aware epochs, then anchor fine-tune | yes | early-injection ablation |
| 2D-B80 | noisy | random | none | SCRN baseline |
| 2D-R80 | noisy + prior | random | none | random two-input model |
| 2D-N80 | noisy | SCRN warm start | none | extra training without prior |
| 2D-W80 | noisy + prior | original SCRN warm start, prior channel zero | none | early prior injection |
| 2D-L20 | noisy + prior | consolidated baseline, prior channel zero | yes | proposed late prior injection |
| 2D-R80-A20 | noisy + prior | random at epoch 0, 80 epochs, then anchor fine-tune | yes | effect of warm start with anchor |
| 2D-W80-A20 | noisy + prior | SCRN warm start, 80 prior-aware epochs, then anchor fine-tune | yes | early-injection ablation |

Anchor weights `0.5`, `1.0`, and `2.0` are compared on validation data for both
modalities. No test result is used to choose between them.

## Prior controls

The final model is evaluated with three paired-input conditions:

| Condition | Prior channel |
|---|---|
| Correct | `P(y)` for the same observation |
| Zero | all-zero prior |
| Shuffled | a deterministic cross-sample cyclic shift of `P` |

Zero-prior inference tests whether the model actually uses the added channel.
Shuffled-prior inference tests whether correct sample pairing matters. The test
set and checkpoint remain fixed across all three conditions.

## Native 2D field benchmark

The native field benchmark uses the FORGE DAS SEGY record described in
`native2d_field.md`. It is a `1280 x 30000` field DAS section with `0.5 ms`
sampling. Because no noiseless field reference exists, normalized field blocks
serve as references and controlled real STEAD ambient noise is added for
scoring. The split is by time blocks, not random overlapping test patches.

All task-specific models and the prior generator are retrained on the field
training split. This benchmark is reported separately from the synthetic main
table.

## Public Marmousi2 SEGY benchmark

The Marmousi2 source audit is in `marmousi2_source.md`. The available public
file is a `2721 x 1001` synthetic time section distributed by the University
of Houston, not a field recording and not the original 1988 Marmousi shot
gathers. It is therefore labelled as a public synthetic SEGY benchmark.

The train/validation/test split is made along traces, so the held-out test
traces never occur in the training or validation patches. The same controlled
STEAD noise injection and `256 x 128` patch protocol are used. All baselines,
task-specific priors, and SeisPACE stages must be retrained on this source
rather than transferred from the FORGE or wavelet checkpoints.

Both external benchmarks standardize each `256 x 128` block as a whole after
percentile clipping. Near-constant blocks are resampled. The earlier row-wise
normalization is archived as `*_rownorm_20261022` and must not be used in the
paper.

The external benchmarks also include same-data retraining of PS-SGMs, SeisDN,
and the MSMHA reimplementation, plus untrained curvelet, f-x deconvolution,
and wavelet thresholding. Correct, all-zero, and deterministically shuffled
prior inputs are evaluated with the same selected SeisPACE checkpoint. The
details and commands are in `native2d_additional_baselines.md`.

## Metrics

### 1D

- SNR gain over 11 nominal levels from `-10` to `10 dB`.
- Mean SNR gain.
- Correlation.
- Amplitude ratio.
- Optimal-scale SNR.

### 2D

- Achieved SNR at `-10`, `0`, and `10 dB`.
- Local Gaussian-window SSIM.
- Correlation.
- Amplitude ratio.
- Optimal-scale SNR.

## Planned figures

1. Conceptual figure: direct denoising versus prior calibration.
2. SeisPACE architecture: frozen prior path, baseline warm start, anchor branch
   during training only, and single-path inference.
3. Warm-start and anchor geometry: `R_0(y,0)=B(y)` and the anchor trust region.
4. Main 1D SNR curves across 11 levels.
5. Main 2D qualitative comparison and diagnostic metrics.
6. Ablation figure covering random/warm initialization, prior-free paired
   continuation, no-anchor warm, and anchored calibration.
7. Prior controls and anchor-weight sensitivity.

## Reporting rules

- Label 1D values as SNR gain and 2D values as achieved SNR.
- Do not mix checkpoint-only sanity checks with same-data retraining.
- Do not call serial `P -> R` inference a single-network or one-pass model.
- Do not claim model ensembling.
- Do not claim full PS-SGMs reproduction; it is compute-limited.
- Label MSMHA as a PyTorch reimplementation.
- Recompute complexity and latency for the complete `P + R` inference path.
