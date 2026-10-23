# Recent baseline plan

The benchmark includes retrained recent learned baselines and training-free
classical references. Methods are included only when their preprocessing,
training procedure, and evaluation protocol can be reproduced on the same data.

## Recommended minimum

| Priority | Method | Year / venue | Code | Why |
|---|---|---|---|---|
| 1 | SCRN | 2024, Computers & Geosciences | `javashs/SCRN` | Retrained from scratch on our wavelet-derived 2D sections |
| 2 | SeisDN | 2026, Journal of Applied Geophysics | `Liang-Ding/seisDN` | Retrained from scratch on our per-trace STFT data using the official preprocessing |
| 3 | MSMHA | 2025, Artificial Intelligence in Geosciences | `DeepWave-KAUST/MSMHA_Denoising-pub` | Self-supervised multi-stage network, public code |
| 4 | PS-SGMs | 2024, IEEE TGRS | `mengchuangji/PS-SGMs-seismic` | Score-based generative baseline, public code |
| 5 | f-x deconvolution | classical | implement / `pyseistr` | Standard seismic random-noise baseline |
| 6 | curvelet thresholding | classical | implement | Tests whether the learned transform really helps |

This gives two modern deep baselines, one self-supervised baseline, one
generative baseline, and two classical references. It is enough without making
the comparison section unmanageable.

Current same-data reproduction status:

- 1D dual-sparse prior: current implementation, 2000 epochs, random
  initialization, official 200-trace synthetic training set.
- trace-wise 2D prior: 1D DCT model retrained on 8000 traces from
  2000 sections, 500 epochs.
- 2D dual-sparse prior: true 2D DCT transform model trained on full sections
  in one forward pass, early-stopped at epoch 80.
- DeepDenoiser: retrained from scratch on the same 200 training traces.
- DeepSeg: retrained from scratch on the same 200 training traces.
- SCRN: retrained from scratch on wavelet-derived 2D sections, 80 epochs.
- SeisDN: retrained from scratch on per-trace STFT data, 80 epochs, using the
  same STFT parameters as the official `predict.py`.
- MSMHA: a PyTorch implementation of the official Keras architecture, trained
  self-supervised from scratch on noisy patches from our sections.
- f-x deconvolution: implemented and evaluated.
- curvelet thresholding: implemented and evaluated.
- wavelet thresholding: implemented and evaluated.
- PS-SGMs: official NCSNv2 score model and conditional posterior sampler
  retrained from scratch on our clean sections. The selected checkpoint is
  5k iterations after validation; training stopped at 18,348 iterations. This
  is compute-limited relative to the official 210,001-iteration recipe.
- SeisPACE: 1D and 2D first train a same-data direct baseline, consolidate it
  with prior-free paired training, inject the learned prior through a zeroed channel,
  and fine-tune with the consolidated baseline as an anchor. The 1D baseline
  is a temporal U-Net; the 2D baseline is SCRN. Neither model uses output
  ensembling.

External checkpoints are sanity checks only. They are not used in the main
comparison table.

## Second tier

These are relevant but have weaker code availability or are less directly
compatible with the current 1D trace benchmark:

| Method | Year / venue | Status |
|---|---|---|
| Improved MCA-SCUNet | 2024, IEEE TGRS | No official code found; reimplementation needed |
| Self-Supervised Pretraining Transformer | 2024, IEEE TGRS | No official code found |
| Swin Transformer for Seismic Denoising | 2024, IEEE GRSL | No official code found |
| DARE U-Net | 2024, Remote Sensing | No official code found |
| DPN2N | 2025, Journal of Geophysics and Engineering | No official code found |
| KAN-Empowered Neural Low-Rank Representation | 2025, IEEE TGRS | No official code found |
| Dual-domain Mamba | 2025, Journal of Applied Geophysics | No official code found |

Do not add these to the paper until their preprocessing and weights are
reproducible.

## Important scope issue

SeisDN, SCRN, MSMHA, and PS-SGMs are primarily 2D/3D methods. Running them on a
1D trace benchmark requires adaptation and may invite criticism if the
adaptation is too loose.

There are two defensible choices:

1. Add a 2D synthetic or field-data section and compare SeisDN, SCRN, and
   PS-SGMs in their native setting.
2. Keep the 1D trace benchmark and add only the methods that can be adapted
   without changing their core idea, then state the adaptation explicitly.

The strongest paper will do both: retain the fast 1D benchmark and add 2D
validation sections.

The controlled 2D section benchmark is synthetic and derived from the same 1D
wavelet/noise source. It is useful for controlled comparisons, but it remains
separate from the following external SEGY benchmarks:

- FORGE DAS `1280 x 30000` native field section with controlled STEAD noise
  injection;
- Marmousi2 `2721 x 1001` public synthetic time section with a held-out trace
  split and controlled STEAD noise injection.

Neither external benchmark has a recorded noiseless field reference. Metrics
measure the removal of injected noise and preservation of the selected SEGY
reference, not supervised removal of the true field noise.

Both external benchmarks now include same-data retraining of PS-SGMs, SeisDN,
and the MSMHA reimplementation, untrained curvelet/f-x/wavelet baselines, and
correct/zero/shuffled prior controls.

## Recommended experiment order

1. Keep the same-data 1D and 2D benchmark tables.
2. Report local Gaussian-window SSIM, waveform correlation, and amplitude ratio.
3. Report PS-SGMs as compute-limited and early-stopped.
4. Run the full 210k-iteration PS-SGMs recipe only if a reviewer requires it.
5. Keep every external checkpoint in a separately labelled sanity-check table.

## What not to do

- Do not compare against methods using different noise distributions without
  matching the noise-generation protocol.
- Do not report only SNR gain. Add local waveform correlation, amplitude
  preservation, arrival-time error, and downstream detection or picking.
- Do not use a 2D method on a single trace and call it an unchanged official
  baseline.
- Do not cite a method only from its abstract; use its code or a transparent
  reimplementation.
