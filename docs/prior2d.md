# 2D dual-sparse prior

The 2D dual-sparse prior applies the same fixed-transform,
analysis-DAE-synthesis design as the 1D prior while preserving spatial context
across traces.

## Processing

1. Extract overlapping `16 x 16` patches with stride `8`.
2. Apply a separable orthonormal 2D DCT, producing 256 coefficients per patch.
3. Map the coefficients to a 128-dimensional learned embedding.
4. Denoise the complete 2D patch grid with a convolutional context DAE.
5. Apply the inverse embedding and 2D inverse DCT with overlap-add averaging.

The embedding and inverse embedding are initialized as a truncated DCT
projection. The inverse-consistency loss targets

```text
W_synthesis @ W_analysis = Pi
```

where `Pi` projects onto the retained 128-dimensional coefficient subspace.

The prior receives a complete `256 x 128` section in one forward pass. Input
normalization is computed over the whole section. Training uses the
coefficient-domain MAE plus the inverse-consistency regularizer.

## Reproducibility

Configuration:

```text
configs/prior2d_wavelet.json
```

Scripts:

```text
scripts/train_prior2d.py
scripts/evaluate_prior2d.py
```

The DCT/IDCT reconstruction and output-shape invariants are tested in
`tests/test_prior2d.py`.
