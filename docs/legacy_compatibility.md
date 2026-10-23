# Legacy checkpoint compatibility

The public SeisPACE code uses two names consistently:

- `DualSparsePrior1D` and `DualSparsePrior2D` for the priors.
- `SeisPACECalibrator1D` for the 1D calibrator.

Historical checkpoints remain usable through
`DualSparsePrior1D.load_compatible_state_dict`. The loader translates two key
layouts:

| Checkpoint family | Main keys |
|---|---|
| released 1D prior | `embed`, `inverse_embed`, `dae.unet` |
| 2023 development checkpoint | `sparse_map`, `unet`, `head` |

The following buffers from the released 1D implementation are ignored because
the current implementation computes DCT/IDCT with SciPy:

```text
dct_weight
idct_weight
identity
```

Example:

```python
import torch
from seispace.prior1d import DualSparsePrior1D

state = torch.load("legacy_prior.pt", map_location="cpu", weights_only=False)
state = state.get("state_dict", state)

model = DualSparsePrior1D(embed_dim=256)
model.load_compatible_state_dict(state)
model.eval()
```

No external prior repository is required.
