# Trained checkpoints

Seed-0 checkpoints for three training regimes, one file per sense and checkpoint stage
(`<sense>_<regime>_stage<k>.pt`, k = 0..3 for the 0/3/6/12-month checkpoints).

| Folder | Regime |
|---|---|
| `curriculum_only/` | Age-indexed immature input, fixed-size network (primary regime) |
| `adult_scratch/` | Adult-quality input from the start |
| `grown/` | Immature input and function-preserving capacity growth |

Each file is a dict `{width, window, state}`. Load with:

```python
import torch
from dwm.models import SenseWorldModel
ck = torch.load("models/curriculum_only/vision_curriculum_only_stage3.pt", map_location="cpu")
m = SenseWorldModel("vision", ck["width"], ck["window"])
m.load_state_dict(ck["state"]); m.eval()
```

Remaining regimes and seeds are regenerated with `scripts/run_experiments.py`.
