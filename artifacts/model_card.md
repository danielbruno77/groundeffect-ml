# Model card: ground-effect wing surrogate

**Dataset** `c9158e509306`: 235 training / 45 test CFD cases
(2D steady RANS, k-omega SST, OpenFOAM v2512, Re = 5e5, moving ground). Only cases that passed
convergence and mesh QA are used.

**Model**: deep ensemble of 5 MLPs (PyTorch) predicting CL (downforce), CD, Cm and 10
POD coefficients of surface Cp, with variance calibrated on a held-out split. Exported with `torch.export`.

## Release gates (independent Latin-hypercube test design)

| gate | value | threshold | result |
|---|---|---|---|
| `cl_down.rmse` | 0.0047 | max 0.013 | pass |
| `cl_down.mape` | 0.0033 | max 0.01 | pass |
| `cl_down.r2` | 0.9997 | min 0.995 | pass |
| `cl_down.coverage_2s` | 0.8889 | min 0.85 | pass |
| `cl_down.pair_sign_accuracy` | 1.0000 | min 0.98 | pass |
| `cl_down.spearman` | 0.9988 | min 0.99 | pass |
| `cd.mape` | 0.0049 | max 0.02 | pass |
| `cd.r2` | 0.9999 | min 0.99 | pass |
| `cd.pair_sign_accuracy` | 0.9960 | min 0.97 | pass |
| `cm.r2` | 0.9996 | min 0.99 | pass |
| `cp.rmse` | 0.0087 | max 0.02 | pass |
| `cp.rmse_worst_case` | 0.0222 | max 0.05 | pass |

Downforce: R2 0.9997, RMSE 0.0047, max error 0.0132.
CFD discretisation uncertainty of CL on the production mesh (mean GCI): 0.0131.
GP baseline on the same split: CL R2 0.9998, RMSE 0.0041.

## Valid domain

| variable | min | max |
|---|---|---|
| `h_c` | 0.08 | 0.6 |
| `alpha_deg` | 0.0 | 8.0 |
| `camber` | 0.02 | 0.06 |
| `thickness` | 0.09 | 0.15 |

## Limitations

* 2D sections: no tip vortices, endplates, or 3D relief; trends, not car numbers.
* Steady RANS: cases with massive separation (lowest ride heights at high incidence and camber)
  are rejected by QA and are therefore *not represented*; the model must not be trusted there.
* Outside the bounds above the service flags the request as extrapolation.
