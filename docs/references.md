# References

These are the sources the code and the results build on, each with the file that uses it.

The papers themselves are not stored in the repository, because publishers' copyrights apply. Keep local copies in `refs/` at the repository root, which git ignores.

## The case

- **Zerihan, J. and Zhang, X. (2000).** Aerodynamics of a single element wing in ground effect. *Journal of Aircraft* 37(6), 1058–1064. [doi:10.2514/2.2711](https://doi.org/10.2514/2.2711)
  - Basis of the case: an inverted single-element wing over a moving ground, with ride height measured to the lowest point of the section (`geometry.py`).
  - Describes the force-reduction regime that the ride-height sweep reproduces: below about 10% of chord, downforce drops as the boundary layer separates at the trailing edge.

## CFD, meshing and verification

- **Menter, F. R. (1994).** Two-equation eddy-viscosity turbulence models for engineering applications. *AIAA Journal* 32(8), 1598–1605. [doi:10.2514/3.12149](https://doi.org/10.2514/3.12149)
  - The k-ω SST turbulence model (`foam/case.py`).
- **Spalding, D. B. (1961).** A single formula for the “law of the wall”. *Journal of Applied Mechanics* 28(3), 455–458. [doi:10.1115/1.3641728](https://doi.org/10.1115/1.3641728)
  - The wall function, valid across the y⁺ range the design space produces (`foam/case.py`).
- **Celik, I. B., Ghia, U., Roache, P. J., Freitas, C. J., Coleman, H. and Raad, P. E. (2008).** Procedure for estimation and reporting of uncertainty due to discretization in CFD applications. *Journal of Fluids Engineering* 130(7), 078001. [doi:10.1115/1.2960953](https://doi.org/10.1115/1.2960953)
  - The three-grid grid convergence index (GCI) behind the noise floor (`verification.py`).
- **Geuzaine, C. and Remacle, J.-F. (2009).** Gmsh: a 3-D finite element mesh generator with built-in pre- and post-processing facilities. *International Journal for Numerical Methods in Engineering* 79(11), 1309–1331. [doi:10.1002/nme.2579](https://doi.org/10.1002/nme.2579)
  - The automated hybrid mesh (`meshing.py`).
- **OpenFOAM v2512**, OpenCFD Ltd. [openfoam.com](https://www.openfoam.com)
  - The solver, run in the official `opencfd/openfoam-default:2512` image (`foam/`).

## Design of experiments and surrogates

- **Sobol', I. M. (1967).** On the distribution of points in a cube and the approximate evaluation of integrals. *USSR Computational Mathematics and Mathematical Physics* 7(4), 86–112. [doi:10.1016/0041-5553(67)90144-9](https://doi.org/10.1016/0041-5553(67)90144-9)
  - The training design (`doe.py`).
- **McKay, M. D., Beckman, R. J. and Conover, W. J. (1979).** A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. *Technometrics* 21(2), 239–245. [doi:10.1080/00401706.1979.10489755](https://doi.org/10.1080/00401706.1979.10489755)
  - Latin hypercube sampling, used for the independent test design (`doe.py`).
- **Ginsbourger, D., Le Riche, R. and Carraro, L. (2010).** Kriging is well-suited to parallelize optimization. In *Computational Intelligence in Expensive Optimization Problems*, Springer, 131–162. [doi:10.1007/978-3-642-10701-6_6](https://doi.org/10.1007/978-3-642-10701-6_6)
  - The “kriging believer” used to pick batches in active learning (`models/gp.py`).
- **Lakshminarayanan, B., Pritzel, A. and Blundell, C. (2017).** Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS 2017.* [arXiv:1612.01474](https://arxiv.org/abs/1612.01474)
  - The PyTorch deep ensemble (`models/ensemble.py`).
- **Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A. and Battaglia, P. W. (2021).** Learning mesh-based simulation with graph networks. *ICLR 2021.* [arXiv:2010.03409](https://arxiv.org/abs/2010.03409)
  - MeshGraphNet, the field model's architecture (`models/mgn.py`).
- **NVIDIA PhysicsNeMo.** [github.com/NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo)
  - The MeshGraphNet implementation used for training (`models/mgn.py`).

## Wind-tunnel data QA

- **Killick, R., Fearnhead, P. and Eckley, I. A. (2012).** Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association* 107(500), 1590–1598. [doi:10.1080/01621459.2012.737745](https://doi.org/10.1080/01621459.2012.737745)
  - PELT change-point detection (`windtunnel/detect.py`).
- **NIST/SEMATECH e-Handbook of Statistical Methods**, sections [6.3.2.3 CUSUM control charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm) and [6.3.2.4 EWMA control charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm).
  - The datum-run control charts (`windtunnel/detect.py`).

## Regulations

- **FIA (2026).** 2026 Formula 1 Regulations, Section F (Operational), Issue 11, 1 October 2026. [PDF](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_f_operational_-_iss_11_-_2026-10-01.pdf)
  - Article F4.4.1 brings machine learning on simulation results into restricted CFD and allows GPU cores from 2028.
  - Article F4.5.5 covers tunnel-infrastructure runs; F4.6.7 covers surface-mesh records.
  - These motivate the CFD-budget and data-lineage choices; no code depends on them.
