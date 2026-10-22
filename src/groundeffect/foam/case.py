"""Write an OpenFOAM (ESI v2512) steady RANS case for the wing in ground effect.

Physics: incompressible, k-omega SST, moving ground at freestream speed
(rolling road), slip roof. Wall treatment uses Spalding's law so the same
setup is valid across the y+ range seen over the design space.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from groundeffect.geometry import PlacedWing

HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


@dataclass(frozen=True)
class FlowSpec:
    u_inf: float = 25.0  # m/s
    nu: float = 1.5e-5  # m^2/s
    turbulence_intensity: float = 0.002
    viscosity_ratio: float = 1.0
    end_iter: int = 1500
    p_tol: float = 1e-5
    u_tol: float = 1e-6

    def k(self) -> float:
        return 1.5 * (self.u_inf * self.turbulence_intensity) ** 2

    def omega(self) -> float:
        return self.k() / (self.nu * self.viscosity_ratio)

    def reynolds(self, chord: float) -> float:
        return self.u_inf * chord / self.nu


def _write(path: Path, cls: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER.format(cls=cls, obj=path.name) + body)


def _field(dims: str, internal: str, patches: dict[str, str]) -> str:
    bf = "\n".join(f"    {name}\n    {{\n        {spec}\n    }}" for name, spec in patches.items())
    return f"dimensions      {dims};\n\ninternalField   {internal};\n\nboundaryField\n{{\n{bf}\n}}\n"


def write_case(case: Path, wing: PlacedWing, flow: FlowSpec, span: float) -> None:
    """Populate ``case`` with 0/, constant/ and system/ (mesh is converted separately)."""
    u = f"uniform ({flow.u_inf} 0 0)"
    k, om = flow.k(), flow.omega()
    empty = "type empty;"
    wall_k = "type kqRWallFunction; value uniform 0;"
    wall_om = f"type omegaWallFunction; value uniform {om};"
    wall_nut = "type nutUSpaldingWallFunction; value uniform 0;"

    _write(
        case / "0/U",
        "volVectorField",
        _field(
            "[0 1 -1 0 0 0 0]",
            u,
            {
                "inlet": f"type fixedValue; value {u};",
                "outlet": f"type inletOutlet; inletValue uniform (0 0 0); value {u};",
                "top": "type slip;",
                "ground": f"type fixedValue; value {u};",  # rolling road
                "wing": "type noSlip;",
                "frontAndBack": empty,
            },
        ),
    )
    _write(
        case / "0/p",
        "volScalarField",
        _field(
            "[0 2 -2 0 0 0 0]",
            "uniform 0",
            {
                "inlet": "type zeroGradient;",
                "outlet": "type fixedValue; value uniform 0;",
                "top": "type slip;",
                "ground": "type zeroGradient;",
                "wing": "type zeroGradient;",
                "frontAndBack": empty,
            },
        ),
    )
    for name, value, wall in (("k", k, wall_k), ("omega", om, wall_om)):
        _write(
            case / f"0/{name}",
            "volScalarField",
            _field(
                "[0 2 -2 0 0 0 0]" if name == "k" else "[0 0 -1 0 0 0 0]",
                f"uniform {value}",
                {
                    "inlet": f"type fixedValue; value uniform {value};",
                    "outlet": f"type inletOutlet; inletValue uniform {value}; value uniform {value};",
                    "top": "type slip;",
                    "ground": wall,
                    "wing": wall,
                    "frontAndBack": empty,
                },
            ),
        )
    _write(
        case / "0/nut",
        "volScalarField",
        _field(
            "[0 2 -1 0 0 0 0]",
            "uniform 0",
            {
                "inlet": "type calculated; value uniform 0;",
                "outlet": "type calculated; value uniform 0;",
                "top": "type calculated; value uniform 0;",
                "ground": wall_nut,
                "wing": wall_nut,
                "frontAndBack": empty,
            },
        ),
    )

    _write(case / "constant/transportProperties", "dictionary", f"transportModel  Newtonian;\nnu              {flow.nu};\n")
    _write(
        case / "constant/turbulenceProperties",
        "dictionary",
        "simulationType  RAS;\n\nRAS\n{\n    RASModel        kOmegaSST;\n    turbulence      on;\n    printCoeffs     off;\n}\n",
    )

    _write(case / "system/controlDict", "dictionary", _control_dict(wing, flow, span))
    _write(case / "system/fvSchemes", "dictionary", FV_SCHEMES)
    _write(case / "system/fvSolution", "dictionary", _fv_solution(flow))
    _write(case / "system/changeDictionaryDict", "dictionary", CHANGE_DICT)


def _control_dict(wing: PlacedWing, flow: FlowSpec, span: float) -> str:
    qx, qy = wing.quarter_chord
    return f"""application     simpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {flow.end_iter};
deltaT          1;
writeControl    timeStep;
writeInterval   {flow.end_iter};
purgeWrite      1;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;

functions
{{
    forceCoeffs
    {{
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        log             false;
        patches         (wing);
        rho             rhoInf;
        rhoInf          1;
        CofR            ({qx} {qy} 0);
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        pitchAxis       (0 0 1);
        magUInf         {flow.u_inf};
        lRef            {wing.chord};
        Aref            {wing.chord * span};
    }}
    solverInfo
    {{
        type            solverInfo;
        libs            (utilityFunctionObjects);
        fields          (U p k omega);
        writeResidualFields no;
    }}
    yPlus
    {{
        type            yPlus;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
        patches         (wing ground);
    }}
    wingSurface
    {{
        type            surfaces;
        libs            (sampling);
        writeControl    writeTime;
        surfaceFormat   raw;
        fields          (p);
        interpolationScheme cell;
        surfaces
        {{
            wing {{ type patch; patches (wing); interpolate false; }}
            ground {{ type patch; patches (ground); interpolate false; }}
        }}
    }}
}}
"""


FV_SCHEMES = """ddtSchemes      { default steadyState; }
gradSchemes
{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
    grad(k)         cellLimited Gauss linear 1;
    grad(omega)     cellLimited Gauss linear 1;
}
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss limitedLinear 1;
    div(phi,omega)  bounded Gauss limitedLinear 1;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited corrected 0.5; }
interpolationSchemes { default linear; }
snGradSchemes   { default limited corrected 0.5; }
wallDist        { method meshWave; }
"""


def _fv_solution(flow: FlowSpec, robust: bool = False) -> str:
    # Default: SIMPLEC, fast. Robust: plain SIMPLE with stronger under-relaxation,
    # used to recover cases that settle into a numerical limit cycle.
    consistent = "no" if robust else "yes"
    relax = 'equations { U 0.7; "(k|omega)" 0.7; }\n    fields { p 0.3; }' if robust else 'equations { U 0.9; "(k|omega)" 0.7; }'
    return f"""solvers
{{
    p
    {{
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.05;
    }}
    "(U|k|omega)"
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-9;
        relTol          0.1;
    }}
}}
SIMPLE
{{
    nNonOrthogonalCorrectors 0;
    consistent      {consistent};
    residualControl
    {{
        p               {flow.p_tol};
        U               {flow.u_tol};
        "(k|omega)"     {flow.u_tol * 10};
    }}
}}
relaxationFactors
{{
    {relax}
}}
"""


def write_fv_solution(case: Path, flow: FlowSpec, robust: bool = False) -> None:
    _write(case / "system/fvSolution", "dictionary", _fv_solution(flow, robust))


def extend_run(case: Path, end_iter: int) -> None:
    """Move endTime (and the single write) to ``end_iter`` for a restart."""
    path = case / "system/controlDict"
    text = re.sub(r"^endTime\s+\d+;", f"endTime         {end_iter};", path.read_text(), flags=re.M)
    text = re.sub(r"^writeInterval\s+\d+;", f"writeInterval   {end_iter};", text, flags=re.M)
    path.write_text(text)


CHANGE_DICT = """boundary
{
    frontAndBack { type empty; }
    wing         { type wall; }
    ground       { type wall; }
    top          { type patch; }
}
"""

ALLRUN = """#!/bin/bash
set -euo pipefail
cd "${0%/*}"
run() { local app=$1; shift; "$app" "$@" > "log.$app" 2>&1 || { echo "FAILED: $app" >&2; tail -20 "log.$app" >&2; exit 1; }; }
run gmshToFoam mesh.msh
run changeDictionary
run renumberMesh -overwrite
checkMesh > log.checkMesh 2>&1 || true
run simpleFoam
run postProcess -func writeCellCentres -latestTime
"""


def write_allrun(case: Path) -> Path:
    path = case / "Allrun"
    path.write_text(ALLRUN)
    path.chmod(0o755)
    return path
