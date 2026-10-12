# inscribed-polytopes

Code and data for computations on convex polytopes with vertices on the unit
sphere, in any dimension. One Python package serves two papers. The project
is managed with [poetry](https://python-poetry.org/).

    poetry install              # core, plus the dev tools
    poetry install -E sos       # adds cvxpy/clarabel for certificate search
    poetry run python scripts/setup_data.py   # builds plantri, fetches Firsching's 4D data
    poetry run pytest tests

## Layout

| path | contents |
| --- | --- |
| `src/inscribed/certify/` | exact polynomial arithmetic and the certificate verifier (no solver dependency) |
| `src/inscribed/sos/` | numerical certificate search, rounding, moment relaxations |
| `src/inscribed/search/` | ascent and per-type sweeps |
| `src/inscribed/combinatorics/`, `enumeration/` | oriented simplicial complexes; plantri (3D) and Firsching (4D) type sources |
| `src/inscribed/analysis/`, `store/`, `catalog/`, `viz/` | validation, records and JSON I/O, published values, figures |
| `scripts/` | command-line entry points; each supports `--help` |
| `results/`, `results/free/` | committed sweep and type-free search records |
| `results/certificates/` | committed exact certificates |
| `papers/sweep/`, `papers/maxvol-d4/` | the two papers; each builds with its own `build.sh` |
| `tests/`, `tests/sos/` | tests of the package, and of the sweep paper's LaTeX against code and data |
| `tests/papers/maxvol_d4/` | tests that read the maxvol-d4 paper's LaTeX and check it against code and data |

The paths under `tests/` and `results/` linked by the sweep paper (already
published) are frozen; everything belonging to the maxvol-d4 paper lives in its
own directories.

## License

Apache 2.0; see LICENSE and NOTICE. The vendored plantri is by Brinkmann and McKay.
