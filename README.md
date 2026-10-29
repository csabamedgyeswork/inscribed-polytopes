# inscribed-polytopes

Code and data for computations on convex polytopes with vertices on the unit
sphere, in any dimension. One Python package serves three papers. The project
is managed with [poetry](https://python-poetry.org/).

    poetry install              # core, plus the dev tools
    poetry install -E sos       # adds cvxpy/clarabel for certificate search
    poetry run python scripts/setup_data.py   # builds plantri, fetches Firsching's 4D data
    poetry run pytest tests

The committed certificates make a clone about 370 MB.

## Layout

| path | contents |
| --- | --- |
| `src/inscribed/certify/` | exact polynomial arithmetic and the certificate verifier (no solver dependency) |
| `src/inscribed/sos/` | numerical certificate search, rounding, moment relaxations |
| `src/inscribed/elimination/` | resumable campaigns certifying every rival type of a case below a bound |
| `src/inscribed/search/` | ascent and per-type sweeps |
| `src/inscribed/combinatorics/`, `enumeration/` | oriented simplicial complexes; plantri (3D) and Firsching (4D) type sources |
| `src/inscribed/analysis/`, `store/`, `catalog/`, `viz/` | validation, records and JSON I/O, published values, figures |
| `scripts/` | command-line entry points; each supports `--help` |
| `results/`, `results/free/` | committed sweep and type-free search records |
| `results/certificates/d4-n0{7,8,9}/` | committed exact certificates; `d4-n09/` also holds the screening cache |
| `papers/sweep/`, `papers/maxvol-d4/`, `papers/maxvol-d4-n9/` | the three papers; each builds with its own `build.sh` |
| `tests/`, `tests/sos/`, `tests/elimination/` | tests of the package, and of the sweep paper's LaTeX against code and data |
| `tests/papers/maxvol_d4/`, `tests/papers/maxvol_d4_n9/` | tests that read each maxvol paper's LaTeX and check it against code and data |

The paths under `tests/` and `results/` linked by the sweep paper (already
published) are frozen; everything belonging to the maxvol-d4 and maxvol-d4-n9
papers lives in their own directories.

## The nine-vertex case (d = 4, n = 9)

`papers/maxvol-d4-n9/` proves that the square–pentagon pair, Firsching's
`firsching-9-10`, is the unique maximum. Every one of the other 1141
combinatorial types, including the two non-inscribable ones, has a stored
certificate bounding its volume by 39/50.

Certify the 1139 inscribable rivals; each type takes roughly 25 to 45 minutes,
proved types are cached and a killed run resumes, and `--shard I/K` splits the
work across terminals or machines:

    poetry run python scripts/eliminate.py -d 4 -n 9 --winner firsching-9-10 --bound 0.78 --shard 0/2
    poetry run python scripts/eliminate.py -d 4 -n 9 --winner firsching-9-10 --bound 0.78 --shard 1/2
    poetry run python scripts/eliminate.py -d 4 -n 9 --winner firsching-9-10 --bound 0.78 --status

Certify `firsching-9-323` and `firsching-9-324`, whose published realisations
are not inscribed and which `eliminate.py` therefore skips:

    poetry run python scripts/eliminate_non_inscribable.py --type 323
    poetry run python scripts/eliminate_non_inscribable.py --type 324

Check the paper against code and data. The default run takes seconds; the slow
tests rebuild the exact facets of all 1142 types, and with
`INSCRIBED_FULL_VERIFY=1` re-verify every certificate in exact arithmetic
(about 90 minutes on 12 workers, set by `INSCRIBED_WORKERS`, default 5):

    poetry run pytest tests/papers/maxvol_d4_n9
    INSCRIBED_FULL_VERIFY=1 INSCRIBED_WORKERS=12 poetry run pytest tests/papers/maxvol_d4_n9 -m slow

## License

Apache 2.0; see LICENSE and NOTICE. The vendored plantri is by Brinkmann and McKay.
