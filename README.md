# inscribed-polytopes

Computational machinery for the extremal problem

> among all convex polytopes with `n` vertices on the unit sphere `S^{d-1}`,
> which has the greatest volume?

Dimension-agnostic: the same code runs on `S^2`, `S^3` and beyond. In three
dimensions it reproduces every known exact answer; in four it reproduces the
known cases and, by exhaustive sweep over all combinatorial types, addresses
the `n = 7` and `n = 8` cross-class questions numerically.

```bash
pip install -e .                                   # numpy, scipy; matplotlib for figures
python -m unittest discover -s tests -t .          # 177 tests
python scripts/setup_data.py                       # build plantri, fetch 4D data
```

The write-up is in [`paper/`](paper/); build it with `scripts/build_paper.sh`.
It is the reference for the mathematics, and a large part of the test suite
exists to keep its claims in step with the committed data. The rest of this
page is one orientation pass over the problem, the method, the code and the
results. Commands are collected under [Running it](#running-it).

## The problem

The answer is known only for small `n`. In three dimensions it is settled for
`n <= 9` and for `n = 12` (the regular icosahedron); `n = 10` and `n = 11` are
numerical conjectures, and the `n = 11` volume provably has no closed form —
its minimal polynomial has degree 100. In four dimensions Horvath--Langi settle
`n = 6` outright; at `n = 7` they solve only the noncyclic case, and at `n = 8`
only that the regular cross-polytope is the best *cross-polytope*.

Two facts shape every method:

- **A maximiser is simplicial** (Berman--Hanes). Every facet is a simplex, so a
  polytope is described by a list of oriented facets plus vertex coordinates.
- **Vertices satisfy Property Z**: at a maximum each vertex sits at
  `p_i = n_i / |n_i|`, where `n_i` depends only on the *other* vertices.

## The approach

**Volume as a polynomial.** For an oriented simplicial sphere,

```
d! * vol = sum over facets f of det(p_{f_0}, ..., p_{f_{d-1}}).
```

This is a polynomial in the coordinates, and it is linear in each vertex
separately. That linearity yields `n_i`, hence the gradient, Property Z, and an
exact coordinate step `p_i = n_i / |n_i|` with no step size to tune. This is
why the optimiser has no learning rate and converges to machine precision.

**The trap.** The formula above is a *signed* volume. It is defined for any
point configuration, but it equals the real polytope volume only when the
points are in convex position with the given combinatorial type. Off that
region it can be *larger*. Ascent therefore happily converges to "folded"
configurations that are not polytopes at all, so every candidate is checked
against `ConvexPosition` before it counts. This is the single most important
thing to know before trusting any number here; see
[`paper/sections/trap.tex`](paper/sections/trap.tex).

**Two search modes.**

- *Type-free* (`HullAscent`): start from random points, recompute the convex
  hull each step, let the combinatorial type emerge. Fast, finds good optima,
  but proves nothing about types it never visited.
- *Per-type* (`TypeSweep`): enumerate **every** combinatorial type on `n`
  vertices and maximise inside each one. Types come from `plantri` in 3D and
  from Firsching's inscribed realisations in 4D. This is what turns a good
  guess into an exhaustive statement.

**The contribution: relaxation triage.** Because the signed-volume polynomial
is defined on all configurations, its maximum over a type is an *upper bound*
on that type's true geometric maximum:

```
max over convex-position configurations  <=  max of the signed polynomial.
```

So if a rival type's signed maximum already falls below the winner's volume, a
certified bound on that polynomial would eliminate the type outright — with no
convexity reasoning at all. Sorting rivals this way splits every case into:

- **excludable** — signed maximum below the winner, so a polynomial bound
  would finish the job;
- **hard** — signed maximum above the winner, so the relaxation is too loose
  and real geometry is needed.

The split is not uniform, and that is the interesting part. At `d = 4, n = 8`
all 36 rivals are excludable. At `d = 3, n = 9` the heptagonal bipyramid has a
convex value of `1.8243` but a signed maximum of `2.2748`, far above the
winner's `2.0438` — exactly one of the cases Hoehner--Ledford had to handle by
hand.

## How the code is organised

Dependencies run one way, with no cycles and no function-body imports:

```
algebra -> geometry -> volume -> optimality -> analysis -> search -> scripts
combinatorics -> {geometry, enumeration, analysis, store, viz}
store, catalog -> {viz, scripts}
```

The pieces you actually need to find things:

| area | what lives there |
| --- | --- |
| `volume.py`, `optimality.py` | the polynomial, its gradient, Property Z |
| `combinatorics/` | `OrientedComplex`, sphere validation, mediality |
| `enumeration/` | `plantri` (3D) and Firsching (4D) type sources |
| `search/` | `HullAscent`, `FixedTypeAscent`, `TypeSweep` |
| `analysis/` | convex-position and Property Z validators, symmetry, gauge |
| `store/` | `CaseResult`, `SearchRecord`, JSON round-trips |
| `viz/` | Schlegel projection and the three-view PNG renderer |

**Everything published is regenerated from committed data.** There are three
output trees:

- `results/` — one `CaseResult` per exhaustive sweep (13 cases: `d=3` at
  `n=4..12`, `d=4` at `n=5..8`), with the full per-type table and the winner's
  coordinates embedded.
- `results/free/` — one `SearchRecord` per type-free run (26 records: `d=3` at
  `n=4..20`, `d=4` at `n=5..13`), always heuristic.
- `images/` — one PNG per case, three orthographic views.

The paper's tables are transcribed by hand, so `tests/test_paper_tables.py` and
`tests/test_paper_method.py` read the numbers back out of the LaTeX and check
them against `results/`. A figure edited in the prose and not in the data, or
the reverse, fails the suite.

## Reading the results

**The evidence ladder** is the first thing to look at. It is *derived from the
data on every load*, never written by hand, so a case cannot claim more than
its own numbers support.

| level | meaning |
| --- | --- |
| E0 | a search converged to something; no completeness claim |
| E1 | every combinatorial type was maximised, but each by finite multistart |
| E2 | an explicit configuration with an exact volume; proves only `max >= V` |
| E3 | every rival type's signed maximum is bounded below the winner |
| E4 | a machine-checkable certificate |

Six cases reach E3: `d = 3` and `d = 4`, each at `n = 6, 7, 8`. **Nothing is at
E4.** The gap matters: `signed_max` is found by ascent, so it is a *lower*
bound on the true polynomial maximum. That direction is safe for ranking (a
type is only called excludable when even the value we found is below the
winner), but it identifies which types *could* be excluded rather than proving
they are. Closing that gap is the E4 step and is out of scope here.

Two further caveats to carry into any write-up:

- **`class-proven` in the 4D table is the published status**, meaning
  Horvath--Langi proved optimality inside a restricted class. Our sweeps range
  over every type, which is broader but numerical.
- **Symmetry orders are tolerance-dependent.** They are computed from the Gram
  matrix at `1e-7`, and these optima converge to about `1e-8`. Tighten the
  tolerance past the convergence level and every configuration reports the
  trivial group.

**The 4D story worth telling.** For `6 <= n <= 8` the sweep winner is the
convex hull of two regular polygons in orthogonal planes. That family cannot
stay optimal for large `n` — its volume saturates at `pi^2 / 6 = 1.645` while
the ball has `pi^2 / 2 = 4.935`.

## Running it

```bash
# exhaustive per-type sweep (dimension-independent)
python scripts/sweep_types.py 4 8 20               # S^3, n = 8, all 37 types
python scripts/sweep_types.py 3 9 40               # S^2, n = 9, all 50 types

# type-free search over a range of n
python scripts/sweep_free.py 3 4 12                # S^2
python scripts/sweep_free.py 4 5 10                # S^3
python scripts/sweep_free.py 3 4 20 --medial       # medial-conjecture check
python scripts/sweep_free.py 4 5 13 --save         # write results/free/

# targeted analyses
python scripts/orthogonal_pair_s3.py               # the 4D closed-form conjecture
python scripts/analyze.py 4 9                      # structure of one optimum
python scripts/validate.py                         # fixed-type solver vs known values

# maximise one combinatorial type of your own
python scripts/optimize.py mytype.json --save
python scripts/optimize.py --dimension 3 --vertices 9 --index 3 --save

# regenerate the committed artefacts
python scripts/build_images.py                     # images/, three views per case
python scripts/reproduce.py                        # re-run the reference sweeps
bash scripts/build_paper.sh                        # paper/main.pdf
```

Every script supports `--help`. `--save` writes the optimum as JSON under the
gitignored `output/` tree, and `--record` writes the full case to the committed
`results/` tree.

## Documentation

The paper is the documentation.

| question | where |
| --- | --- |
| why signed volume bites — **read this first** | [`paper/sections/trap.tex`](paper/sections/trap.tex) |
| the identity, Property Z, gauge fixing | [`paper/sections/preliminaries.tex`](paper/sections/preliminaries.tex) |
| what counts as a proof here, and the type sources | [`paper/sections/method.tex`](paper/sections/method.tex) |
| results in 3D / 4D | [`paper/sections/results-3d.tex`](paper/sections/results-3d.tex), [`paper/sections/results-4d.tex`](paper/sections/results-4d.tex) |

## References

Every entry the project cites is in
[paper/references.bib](paper/references.bib). The four that carry the headline
claims:

- J. D. Berman and K. Hanes, *Volumes of polyhedra inscribed in the unit sphere
  in E^3*, Math. Ann. 188 (1970), 78–84.
- S. Hoehner and J. Ledford, *The maximum volume polytope with nine vertices
  inscribed in the sphere*, [arXiv:2608.04392](https://arxiv.org/abs/2608.04392).
- A. G. Horvath and Z. Langi, *Maximum volume polytopes inscribed in the unit
  sphere*, Monatsh. Math. 181 (2016), 341–354.
- M. Firsching, *Realizability and inscribability for simplicial polytopes via
  nonlinear optimization*, Math. Program. 166 (2017), 273–295.

## License and citation

Apache 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The vendored plantri
is under the same licence, held by Brinkmann and McKay.

The licence asks only that you keep the notices. If you use the code or the
numbers in published work, please also cite it — see [CITATION.cff](CITATION.cff).
