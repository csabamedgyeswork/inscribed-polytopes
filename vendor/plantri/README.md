# plantri (vendored)

Exhaustive isomorph-free generator for planar triangulations, used by
`inscribed.enumeration.PlantriSource` to enumerate simplicial 2-spheres.

- Version 5.3, by Gunnar Brinkmann (Ghent) and Brendan McKay (ANU).
- Upstream: <https://users.cecs.anu.edu.au/~bdm/plantri/>
- Licensed under Apache 2.0; see `LICENSE-2.0.txt` and Appendix G of
  `plantri-guide.txt`.

Vendored rather than fetched so that the enumeration counts underpinning the
paper's three-dimensional sweeps stay reproducible if upstream moves.

Build with `python scripts/setup_data.py`, or directly:

```bash
cc -O3 -o vendor/plantri/plantri vendor/plantri/plantri.c
```

The compiled binary is gitignored.
